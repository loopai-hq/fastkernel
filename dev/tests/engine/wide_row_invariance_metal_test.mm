// Modified by meowkernels.
// Pulsar wide prompt lookup, row invariance at the kernel level: a wide lookup
// verifies one request's 16 or 32 rows as 2 or 4 lanes, and each row must
// keep the bytes the 8-row verify gives it. For each core class's planner
// and each Q4 projection shape of the 27B target's verify (the mixer input
// projections behind their RMS norm, which may write the projection's group
// sums), every 8-row tile of a 16- or 32-row plan must equal that tile run
// alone through the 1-lane plan, byte for byte, wherever
// Linear::sameRowReduction admits the width (QwenTarget::rowStableVerifyRows
// caps wide lookup at the widest such width). Widths it refuses are reported.
#include "AffineQ4Fixture.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace splash;
using splash::test::require;

constexpr uint32_t kRows = 8;
constexpr uint32_t kHidden = 5120;

struct Shape final {
  const char *name;
  ops::LinearMatrix matrix;
  ops::LinearEpilogue epilogue;
  bool mixerInput;
  ops::FloatOutput destination = ops::FloatOutput::BFloat16;
};

// The 27B target's verify projections (QwenTarget::rowStableVerifyRows).
constexpr std::array<Shape, 6> kShapes{{
    {"gdn input", {16640, kHidden}, ops::LinearEpilogue::None, true},
    {"attention qkv", {14336, kHidden}, ops::LinearEpilogue::None, true},
    {"mixer output", {kHidden, 6144}, ops::LinearEpilogue::Residual, false},
    {"gate/up", {17408, kHidden}, ops::LinearEpilogue::GateUp, false},
    {"down", {kHidden, 17408}, ops::LinearEpilogue::Residual, false},
    {"head", {248320, kHidden}, ops::LinearEpilogue::None, false, ops::FloatOutput::Float32},
}};

metal::MetalBuffer filled(metal::MetalBackend &backend, uint64_t elements, uint32_t seed, float scale) {
  metal::MetalBuffer buffer = test::sharedBuffer(backend, elements * 2);
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t i = 0; i < elements; ++i)
    values[i] = ops::tuning::floatToBf16(scale * (float(test::mix(uint32_t(i) + seed) % 2001) / 1000.0f - 1.0f));
  return buffer;
}

metal::MetalBuffer zeroed(metal::MetalBackend &backend, uint64_t bytes) {
  if (!bytes) return {};
  metal::MetalBuffer buffer = test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

struct Weights final {
  ops::Projection projection, gate;
};

// One run of `lanes` lanes over rows [first, first + 8 * lanes) of the inputs.
std::vector<uint8_t> run(metal::MetalBackend &backend, const ops::Linear &linear, const Shape &shape,
                         const Weights &weights, metal::MetalBuffer hidden, metal::MetalBuffer residual,
                         const ops::NormWeights &norm, uint32_t first, uint32_t lanes) {
  const uint32_t rows = lanes * kRows, n = shape.matrix.outputSize, k = shape.matrix.inputSize;
  const ops::LinearPlan plan = linear.decodePlan(weights.projection, lanes, shape.epilogue,
                                                 shape.epilogue == ops::LinearEpilogue::GateUp ? &weights.gate
                                                                                               : nullptr);
  const ops::LinearScratchSize size = linear.decodeScratchSize(weights.projection.shape());
  ops::LinearScratch scratch{zeroed(backend, size.input), zeroed(backend, std::max<uint64_t>(size.sums, uint64_t{k} / 64 * 32 * 4)),
                             zeroed(backend, size.partials), zeroed(backend, size.counters)};
  const uint64_t outputBytes = uint64_t{rows} * n * (shape.destination == ops::FloatOutput::Float32 ? 4 : 2);
  metal::MetalBuffer output = zeroed(backend, outputBytes);
  metal::MetalBuffer input = backend.view(hidden, uint64_t{first} * k * 2, uint64_t{rows} * k * 2);
  metal::CommandGraph graph;
  ops::PreparedInput prepared{};
  if (shape.mixerInput) {
    metal::MetalBuffer normalized = zeroed(backend, uint64_t{rows} * k * 2);
    prepared = ops::Normalization::addRms(graph, input, norm, normalized, k, rows, scratch, linear.normInput(plan));
    input = normalized;
  }
  ops::LinearBuffers buffers{.input = input, .output = output, .scratch = scratch, .prepared = prepared};
  if (shape.epilogue == ops::LinearEpilogue::Residual)
    buffers.residual = backend.view(residual, uint64_t{first} * n * 2, uint64_t{rows} * n * 2);
  if (shape.epilogue == ops::LinearEpilogue::GateUp)
    buffers.gateScratch = zeroed(backend, plan.gateScratchBytes());
  linear.add(graph, buffers, weights.projection, plan,
             shape.epilogue == ops::LinearEpilogue::GateUp ? &weights.gate : nullptr);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *bytes = static_cast<const uint8_t *>(output.contents());
  return {bytes, bytes + outputBytes};
}

}  // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: wide-row-invariance METALLIB");
    metal::MetalBackend backend(argv[1]);
    const DeviceCapabilities device = backend.capabilities();
    const metal::MetalBuffer hidden = filled(backend, uint64_t{4 * kRows} * 17408, 11, 2.0f);
    const metal::MetalBuffer residual = filled(backend, uint64_t{4 * kRows} * 248320, 23, 0.5f);
    const ops::NormWeights norm{filled(backend, 17408, 37, 0.5f)};
    uint32_t checked = 0;
    for (const Shape &shape : kShapes) {
      Weights weights{test::deterministicQ4Projection(backend, shape.matrix, 101),
                      shape.epilogue == ops::LinearEpilogue::GateUp
                          ? test::deterministicQ4Projection(backend, shape.matrix, 211)
                          : ops::Projection{}};
      weights.projection.destination = shape.destination;
      for (const uint32_t cores : {device.gpuCoreCount, 20U, 16U, 10U}) {
        DeviceCapabilities planned = device;
        planned.gpuCoreCount = cores;
        const ops::Linear linear(planned);
        std::vector<std::vector<uint8_t>> tiles;
        for (uint32_t tile = 0; tile < 4; ++tile)
          tiles.push_back(run(backend, linear, shape, weights, hidden, residual, norm, tile * kRows, 1));
        const ops::Projection *gate = shape.epilogue == ops::LinearEpilogue::GateUp ? &weights.gate : nullptr;
        const ops::LinearPlan one = linear.decodePlan(weights.projection, 1, shape.epilogue, gate);
        for (const uint32_t lanes : {2U, 4U}) {
          const ops::LinearPlan wide = linear.decodePlan(weights.projection, lanes, shape.epilogue, gate);
          const std::string label = std::string(shape.name) + " cores " + std::to_string(cores) + " rows " +
                                    std::to_string(lanes * kRows) + " (" + std::string(one.pipeline()) + " vs " +
                                    std::string(wide.pipeline()) + ")";
          if (!linear.sameRowReduction(one, wide, shape.mixerInput)) {
            std::cout << "wide row invariance: " << label << ": not row-stable, wide lookup capped\n";
            continue;
          }
          const std::vector<uint8_t> rows = run(backend, linear, shape, weights, hidden, residual, norm, 0, lanes);
          const uint64_t tileBytes = tiles[0].size();
          uint64_t differ = 0;
          for (uint32_t tile = 0; tile < lanes; ++tile)
            for (uint64_t byte = 0; byte < tileBytes; ++byte)
              differ += rows[tile * tileBytes + byte] != tiles[tile][byte];
          std::cout << "wide row invariance: " << label << ": " << (differ ? "DIFFER" : "identical") << " ("
                    << differ << " bytes)\n";
          require(differ == 0, label + ": a wide row differs from its 8-row verify");
          ++checked;
        }
      }
    }
    require(checked > 0, "no width was row-stable");
    std::cout << "wide_row_invariance_metal_test: PASS (" << checked << " widths byte-identical)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "wide_row_invariance_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
