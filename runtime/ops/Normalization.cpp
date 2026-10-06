// Modified by meowkernels.
#include "Normalization.hpp"

#include "metal/EnvSwitch.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/BufferExtent.hpp"

#include <utility>
#include <stdexcept>

namespace splash::ops {
namespace {

// norm_rms_staged_wide and norm_rms_staged_split_sums stage a row of up to
// SPLASH_STAGED_NORM_WIDE_WIDTH bf16 values in threadgroup memory and bind
// bfloat4 views (8-byte aligned buffers, as every arena and weight section
// is), with the 1024-thread dispatch every supported GPU family admits.
// SPLASH_STAGED_NORM_WIDE (default on), read per process: plain bf16 norms
// wider than upstream's staged region run norm_rms_staged_wide, the same
// bits as norm_rms (fastkernel 1.0.0, M5 Max 40 vs 1.0.x's norm_rms: norm
// 1.80 -> 0.69 ms per B1 cycle at 5K; upstream's 1.3.0 norm_rms is faster
// itself, so this is to be re-measured).
bool stagedWide(const NormWeights &weight, uint32_t width) noexcept {
  static const bool on = metal::envSwitch("SPLASH_STAGED_NORM_WIDE");
  return on && !weight.float32 && width <= SPLASH_STAGED_NORM_WIDE_WIDTH && width % 4 == 0;
}

} // namespace

std::string normKernel(std::string_view name, const NormWeights &weights, uint32_t width) {
  requireBytes(weights.buffer, weights.bytes(width), "norm weight");
  return std::string(name) + (weights.float32 ? "_f32" : "");
}

PreparedInput Normalization::addRms(metal::CommandGraph &graph,
                                    metal::MetalBuffer input,
                                    const NormWeights &weight,
                                    metal::MetalBuffer output, uint32_t width,
                                    uint32_t rows, LinearScratch scratch,
                                    LinearInput layout) {
  const uint64_t bytes = uint64_t{rows} * width * 2;
  requireBytes(input, bytes, "norm input");
  requireBytes(output, bytes, "norm output");
  // fastkernel's fused input sums: the outputs and each row's sums per
  // 64-input group, which the consumer otherwise computes itself.
  if (layout == LinearInput::GroupSums) {
    if (!weight.float32 && width <= SPLASH_STAGED_NORM_WIDE_WIDTH && width % 64 == 0 &&
        scratch.sums.sizeBytes() >= uint64_t{width / 64} * rows * sizeof(float)) {
      graph.add(normKernel("norm_rms_staged_split_sums", weight, width),
                {std::move(input), weight.buffer, output, scratch.sums}, width, {rows, 1, 1},
                {SPLASH_STAGED_NORM_THREADS, 1, 1});
      return {std::move(output), LinearInput::GroupSums};
    }
    layout = LinearInput::Plain;
  }
  if (layout != LinearInput::Plain) {
    requireTableScratch(scratch, layout, width, rows);
    graph.add(normKernel(std::string("norm_rms") + tableSuffix(layout) + "_decode", weight, width),
              {input, weight.buffer, output, scratch.input, scratch.sums}, width, {rows, 1, 1});
    return {std::move(output), layout};
  }
  if (rows <= SPLASH_STAGED_NORM_ROWS && width <= SPLASH_STAGED_NORM_WIDTH && width % 4 == 0)
    graph.add(normKernel("norm_rms_staged", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1}, {SPLASH_STAGED_NORM_THREADS, 1, 1});
  else if (stagedWide(weight, width))
    graph.add(normKernel("norm_rms_staged_wide", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1}, {SPLASH_STAGED_NORM_THREADS, 1, 1});
  else
    graph.add(normKernel("norm_rms", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1});
  return {};
}

void Normalization::addRmsWithQ4Sums(
    metal::CommandGraph &graph, metal::MetalBuffer input,
    const NormWeights &weight, metal::MetalBuffer output,
    metal::MetalBuffer sums, uint32_t width, uint32_t rows) {
  if (weight.float32 || !rows || !width || width % 64)
    throw std::invalid_argument("the Q4-sum norm takes bf16 weights and whole 64-input groups");
  const uint64_t bytes = uint64_t{rows} * width * 2;
  requireBytes(input, bytes, "norm input");
  requireBytes(output, bytes, "norm output");
  // The sums are [32-row tile][64-input group][row of the tile]: the last
  // row's sum of the last group ends them.
  const uint64_t groups = width / 64, last = rows - 1;
  requireBytes(sums, ((last / 32 * groups + groups - 1) * 32 + last % 32 + 1) * sizeof(float), "norm sums");
  graph.add(normKernel("prefill_norm_rms_sums32", weight, width),
            {std::move(input), weight.buffer, std::move(output),
             std::move(sums)},
            width, {rows, 1, 1});
}

} // namespace splash::ops
