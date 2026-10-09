// Modified by Pulsar.
#include "ops/GDN.hpp"

#include "metal/EnvSwitch.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

static_assert(offsetof(GDNDecodeBatchParams, conv_layer_bytes) == 8);

enum class KernelLayout : uint8_t { Value48, Value32 };

[[nodiscard]] KernelLayout kernelShape(const GdnShape &shape) {
  if (!shape.valid())
    throw std::invalid_argument("invalid GDN shape");
  if (shape == GdnShape{16, 48, 128, 10240, 16640})
    return KernelLayout::Value48;
  if (shape == GdnShape{16, 32, 128, 8192, 12544})
    return KernelLayout::Value32;
  throw std::invalid_argument("unsupported compiled GDN shape");
}

[[nodiscard]] const char *kernelName(KernelLayout shape,
                                     const char *value48,
                                     const char *value32) noexcept {
  return shape == KernelLayout::Value48 ? value48 : value32;
}

uint64_t valueWidth(const GdnShape &shape) { return uint64_t{shape.valueHeads} * shape.headDimension; }
// One layer's carried convolution rows and fp32 recurrent state, a head
// dimension square per value head.
uint64_t carriedBytes(const GdnShape &shape) {
  return uint64_t{SPLASH_GDN_CONVOLUTION_TAPS - 1} * shape.convolutionDimension * 2;
}
uint64_t recurrentBytes(const GdnShape &shape) { return valueWidth(shape) * shape.headDimension * sizeof(float); }

// The bytes of `rows` rows of the packed projection the kernels read: each
// row's q, k and v convolution inputs, z, then beta and alpha per value head.
uint64_t packedBytes(const GdnShape &shape, uint64_t rows) {
  return rowBytes(rows, shape.packedWidth, shape.convolutionDimension + valueWidth(shape) + 2 * shape.valueHeads, 2);
}

// The convolution weights of every channel and a_scale and dt_bias of every
// value head.
void requireMixerWeights(const GdnShape &shape, const metal::MetalBuffer &convolution,
                         const metal::MetalBuffer &decay, const metal::MetalBuffer &timeBias) {
  requireBytes(convolution, uint64_t{shape.convolutionDimension} * SPLASH_GDN_CONVOLUTION_TAPS * 2,
               "GDN convolution weight");
  requireBytes(decay, uint64_t{shape.valueHeads} * sizeof(float), "GDN decay weight");
  requireBytes(timeBias, uint64_t{shape.valueHeads} * 2, "GDN time bias");
}

// The fp32 decay and bf16 beta gates of `rows` rows of value heads.
void requireGates(const GdnShape &shape, uint64_t rows, const metal::MetalBuffer &decay,
                  const metal::MetalBuffer &beta) {
  requireBytes(decay, rows * shape.valueHeads * sizeof(float), "GDN decay");
  requireBytes(beta, rows * shape.valueHeads * 2, "GDN beta");
}

// Each running lane's current and next state cells reach the end of layer
// `layer`'s state: its carried rows from layer x convolutionLayerBytes, its
// recurrent state from convolutionStateBytes + layer x recurrentLayerBytes.
void requireStates(const GdnShape &shape, GdnStateStrides strides, uint32_t layer, uint32_t lanes,
                   std::span<const metal::MetalBuffer> current, std::span<const metal::MetalBuffer> next) {
  const uint64_t bytes =
      std::max(uint64_t{layer} * strides.convolutionLayerBytes + carriedBytes(shape),
               strides.convolutionStateBytes + uint64_t{layer} * strides.recurrentLayerBytes + recurrentBytes(shape));
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requireBytes(current[lane], bytes, "GDN current state");
    requireBytes(next[lane], bytes, "GDN next state");
  }
}

// SPLASH_GDN_VALUE_PARTS=4 (default; any other value turns it off): the
// one-lane GDN scan in four value partitions. Serving +1.07%, texts 18/18
// identical (Pulsar 1.0.0). Runs only under the fused split-K sums
// (GroupSums), whose finalize writes them.
[[nodiscard]] bool valueParts4Enabled() noexcept {
  static const bool enabled = metal::envSwitch("SPLASH_GDN_VALUE_PARTS", "4");
  return enabled;
}

// SPLASH_GDN_SCAN_NOSTORE (default on; exact; read per call, so an
// in-process A/B can toggle it between cycles): every VH48 verify scan (the
// value parts, the fused split sums and the batch scans with or without a
// table) stores neither its eight-row recurrent state nor the convolution
// carry, and the commit replays every lane's retained rows even when all
// eight are kept (verify_gdn_commit_replay). The replay's state is the
// scan's bit for bit (checked on 8-token cycles
// at B1 and B2). Per lane, a cycle that keeps fewer than eight rows skips one
// store of the whole recurrent state (151 MB on the 27B); one that keeps all
// eight replays them instead of keeping the store. The scans and the commit
// take the same predicate.
[[nodiscard]] bool scanSkipsStore(KernelLayout kernel) noexcept {
  return kernel == KernelLayout::Value48 && metal::envSwitch("SPLASH_GDN_SCAN_NOSTORE");
}
[[nodiscard]] std::string storeSuffix(std::string name, KernelLayout kernel) {
  return scanSkipsStore(kernel) ? name + "_nostore" : name;
}

// Pulsar's GDN kernels gate the grouped head order with bf16 norm weights.
void requireForkVariant(const GdnDecodeBuffers &buffers, const GdnShape &shape, GdnHeadOrder order,
                        const char *what) {
  if (order != GdnHeadOrder::Grouped || buffers.mixerNorm.float32)
    throw std::invalid_argument(std::string(what) + " takes the grouped head order and bf16 norm weights");
  requireBytes(buffers.mixerNorm.buffer, buffers.mixerNorm.bytes(shape.headDimension), "norm weight");
}

// The wide kernels' tile names: 16 rows keep their first/second names.
[[nodiscard]] std::string wideKernelName(KernelLayout kernel, const char *stage, uint32_t tile, uint32_t tiles) {
  std::string name = tiles == 2 ? std::string(stage) + "16_" + (tile ? "second" : "first")
                                : std::string(stage) + "32_t" + std::to_string(tile);
  return kernel == KernelLayout::Value48 ? name : name + "_vh32";
}

} // namespace

void GDN::addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                     GdnShape shape, uint32_t tokens, GdnHeadOrder order) {
  if (!tokens)
    throw std::invalid_argument("invalid GDN prefill geometry");
  const KernelLayout kernel = kernelShape(shape);
  const std::string gate = normKernel(kernelName(kernel, "prefill_gdn_gate", "prefill_gdn_gate_vh32"),
                                      buffers.mixerNorm, shape.headDimension);
  const uint64_t keyRows = uint64_t{tokens} * shape.keyHeads * shape.headDimension * 2;
  const uint64_t valueRows = tokens * valueWidth(shape) * 2;
  requireBytes(buffers.packed, packedBytes(shape, tokens), "GDN packed");
  requireMixerWeights(shape, buffers.convolutionWeights, buffers.decayWeights, buffers.timeBias);
  requireBytes(buffers.convolutionIn, carriedBytes(shape), "GDN convolution state");
  requireBytes(buffers.convolutionOut, carriedBytes(shape), "GDN next convolution state");
  requireBytes(buffers.queries, keyRows, "GDN query");
  requireBytes(buffers.keys, keyRows, "GDN key");
  requireBytes(buffers.values, valueRows, "GDN value");
  requireGates(shape, tokens, buffers.decay, buffers.beta);
  requireBytes(buffers.recurrentIn, recurrentBytes(shape), "GDN recurrent state");
  requireBytes(buffers.recurrentOut, recurrentBytes(shape), "GDN next recurrent state");
  requireBytes(buffers.recurrentRows, valueRows, "GDN recurrent row");
  requireBytes(buffers.hidden, valueRows, "GDN hidden");
  const GDNPrefillParams params{tokens};
  graph.add(kernelName(kernel, "prefill_gdn_prepare",
                       "prefill_gdn_prepare_vh32"),
            {buffers.packed, buffers.convolutionWeights,
             buffers.convolutionIn, buffers.convolutionOut, buffers.queries,
             buffers.keys, buffers.values, buffers.decayWeights,
             buffers.timeBias, buffers.decay, buffers.beta},
            params, {uint64_t{tokens} * shape.keyHeads, 1, 1},
            {shape.headDimension, 1, 1});
  graph.add(kernelName(kernel, "prefill_gdn_scan",
                       "prefill_gdn_scan_vh32"),
            {buffers.queries, buffers.keys, buffers.values, buffers.decay,
             buffers.beta, buffers.recurrentIn, buffers.recurrentOut,
             buffers.recurrentRows},
            params,
            {uint64_t{shape.valueHeads} * shape.headDimension /
                 SPLASH_GDN_SCAN_STATE_ROWS,
             1, 1},
            {SPLASH_GDN_SCAN_THREADS, 1, 1});
  graph.add(gate,
            {buffers.recurrentRows, buffers.packed, buffers.mixerNorm.buffer,
             buffers.hidden},
            GDNGatePrefillParams{order == GdnHeadOrder::Tiled},
            {uint64_t{tokens} * shape.valueHeads, 1, 1}, {128, 1, 1});
}

// SPLASH_GDN_FUSED_SUMS (default on): the one-lane GDN output kernel writes
// the split-K out-projection's input sums the separate dispatch wrote
// (byte-exact gate); texts 18/18 identical, ms/cycle -0.083 (Pulsar 1.0.0).
LinearInput GDN::outputInput(const LinearPlan &plan, GdnShape shape, GdnHeadOrder order, const NormWeights &norm) {
  static const bool fused = metal::envSwitch("SPLASH_GDN_FUSED_SUMS");
  const bool groupSums = fused && plan.configuration().tile == LinearTile::Split32PrecomputedSums &&
                         plan.workload().rows == SPLASH_TARGET_VERIFY_ROWS &&
                         shape == GdnShape{16, 48, 128, 10240, 16640} && order == GdnHeadOrder::Grouped &&
                         !norm.float32;
  return groupSums ? LinearInput::GroupSums : plan.input();
}

PreparedInput GDN::addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                             GdnShape shape, uint32_t lanes, uint32_t layer,
                             GdnStateStrides state, GdnHeadOrder order, LinearInput input) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH || !state.valid())
    throw std::invalid_argument("invalid GDN decode geometry");
  const KernelLayout kernel = kernelShape(shape);
  // Lane l's rows of the packed, mixed, gate and hidden rows are rows
  // [8 l, 8 l + 8).
  const uint64_t rows = uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(buffers.packed, packedBytes(shape, rows), "GDN packed");
  requireMixerWeights(shape, buffers.convolutionWeights, buffers.decayWeights, buffers.timeBias);
  requireBytes(buffers.mixed, rows * shape.convolutionDimension * 2, "GDN mixed");
  requireGates(shape, rows, buffers.decay, buffers.beta);
  requireBytes(buffers.hidden, rows * valueWidth(shape) * 2, "GDN hidden");
  const GDNDecodeBatchParams params{order == GdnHeadOrder::Tiled,
                                    layer,
                                    state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  if (buffers.defer.on && (lanes != 1 || !deferRoute(shape, input)))
    throw std::invalid_argument("a deferred GDN commit takes the one-lane value-parts route");
  if (input == LinearInput::GroupSums) {
    if (lanes != 1 || kernel != KernelLayout::Value48 || buffers.currentStates.size() != SPLASH_MAXIMUM_BATCH_WIDTH ||
        buffers.nextStates.size() != SPLASH_MAXIMUM_BATCH_WIDTH)
      throw std::invalid_argument("GDN group sums take one lane of the VH48 variant and every lane's bindings");
    requireForkVariant(buffers, shape, order, "GDN group sums");
    requireStates(shape, state, layer, lanes, buffers.currentStates, buffers.nextStates);
    // Each row's fp32 sum per 64 outputs, [group][row].
    requireBytes(buffers.linearScratch.sums, valueWidth(shape) / 64 * rows * sizeof(float), "GDN split sums");
    if (valueParts4Enabled()) {
      if (const GdnDeferScan &defer = buffers.defer; defer.on) {
        if (defer.rows > SPLASH_TARGET_VERIFY_ROWS)
          throw std::invalid_argument("too many pending GDN rows");
        requireBytes(defer.base, state.convolutionStateBytes + uint64_t{layer + 1} * state.recurrentLayerBytes,
                     "GDN deferred base cell");
        requireBytes(defer.mixed, SPLASH_TARGET_VERIFY_ROWS * shape.convolutionDimension * 2, "GDN pending mixed");
        requireGates(shape, SPLASH_TARGET_VERIFY_ROWS, defer.decay, defer.beta);
        const GDNDeferParams deferParams{layer, defer.rows, 0, order == GdnHeadOrder::Tiled,
                                         state.convolutionLayerBytes, state.recurrentLayerBytes,
                                         state.convolutionStateBytes};
        graph.add("verify_gdn_value_parts4_scan_defer",
                  {buffers.packed, buffers.convolutionWeights, buffers.currentStates[0], defer.base,
                   buffers.mixed, buffers.decayWeights, buffers.timeBias, buffers.decay, buffers.beta,
                   buffers.hidden, defer.mixed, defer.decay, defer.beta},
                  deferParams, {uint64_t{shape.valueHeads} * 4, 1, 1});
      } else {
        graph.add(storeSuffix("verify_gdn_value_parts4_scan", kernel),
                  {buffers.packed, buffers.convolutionWeights, buffers.currentStates[0], buffers.nextStates[0],
                   buffers.mixed, buffers.decayWeights, buffers.timeBias, buffers.decay, buffers.beta,
                   buffers.hidden},
                  params, {uint64_t{shape.valueHeads} * 4, 1, 1});
      }
      graph.add("verify_gdn_value_parts4_finalize",
                {buffers.packed, buffers.mixerNorm.buffer, buffers.hidden, buffers.linearScratch.sums},
                {shape.valueHeads, 1, 1});
    } else {
      std::vector<metal::MetalBuffer> bindings{buffers.packed, buffers.convolutionWeights};
      bindings.reserve(18);
      appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
      bindings.insert(bindings.end(), {buffers.mixed, buffers.decayWeights, buffers.timeBias, buffers.decay,
                                       buffers.beta, buffers.mixerNorm.buffer, buffers.hidden,
                                       buffers.linearScratch.sums});
      graph.add(storeSuffix("verify_gdn_fused_split_sums", kernel), std::move(bindings), params,
                {shape.valueHeads, 1, 1});
    }
    return {buffers.hidden, LinearInput::GroupSums};
  }
  std::vector<metal::MetalBuffer> bindings{buffers.packed,
                                           buffers.convolutionWeights};
  const bool prepare = input != LinearInput::Plain;
  if (prepare)
    requireTableScratch(buffers.linearScratch, input, shape.valueHeads * shape.headDimension,
                        lanes * SPLASH_TARGET_VERIFY_ROWS);
  bindings.reserve(prepare ? 19 : 17);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  requireStates(shape, state, layer, lanes, buffers.currentStates, buffers.nextStates);
  bindings.insert(bindings.end(),
                  {buffers.mixed, buffers.decayWeights, buffers.timeBias,
                   buffers.decay, buffers.beta, buffers.mixerNorm.buffer,
                   buffers.hidden});
  if (prepare)
    bindings.insert(bindings.end(), {buffers.linearScratch.input, buffers.linearScratch.sums});
  const std::string name = std::string("verify_gdn_fused") + tableSuffix(input) + kernelName(kernel, "", "_vh32");
  graph.add(storeSuffix(normKernel(name, buffers.mixerNorm, shape.headDimension), kernel), std::move(bindings),
            params, {shape.valueHeads, lanes, 1});
  if (!prepare) return {};
  return {buffers.hidden, input};
}

namespace {

// The commits of every lane (verify_gdn_commit and its variants).
void addCommitAs(metal::CommandGraph &graph, GdnCommitBuffers buffers, GdnShape shape, uint32_t layers,
                 uint32_t lanes, GdnStateStrides state, const char *name) {
  if (!layers || !lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH ||
      !state.valid())
    throw std::invalid_argument("invalid GDN commit geometry");
  // The decoded rows of lane l in layer y start at row (y x the maximum batch
  // width + l) x 8 of the packed, mixed and gate rows. A lane replays at most
  // the 7 rows it retains (all 8 leave the decoded state) and carries its
  // last three retained inputs, within those rows.
  const uint64_t rows =
      (uint64_t{layers - 1} * SPLASH_MAXIMUM_BATCH_WIDTH + lanes - 1) * SPLASH_TARGET_VERIFY_ROWS +
      SPLASH_TARGET_VERIFY_ROWS - 1;
  requireBytes(buffers.packed, rowBytes(rows, shape.packedWidth, shape.convolutionDimension, 2), "GDN packed");
  requireBytes(buffers.mixed, rows * shape.convolutionDimension * 2, "GDN mixed");
  requireGates(shape, rows, buffers.decay, buffers.beta);
  requireBytes(buffers.retainedCounts, uint64_t{lanes} * sizeof(uint32_t), "GDN retained counts");
  std::vector<metal::MetalBuffer> bindings{
      buffers.packed, buffers.mixed, buffers.decay, buffers.beta};
  bindings.reserve(13);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  requireStates(shape, state, layers - 1, lanes, buffers.currentStates, buffers.nextStates);
  bindings.push_back(buffers.retainedCounts);
  const GDNBatchCommitParams params{state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  graph.add(name, std::move(bindings), params, {shape.valueHeads, layers, lanes});
}

} // namespace

void GDN::addCommit(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                    GdnShape shape, uint32_t layers, uint32_t lanes,
                    GdnStateStrides state) {
  const KernelLayout kernel = kernelShape(shape);
  addCommitAs(graph, std::move(buffers), shape, layers, lanes, state,
              scanSkipsStore(kernel) ? "verify_gdn_commit_replay"
                                     : kernelName(kernel, "verify_gdn_commit", "verify_gdn_commit_vh32"));
}

bool GDN::deferRoute(GdnShape shape, LinearInput input) noexcept {
  return kernelShape(shape) == KernelLayout::Value48 && input == LinearInput::GroupSums && valueParts4Enabled();
}

void GDN::addCommitConv(metal::CommandGraph &graph, GdnCommitBuffers buffers, GdnShape shape, uint32_t layers,
                        GdnStateStrides state) {
  if (kernelShape(shape) != KernelLayout::Value48)
    throw std::invalid_argument("a deferred GDN commit takes the VH48 variant");
  addCommitAs(graph, std::move(buffers), shape, layers, 1, state, "verify_gdn_commit_conv");
}

void GDN::addFlush(metal::CommandGraph &graph, GdnFlushBuffers buffers, GdnShape shape, uint32_t layers,
                   uint32_t rows, uint32_t slot, GdnStateStrides state) {
  if (kernelShape(shape) != KernelLayout::Value48 || !layers || !rows || rows > SPLASH_TARGET_VERIFY_ROWS ||
      slot >= SPLASH_MAXIMUM_BATCH_WIDTH || !state.valid())
    throw std::invalid_argument("invalid deferred GDN flush");
  const uint64_t scratchRows = uint64_t{layers} * SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(buffers.mixed, scratchRows * shape.convolutionDimension * 2, "GDN pending mixed");
  requireGates(shape, scratchRows, buffers.decay, buffers.beta);
  const uint64_t cell = state.convolutionStateBytes + uint64_t{layers} * state.recurrentLayerBytes;
  requireBytes(buffers.base, cell, "GDN flush base cell");
  requireBytes(buffers.out, cell, "GDN flush cell");
  const GDNDeferParams params{0, rows, slot, 0, state.convolutionLayerBytes, state.recurrentLayerBytes,
                              state.convolutionStateBytes};
  graph.add("verify_gdn_commit_flush", {buffers.mixed, buffers.decay, buffers.beta, buffers.base, buffers.out},
            params, {shape.valueHeads, layers, 1});
}

void GDN::addDecode16(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                      metal::MetalBuffer convolutionScratch, GdnShape shape,
                      uint32_t layer, GdnStateStrides state, GdnHeadOrder order,
                      uint32_t tiles, WideGdn route) {
  if ((tiles != 2 && tiles != 4) || !state.valid() || buffers.currentStates.empty() ||
      buffers.nextStates.empty() || buffers.linearScratch.input || buffers.linearScratch.sums)
    throw std::invalid_argument("invalid wide GDN decode geometry");
  const KernelLayout kernel = kernelShape(shape);
  requireForkVariant(buffers, shape, order, "wide GDN");
  // The request's packed, mixed, gate and hidden rows are those of physical
  // lanes 0..tiles-1.
  const uint64_t rows = uint64_t{tiles} * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(buffers.packed, packedBytes(shape, rows), "GDN packed");
  requireMixerWeights(shape, buffers.convolutionWeights, buffers.decayWeights, buffers.timeBias);
  requireBytes(buffers.mixed, rows * shape.convolutionDimension * 2, "GDN mixed");
  requireGates(shape, rows, buffers.decay, buffers.beta);
  requireBytes(buffers.hidden, rows * valueWidth(shape) * 2, "GDN hidden");
  requireStates(shape, state, layer, 1, buffers.currentStates, buffers.nextStates);
  requireBytes(convolutionScratch, gdnDecode16ConvolutionScratchBytes(state, tiles),
               "wide GDN convolution scratch");
  const GDNDecodeBatchParams params{0, layer, state.convolutionLayerBytes, state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  // SPLASH_WIDE_GDN_SINGLE=parts: the single pass as 4 value parts per head,
  // then a finalize for the complete-head gated norm (VH48 only).
  if (route == WideGdn::SingleParts && kernel == KernelLayout::Value48) {
    const std::string name = tiles == 2 ? "verify_gdn_wide16_parts4" : "verify_gdn_wide32_parts4";
    graph.add(name + "_scan",
              {buffers.packed, buffers.convolutionWeights, buffers.currentStates[0], buffers.nextStates[0],
               buffers.mixed, buffers.decayWeights, buffers.timeBias, buffers.decay, buffers.beta,
               buffers.hidden},
              params, {uint64_t{shape.valueHeads} * 4, 1, 1});
    graph.add(name + "_finalize", {buffers.packed, buffers.mixerNorm.buffer, buffers.hidden},
              {shape.valueHeads, 1, 1});
    return;
  }
  const std::vector<metal::MetalBuffer> bindings{
      buffers.packed,          buffers.convolutionWeights, buffers.currentStates[0],
      buffers.nextStates[0],   buffers.mixed,              buffers.decayWeights,
      buffers.timeBias,        buffers.decay,              buffers.beta,
      buffers.mixerNorm.buffer, buffers.hidden,            convolutionScratch};
  // SPLASH_WIDE_GDN_SINGLE: all tiles in one dispatch (current -> next).
  if (route != WideGdn::Chain) {
    const std::string name = tiles == 2 ? "verify_gdn_wide16" : "verify_gdn_wide32";
    graph.add(kernel == KernelLayout::Value48 ? name : name + "_vh32", bindings, params,
              {shape.valueHeads, 1, 1});
    return;
  }
  for (uint32_t tile = 0; tile < tiles; ++tile)
    graph.add(wideKernelName(kernel, "verify_gdn_fused", tile, tiles), bindings, params,
              {shape.valueHeads, 1, 1});
}

void GDN::addCommit16(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                      metal::MetalBuffer convolutionScratch, GdnShape shape,
                      uint32_t layers, GdnStateStrides state, uint32_t tiles) {
  if ((tiles != 2 && tiles != 4) || !layers || !state.valid() || buffers.currentStates.empty() ||
      buffers.nextStates.empty())
    throw std::invalid_argument("invalid wide GDN commit geometry");
  const KernelLayout kernel = kernelShape(shape);
  // Tile t of layer y replays the rows of physical lane t from row
  // (y x the maximum batch width + t) x 8: all 8 of a tile before the last,
  // at most the 7 a request retains short of its rows in the last.
  const uint64_t rows =
      (uint64_t{layers - 1} * SPLASH_MAXIMUM_BATCH_WIDTH + tiles - 1) * SPLASH_TARGET_VERIFY_ROWS +
      SPLASH_TARGET_VERIFY_ROWS - 1;
  requireBytes(buffers.packed, rowBytes(rows, shape.packedWidth, shape.convolutionDimension, 2), "GDN packed");
  requireBytes(buffers.mixed, rows * shape.convolutionDimension * 2, "GDN mixed");
  requireGates(shape, rows, buffers.decay, buffers.beta);
  requireBytes(buffers.retainedCounts, sizeof(uint32_t), "GDN retained counts");
  requireStates(shape, state, layers - 1, 1, buffers.currentStates, buffers.nextStates);
  requireBytes(convolutionScratch, gdnCommit16ConvolutionScratchBytes(state, tiles),
               "wide GDN convolution scratch");
  const std::vector<metal::MetalBuffer> bindings{
      buffers.packed,           buffers.mixed,         buffers.decay,
      buffers.beta,             buffers.currentStates[0], buffers.nextStates[0],
      convolutionScratch,       buffers.retainedCounts};
  const GDNBatchCommitParams params{state.convolutionLayerBytes, state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  for (uint32_t tile = 0; tile < tiles; ++tile)
    graph.add(wideKernelName(kernel, "verify_gdn_commit", tile, tiles), bindings, params,
              {shape.valueHeads, layers, 1});
}

} // namespace splash::ops
