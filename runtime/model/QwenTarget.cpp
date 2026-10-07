// Modified by meowkernels.
#include "model/QwenTarget.hpp"

#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/WeightStore.hpp"
#include "metal/EnvSwitch.hpp"
#include "ops/AneFfn.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace splash::model {
namespace {

template <class Weights>
void requireWeights(const Weights &weights,
                    const QwenTargetGeometry &geometry) {
  const uint32_t attentionLayers = static_cast<uint32_t>(std::count_if(
      weights.layers.begin(), weights.layers.end(), [](const auto &layer) {
        return std::holds_alternative<QwenAttentionWeights>(layer.mixer);
      }));
  if (!geometry.valid() || weights.layers.size() != geometry.layers ||
      attentionLayers != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument(
        "Qwen target weights do not match execution geometry");
  }
}

} // namespace

template <class Layout, class Layer>
QwenTarget::QwenTarget(const QwenTargetWeights<Layout, Layer> &weights,
                       const QwenTargetGeometry &geometry,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), weightsBase_(weights), geometry_(geometry),
      backend_(backend), operators_(operators) {
  requireWeights(weights, geometry_);
}

template QwenTarget::QwenTarget(const Qwen3_8Weights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template QwenTarget::QwenTarget(const Qwen3_6MoeWeights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);

namespace {

void includeProjection(QwenTargetGeometry &geometry, const ops::Projection &projection) {
  geometry.decodeProjections.push_back(projection.shape());
}

// The projections each layer's FFN dispatches, from the first layer on.
void includeFfn(QwenTargetGeometry &geometry, const Qwen3_8LayerWeights &layer, bool) {
  if (layer.gateProjection.shape() != layer.upProjection.shape())
    throw WeightStoreError("fused gate/up projections must have matching shapes and layouts");
  includeProjection(geometry, layer.gateProjection);
  includeProjection(geometry, layer.upProjection);
  includeProjection(geometry, layer.downProjection);
  geometry.gateUpProjections.push_back(layer.upProjection.shape());
}
// No source mixes MoE layouts, so one plan runs every block of a step.
void includeFfn(QwenTargetGeometry &geometry, const Qwen3_6MoeLayerWeights &layer, bool first) {
  if (first) geometry.moeLayout = layer.ffn.layout();
  if (layer.ffn.layout() != geometry.moeLayout)
    throw WeightStoreError("the MoE blocks of a target must share one weight layout");
}

// The format of most routed expert weights of a GGUF target's MoE blocks,
// GGUF_FMT_COUNT for none (ops::MoeShape::expertFormat).
uint32_t routedExpertFormat(std::span<const Qwen3_8LayerWeights>) { return GGUF_FMT_COUNT; }
uint32_t routedExpertFormat(std::span<const Qwen3_6MoeLayerWeights> layers) {
  std::array<uint64_t, GGUF_FMT_COUNT> weights{};
  for (const Qwen3_6MoeLayerWeights &layer : layers) {
    if (layer.ffn.layout() != ops::WeightLayout::Block32) return GGUF_FMT_COUNT;
    const ops::BlockMoeWeights &block = layer.ffn.blocks();
    for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
      if (!projection->routed.isFloat())
        weights[projection->routed.formatId] += uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
  }
  const auto most = std::max_element(weights.begin(), weights.end());
  return *most ? uint32_t(most - weights.begin()) : GGUF_FMT_COUNT;
}

} // namespace

template <class Layout, class Layer>
QwenTargetGeometry qwenTargetGeometry(const QwenTargetWeights<Layout, Layer> &weights) {
  const Layout &layout = weights.layout;
  static_assert(std::tuple_size_v<decltype(Layout::hiddenCaptureLayers)> <=
                QwenTargetGeometry::maximumCaptureLayers);
  QwenTargetGeometry geometry(layout);
  geometry.captureLayerCount = static_cast<uint32_t>(layout.hiddenCaptureLayers.size());
  std::copy(layout.hiddenCaptureLayers.begin(), layout.hiddenCaptureLayers.end(),
            geometry.captureLayerValues.begin());
  geometry.kvLayout = layout.kvLayout();
  geometry.stateLayout = layout.gdnStateLayout();
  for (const auto &layer : weights.layers) {
    std::visit([&](const auto &mixer) {
      includeProjection(geometry, mixer.inputProjection);
      includeProjection(geometry, mixer.outputProjection);
    }, layer.mixer);
    includeFfn(geometry, layer, &layer == &weights.layers.front());
  }
  geometry.moeExpertFormat = routedExpertFormat(weights.layers);
  geometry.prefillProjections = geometry.decodeProjections;
  includeProjection(geometry, weights.logitsProjection);
  for (auto *shapes : {&geometry.prefillProjections, &geometry.decodeProjections,
                       &geometry.gateUpProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
  return geometry;
}

template QwenTargetGeometry qwenTargetGeometry(const Qwen3_8Weights &);
template QwenTargetGeometry qwenTargetGeometry(const Qwen3_6MoeWeights &);

const ops::Projection &QwenTarget::vocabularyProjection() const noexcept {
  return weightsBase_.logitsProjection;
}

uint32_t QwenTarget::decodeStorageLanes(uint32_t lanes) const {
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  uint32_t storageRows = rows;
  for (const auto &shape : geometry_.decodeProjections)
    storageRows = std::max(storageRows, operators_.linear().decodeStorageRows(rows, shape));
  return storageRows / ExecutionLimits::targetVerifyRows;
}

namespace {

// Rows [begin, begin + count) of a row-major buffer of `width` values of T.
template <class T>
metal::MetalBuffer rowsOf(metal::MetalBackend &backend, const metal::MetalBuffer &buffer, uint32_t begin,
                          uint32_t count, uint32_t width) {
  return backend.view(buffer, uint64_t{begin} * width * sizeof(T), uint64_t{count} * width * sizeof(T));
}

void requireLayerPartition(const QwenTargetGeometry &geometry, uint32_t gdnLayers, uint32_t attentionLayers) {
  if (gdnLayers != geometry.stateLayout.layers || attentionLayers != geometry.kvLayout.attentionLayers)
    throw std::logic_error("Qwen target layer partition mismatch");
}

// Copies `rows` rows of a capture layer's output, from row `sourceRow`, into
// capture slot `slot` of the captured hidden rows from row `destinationRow`.
void addCapture(metal::CommandGraph &graph, const QwenTargetGeometry &geometry, uint32_t slot,
                metal::MetalBuffer output, uint32_t sourceRow, metal::MetalBuffer captured,
                uint32_t destinationRow, uint32_t rows) {
  const uint32_t width = geometry.hiddenSize, capturedWidth = geometry.capturedHiddenSize();
  if (slot >= capturedWidth / width)
    throw std::logic_error("Qwen target capture slot past the captured hidden rows");
  ops::RowCopy::add(graph, std::move(output), {sourceRow, width, 0}, std::move(captured),
                    {destinationRow, capturedWidth, slot * width}, rows, width);
}

} // namespace

// The state a prefill command's layers share: its inputs and the next GDN
// and attention layer of the step.
struct QwenTarget::PrefillStep {
  metal::CommandGraph &graph;
  const QwenTargetPrefillBuffers &buffers;
  std::span<const QwenTargetPrefillSequence> sequences;
  uint32_t rows;
  std::span<const SplashKvLayer> kvLayers;
  // Each sequence's attention plan, which every attention layer runs.
  std::vector<ops::PrefillAttentionPlan> attention{};
  std::optional<ops::MoePlan> moe{};
  ops::AneFfn *aneFfn = nullptr;
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
};

struct QwenTarget::VerifyStep {
  metal::CommandGraph &graph;
  const QwenTargetVerifyBuffers &buffers;
  std::span<const SplashKvLayer> kvLayers;
  std::span<const kv::ChunkedPrefillParams> chunks;
  uint32_t lanes;
  uint32_t rows;
  ops::VerifyAttentionPlan attention;
  std::optional<ops::MoePlan> moe{};
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
  // SPLASH_SPLIT4_M16 (default on; read at every encode so one binary serves
  // both arms of an A/B): a two-request (B2) verify's 16-row split-K residual
  // bodies (mixer out, FFN down) take the M8 footer and metadata hoist. B2
  // lockstep -0.216 ms per cycle; byte-identical (fastkernel 1.0.0).
  bool m16HoistFooter = false;
  // fastkernel wide prompt lookup (addVerify16): one request's aliased lanes.
  // Its GDN layers run the wide decode; like 1.0.0 it keeps the plain residual
  // bodies and the separate down-projection sums (both byte-identical).
  bool wide = false;
  metal::MetalBuffer convolutionScratch{};
  ops::WideGdn gdnRoute = ops::WideGdn::Chain;
};

metal::MetalBuffer QwenTarget::addPrefill(
    metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const SplashKvLayer> kvLayers, ops::AneFfn *aneFfn) const {
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen ragged prefill batch");
  }
  for (const QwenTargetPrefillSequence &sequence : sequences) {
    if (sequence.convolutionIn.size() != geometry_.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry_.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry_.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry_.stateLayout.layers) {
      throw std::invalid_argument("Qwen prefill state layer mismatch");
    }
  }
  PrefillStep step{graph, buffers, sequences, rows, kvLayers};
  for (const QwenTargetPrefillSequence &sequence : sequences)
    step.attention.push_back(operators_.prefillAttention(
        sequence.rows, geometry_.attentionQueryHeads, geometry_.kvLayout));
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moePrefill(geometry_.moeShape(), rows);
  if (aneFfn && aneFfn->splits(rows)) step.aneFfn = aneFfn;
  std::visit([&](const auto *weights) {
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      const auto &layer = weights->layers[index];
      const metal::MetalBuffer input = buffers.hidden[index & 1];
      const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
      const metal::MetalBuffer residual = std::visit(
          [&](const auto &mixer) { return addPrefillMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
      addPrefillFfn(step, index, layer, residual, output);
      if (const auto slot = geometry_.captureSlot(index))
        for (const QwenTargetPrefillSequence &sequence : sequences)
          for (uint32_t capture = 0; capture < sequence.captureCount; ++capture) {
            const QwenTargetPrefillCapture &c = sequence.captures[capture];
            addCapture(graph, geometry_, *slot, output, c.sourceStart, buffers.captured, c.destinationStart,
                       c.rows);
          }
    }
  }, weights_);
  requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  return buffers.hidden[geometry_.layers & 1];
}

// An affine prefill projection reads the Q4 input sums of its rows, which the
// norm writes beside them; a block projection reads none.
void QwenTarget::addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm,
                                ops::WeightLayout consumer) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  if (consumer == ops::WeightLayout::Affine64)
    ops::Normalization::addRmsWithQ4Sums(step.graph, input, norm, b.normalized, b.projectionSums,
                                         geometry_.hiddenSize, step.rows);
  else
    ops::Normalization::addRms(step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows);
}

// The mixer output projection adds the mixer's rows to `input`.
void QwenTarget::addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                                  metal::MetalBuffer input, metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  if (projection.layout() == ops::WeightLayout::Affine64)
    operators_.linear().addPrefillSums(step.graph, hidden, b.projectionSums, projection, step.rows);
  operators_.linear().addPrefillResidual(step.graph, hidden, projection, input, output, b.projectionSums,
                                         step.rows, b.linearScratch);
}

metal::MetalBuffer QwenTarget::addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.gdnLayer++;
  addPrefillNorm(step, input, norm, mixer.inputProjection.layout());
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked, b.projectionSums,
                                 step.rows, b.linearScratch);
  for (const QwenTargetPrefillSequence &sequence : step.sequences) {
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<float>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    ops::GDN::addPrefill(
        step.graph,
        {u16(b.gdnPacked, geometry_.packedGdnWidth), mixer.convolutionWeights, sequence.convolutionIn[layer],
         sequence.convolutionOut[layer], u16(b.gdnQueries, geometry_.gdnKeyWidth()),
         u16(b.gdnKeys, geometry_.gdnKeyWidth()), u16(b.gdnValues, geometry_.attentionWidth), mixer.decay,
         mixer.timeBias, f32(b.gdnDecay, geometry_.gdnValueHeads), u16(b.gdnBeta, geometry_.gdnValueHeads),
         sequence.recurrentIn[layer], sequence.recurrentOut[layer], u16(b.recurrent, geometry_.attentionWidth),
         mixer.mixerNorm, u16(b.gdnHidden, geometry_.attentionWidth)},
        geometry_.gdnShape(), sequence.rows, mixer.outputHeadOrder);
  }
  addPrefillOutput(step, b.gdnHidden, mixer.outputProjection, input, b.gdnOutput);
  return b.gdnOutput;
}

metal::MetalBuffer QwenTarget::addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.attentionLayer++;
  addPrefillNorm(step, input, norm, mixer.inputProjection.layout());
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, b.projectionSums,
                                 step.rows, b.linearScratch);
  for (size_t index = 0; index < step.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<float>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const uint64_t headBytes = uint64_t{sequence.attentionStride} * geometry_.attentionHeadDimension * sizeof(uint16_t);
    const uint64_t queryBytes = geometry_.attentionQueryHeads * headBytes;
    const uint64_t kvBytes = geometry_.attentionKvHeads * headBytes;
    const metal::MetalBuffer queries = backend_.view(b.fullQueries, sequence.queryOffset, queryBytes);
    const metal::MetalBuffer attentionRows = backend_.view(b.fullAttention, sequence.queryOffset, queryBytes);
    const metal::MetalBuffer keys = backend_.view(b.chunkKeys, sequence.kvOffset, kvBytes);
    const metal::MetalBuffer values = backend_.view(b.chunkValues, sequence.kvOffset, kvBytes);
    ops::PagedAttention::addPrefillProjection(
        step.graph, u16(b.fullPacked, geometry_.packedFullWidth), mixer.queryNorm, mixer.keyNorm,
        f32(b.ropeCos, geometry_.rotaryPairs), f32(b.ropeSin, geometry_.rotaryPairs), queries, keys, values,
        sequence.rows, sequence.attentionStride, geometry_.attentionQueryHeads, geometry_.kvLayout);
    ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[layer], keys, values, sequence.pageTable,
                                         sequence.chunk, geometry_.kvLayout);
    ops::PagedAttention::addPrefill(
        step.graph, step.kvLayers[layer], queries, attentionRows, b.attentionPartials, b.attentionStatistics,
        sequence.pageTable, sequence.chunk, step.attention[index]);
    ops::PagedAttention::addPrefillGate(
        step.graph, u16(b.fullPacked, geometry_.packedFullWidth), attentionRows,
        u16(b.attentionHidden, geometry_.attentionWidth), sequence.rows, sequence.attentionStride,
        geometry_.attentionQueryHeads, geometry_.kvLayout);
  }
  addPrefillOutput(step, b.attentionHidden, mixer.outputProjection, input, b.attentionOutput);
  return b.attentionOutput;
}

void QwenTarget::addPrefillFfn(PrefillStep &step, uint32_t index, const Qwen3_8LayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  addPrefillNorm(step, residual, layer.postAttentionNorm, layer.gateProjection.layout());
  const ops::PrefillFfnBuffers ffn = step.buffers.ffn();
  if (step.aneFfn)
    step.aneFfn->add(step.graph, index, ffn, residual, output, step.rows);
  else
    operators_.linear().addPrefillSwiGlu(step.graph, {&layer.gateProjection, &layer.upProjection,
                                                      &layer.downProjection},
                                         ffn, residual, output, step.rows);
}

void QwenTarget::addPrefillFfn(PrefillStep &step, uint32_t, const Qwen3_6MoeLayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                             step.rows);
  ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, layer.ffn, *step.moe);
}

void QwenTarget::addVerify(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const SplashKvLayer> kvLayers,
    std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || chunks.size() != lanes ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry_.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry_.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry_.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry_.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers ||
      (buffers.gdnDefer && (lanes != 1 || buffers.gdnPendingMixed.size() != geometry_.stateLayout.layers ||
                            buffers.gdnPendingDecay.size() != geometry_.stateLayout.layers ||
                            buffers.gdnPendingBeta.size() != geometry_.stateLayout.layers))) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = chunks[lane].committed_tokens;
  VerifyStep step{graph, buffers, kvLayers, chunks, lanes, rows,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.kvLayout,
                                             std::span(histories).first(lanes))};
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moeDecode(geometry_.moeShape(), lanes);
  step.m16HoistFooter = lanes == 2 && metal::envSwitch("SPLASH_SPLIT4_M16");
  addVerifyLayers(step);
}

void QwenTarget::addVerify16(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const SplashKvLayer> kvLayers,
    std::span<const kv::ChunkedPrefillParams> chunks, uint32_t tiles,
    metal::MetalBuffer convolutionScratch, ops::WideGdn gdnRoute) const {
  const ops::GdnStateStrides state{geometry_.stateLayout.convolutionLayerBytes(),
                                   geometry_.stateLayout.recurrentLayerBytes(),
                                   geometry_.stateLayout.convolutionBytes()};
  if ((tiles != 2 && tiles != 4) || geometry_.ffnKind != QwenFfnKind::Dense || chunks.size() != tiles ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry_.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry_.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry_.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry_.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers || buffers.gdnDefer ||
      convolutionScratch.sizeBytes() < std::max(ops::gdnDecode16ConvolutionScratchBytes(state, tiles),
                                                ops::gdnCommit16ConvolutionScratchBytes(state, tiles)))
    throw std::invalid_argument("invalid wide Qwen verify batch");
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < tiles; ++lane)
    histories[lane] = chunks[lane].committed_tokens;
  VerifyStep step{graph, buffers, kvLayers, chunks, tiles, tiles * ExecutionLimits::targetVerifyRows,
                  operators_.verifyAttention(tiles, geometry_.attentionQueryHeads, geometry_.kvLayout,
                                             std::span(histories).first(tiles))};
  step.wide = true;
  step.convolutionScratch = std::move(convolutionScratch);
  step.gdnRoute = gdnRoute;
  addVerifyLayers(step);
}

void QwenTarget::addVerifyLayers(VerifyStep &step) const {
  metal::CommandGraph &graph = step.graph;
  const QwenTargetVerifyBuffers &buffers = step.buffers;
  const uint32_t rows = step.rows, lanes = step.lanes;
  std::visit([&](const auto *weights) {
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      const auto &layer = weights->layers[index];
      const metal::MetalBuffer input = buffers.hidden[index & 1];
      const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
      const metal::MetalBuffer residual = std::visit(
          [&](const auto &mixer) { return addVerifyMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
      addVerifyFfn(step, layer, residual, output);
      if (const auto slot = geometry_.captureSlot(index))
        addCapture(graph, geometry_, *slot, output, 0, buffers.capturedTargetHidden, 0, rows);
    }
    requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  }, weights_);
  addHeadBatch(graph, buffers.hidden[geometry_.layers & 1], buffers.finalHidden, buffers.logits, lanes,
               buffers.linearScratch);
}

void QwenTarget::addMixerInput(VerifyStep &step, const ops::NormWeights &norm, metal::MetalBuffer input,
                               const ops::Projection &projection, metal::MetalBuffer packed,
                               metal::MetalBuffer packedPadded) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  // SPLASH_M24_PAD3 (default on; read at every encode): a three-lane
  // verify whose input projection takes the RMS's group sums (normInput)
  // runs the RMS and the projection over the idle fourth lane too, as the
  // split-K M32 consumer, cheaper than M24 (fastkernel 1.0.0: B3 -4.58 ms per
  // cycle). Exact per row: the split-K tile's rows don't depend on the rows
  // beside them; the fourth lane's rows are discarded.
  const bool pad3 = step.lanes == 3 && packedPadded && b.normalizedPadded &&
      linear.normInput(linear.decodePlan(projection, 3)) == ops::LinearInput::GroupSums &&
      metal::envSwitch("SPLASH_M24_PAD3");
  const uint32_t lanes = pad3 ? 4 : step.lanes;
  const ops::LinearPlan plan = linear.decodePlan(projection, lanes);
  const metal::MetalBuffer normalized = pad3 ? b.normalizedPadded : b.normalized;
  const metal::MetalBuffer rows = pad3 ? b.hiddenPadded[input.sameView(b.hidden[0]) ? 0 : 1] : input;
  const ops::PreparedInput prepared =
      ops::Normalization::addRms(step.graph, rows, norm, normalized, geometry_.hiddenSize,
                                 lanes * ExecutionLimits::targetVerifyRows, b.linearScratch, linear.normInput(plan));
  linear.add(step.graph,
             {.input = normalized, .output = pad3 ? packedPadded : packed, .scratch = b.linearScratch,
              .prepared = prepared},
             projection, plan);
}

// Each producer emits the table (if any) its consumer's plan reads.
metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const QwenGdnWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.gdnLayer++;
  addMixerInput(step, norm, input, mixer.inputProjection, b.gdnPacked[layer],
                b.gdnPackedPadded.size() == b.gdnPacked.size() ? b.gdnPackedPadded[layer] : metal::MetalBuffer{});
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.lanes, ops::LinearEpilogue::Residual);
  const ops::GdnStateStrides state{geometry_.stateLayout.convolutionLayerBytes(),
                                   geometry_.stateLayout.recurrentLayerBytes(),
                                   geometry_.stateLayout.convolutionBytes()};
  ops::GdnDecodeBuffers gdn{b.gdnPacked[layer], mixer.convolutionWeights, b.currentGdnStates,
                            b.nextGdnStates, b.gdnMixed[layer], mixer.decay, mixer.timeBias,
                            b.gdnDecay[layer], b.gdnBeta[layer], mixer.mixerNorm, b.gdnHidden,
                            step.wide ? ops::LinearScratch{} : b.linearScratch};
  if (b.gdnDefer)
    gdn.defer = {true, b.gdnPendingRows, b.gdnDeferBase, b.gdnPendingMixed[layer], b.gdnPendingDecay[layer],
                 b.gdnPendingBeta[layer]};
  // A wide lookup's GDN writes plain rows; the out-projection prepares its own input.
  ops::PreparedInput hidden{};
  if (step.wide)
    ops::GDN::addDecode16(step.graph, gdn, step.convolutionScratch, geometry_.gdnShape(), layer, state,
                          mixer.outputHeadOrder, step.lanes, step.gdnRoute);
  else
    hidden = ops::GDN::addDecode(
        step.graph, gdn, geometry_.gdnShape(), step.lanes, layer, state, mixer.outputHeadOrder,
        ops::GDN::outputInput(outputPlan, geometry_.gdnShape(), mixer.outputHeadOrder, mixer.mixerNorm));
  linear.add(step.graph,
             {.input = b.gdnHidden, .output = b.gdnOutput, .residual = input, .scratch = b.linearScratch,
              .prepared = hidden, .m16HoistFooter = step.m16HoistFooter},
             mixer.outputProjection, outputPlan);
  return b.gdnOutput;
}

metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer++;
  addMixerInput(step, norm, input, mixer.inputProjection, b.fullPacked, b.fullPackedPadded);
  ops::PagedAttention::addVerifyProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                           b.ropeSin, b.fullQueries, b.chunkKeys[layer], b.chunkValues[layer],
                                           geometry_.attentionQueryHeads, geometry_.kvLayout, step.lanes);
  ops::PagedAttention::addVerify(step.graph, step.kvLayers[layer],
                                 {b.chunkKeys[layer], b.chunkValues[layer], b.fullQueries, b.attentionPartials,
                                  b.attentionStatistics, b.fullAttention, b.pageTables},
                                 step.chunks, step.attention);
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.lanes, ops::LinearEpilogue::Residual);
  const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
      step.graph, b.fullPacked, b.fullAttention, b.attentionHidden, geometry_.attentionQueryHeads,
      geometry_.kvLayout, step.lanes, b.linearScratch, outputPlan.input());
  linear.add(step.graph,
             {.input = b.attentionHidden, .output = b.attentionOutput, .residual = input,
              .scratch = b.linearScratch, .prepared = hidden, .m16HoistFooter = step.m16HoistFooter},
             mixer.outputProjection, outputPlan);
  return b.attentionOutput;
}

void QwenTarget::addVerifyFfn(VerifyStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan gateUpPlan =
      linear.decodePlan(layer.upProjection, step.lanes, ops::LinearEpilogue::GateUp, &layer.gateProjection);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize, step.rows,
      b.linearScratch, gateUpPlan.input());
  const ops::LinearPlan downPlan =
      linear.decodePlan(layer.downProjection, step.lanes, ops::LinearEpilogue::Residual);
  const ops::PreparedInput intermediate = linear.add(
      step.graph,
      {.input = b.normalized, .output = b.denseIntermediate, .gateScratch = b.denseGateScratch,
       .downSums = !step.wide && linear.gateUpWritesDownSums(gateUpPlan, downPlan) ? b.linearScratch.sums
                                                                                    : metal::MetalBuffer{},
       .scratch = b.linearScratch, .prepared = normalized},
      layer.upProjection, gateUpPlan, &layer.gateProjection);
  linear.add(step.graph,
             {.input = b.denseIntermediate, .output = output, .residual = residual, .scratch = b.linearScratch,
              .prepared = intermediate, .m16HoistFooter = step.m16HoistFooter},
             layer.downProjection, downPlan);
}

void QwenTarget::addVerifyFfn(VerifyStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                             step.rows);
  ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, layer.ffn, *step.moe);
}

void QwenTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                              metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
                              ops::LinearScratch scratch) const {
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan logitsPlan = linear.decodePlan(vocabularyProjection(), lanes);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      graph, std::move(hidden), weightsBase_.finalNorm, finalHidden, geometry_.hiddenSize,
      lanes * ExecutionLimits::targetVerifyRows, scratch, logitsPlan.input());
  linear.add(graph,
             {.input = std::move(finalHidden), .output = std::move(logits), .scratch = scratch,
              .prepared = normalized},
             vocabularyProjection(), logitsPlan);
}

void QwenTarget::addVerifyInput(metal::CommandGraph &graph,
                                metal::MetalBuffer draftInput,
                                metal::MetalBuffer proposals,
                                metal::MetalBuffer verifyInput,
                                uint32_t lanes) const {
  ops::Embedding::addVerifyInput(graph, std::move(draftInput),
                                 std::move(proposals), std::move(verifyInput),
                                 geometry_.vocabularySize, lanes);
}

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, std::move(hidden),
                      rows);
}

void QwenTarget::addStateCommit16(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                                  metal::MetalBuffer convolutionScratch, uint32_t tiles) const {
  if (geometry_.ffnKind != QwenFfnKind::Dense)
    throw std::invalid_argument("wide Qwen commit requires a dense target");
  ops::GDN::addCommit16(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed), std::move(buffers.decay), std::move(buffers.beta),
       buffers.currentStates, buffers.nextStates, std::move(buffers.retainedCounts)},
      std::move(convolutionScratch), geometry_.gdnShape(), geometry_.stateLayout.layers,
      {geometry_.stateLayout.convolutionLayerBytes(), geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()},
      tiles);
}

void QwenTarget::addStateCommit(metal::CommandGraph &graph,
                                QwenTargetCommitBuffers buffers,
                                uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  ops::GDN::addCommit(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed),
       std::move(buffers.decay), std::move(buffers.beta), buffers.currentStates,
       buffers.nextStates, std::move(buffers.retainedCounts)},
      geometry_.gdnShape(), geometry_.stateLayout.layers, lanes,
      {geometry_.stateLayout.convolutionLayerBytes(),
       geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()});
}

bool QwenTarget::gdnDeferSupported() const {
  const auto *dense = std::get_if<0>(&weights_);
  if (geometry_.ffnKind != QwenFfnKind::Dense || !dense)
    return false;
  for (const auto &layer : (*dense)->layers)
    if (const auto *mixer = std::get_if<QwenGdnWeights>(&layer.mixer)) {
      const ops::LinearPlan plan =
          operators_.linear().decodePlan(mixer->outputProjection, 1, ops::LinearEpilogue::Residual);
      return ops::GDN::deferRoute(
          geometry_.gdnShape(),
          ops::GDN::outputInput(plan, geometry_.gdnShape(), mixer->outputHeadOrder, mixer->mixerNorm));
    }
  return false;
}

void QwenTarget::addStateCommitConv(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers) const {
  ops::GDN::addCommitConv(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed), std::move(buffers.decay), std::move(buffers.beta),
       buffers.currentStates, buffers.nextStates, std::move(buffers.retainedCounts)},
      geometry_.gdnShape(), geometry_.stateLayout.layers,
      {geometry_.stateLayout.convolutionLayerBytes(), geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()});
}

void QwenTarget::addStateFlush(metal::CommandGraph &graph, ops::GdnFlushBuffers buffers, uint32_t rows,
                               uint32_t slot) const {
  ops::GDN::addFlush(graph, std::move(buffers), geometry_.gdnShape(), geometry_.stateLayout.layers, rows, slot,
                     {geometry_.stateLayout.convolutionLayerBytes(), geometry_.stateLayout.recurrentLayerBytes(),
                      geometry_.stateLayout.convolutionBytes()});
}

uint32_t QwenTarget::rowStableVerifyRows() const {
  constexpr uint32_t rows8 = ExecutionLimits::targetVerifyRows;
  const auto *dense = std::get_if<0>(&weights_);
  if (geometry_.ffnKind != QwenFfnKind::Dense || !dense)
    return rows8;
  const ops::Linear &linear = operators_.linear();
  struct Use {
    const ops::Projection *projection;
    ops::LinearEpilogue epilogue;
    const ops::Projection *gate;
    bool input;  // a mixer input projection, whose norm may write its sums
  };
  std::vector<Use> uses;
  bool gdn = false, attention = false;
  for (const auto &layer : (*dense)->layers) {
    std::visit([&](const auto &mixer) {
      using Mixer = std::remove_cvref_t<decltype(mixer)>;
      bool &seen = std::is_same_v<Mixer, QwenGdnWeights> ? gdn : attention;
      if (std::exchange(seen, true)) return;
      uses.push_back({&mixer.inputProjection, ops::LinearEpilogue::None, nullptr, true});
      uses.push_back({&mixer.outputProjection, ops::LinearEpilogue::Residual, nullptr, false});
    }, layer.mixer);
  }
  if (const auto &layers = (*dense)->layers; !layers.empty()) {
    uses.push_back({&layers[0].upProjection, ops::LinearEpilogue::GateUp, &layers[0].gateProjection, false});
    uses.push_back({&layers[0].downProjection, ops::LinearEpilogue::Residual, nullptr, false});
  }
  uses.push_back({&vocabularyProjection(), ops::LinearEpilogue::None, nullptr, false});
  const auto plan = [&](const Use &use, uint32_t lanes) {
    return linear.decodePlan(*use.projection, lanes, use.epilogue, use.gate);
  };
  uint32_t rows = rows8;
  for (const uint32_t lanes : {2U, 4U}) {  // the wide lookup widths: 16 and 32 rows
    for (const Use &use : uses)
      if (!linear.sameRowReduction(plan(use, 1), plan(use, lanes), use.input)) return rows;
    rows = lanes * rows8;
  }
  return rows;
}

} // namespace splash::model
