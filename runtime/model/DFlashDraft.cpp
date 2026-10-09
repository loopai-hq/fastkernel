// Modified by Pulsar.
#include "DFlashDraft.hpp"
#include "Checked.hpp"
#include "DraftCheckpoint.hpp"
#include "metal/abi/DraftAttention.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

void requireLayout(const DFlashDraftLayout &layout) {
  if (!layout.layers || !layout.hiddenSize || !layout.vocabularySize ||
      !layout.dynamicSize || !layout.qkvSize || !layout.attentionSize ||
      !layout.intermediateSize || !layout.attentionHeadDimension ||
      !(layout.rotaryTheta > 0.0F) ||
      !layout.targetHiddenSize || !layout.selectorRank ||
      !layout.kvHeads) {
    throw WeightStoreError("DFlash draft layout contains a zero dimension");
  }
  if (layout.selectorRank != SPLASH_DRAFT_SELECTOR_RANK) {
    throw WeightStoreError("draft selector kernels are compiled for rank " +
                           std::to_string(SPLASH_DRAFT_SELECTOR_RANK));
  }
  validateQ4Layout(layout.dynamicSize, layout.hiddenSize);
  validateQ4Layout(layout.qkvSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionSize);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.hiddenSize, layout.targetHiddenSize);
  validateQ4Layout(layout.selectorRank, layout.hiddenSize);
}

// The key and value rows of each layer's fused QKV projection, without a
// copy: affine planes store whole tiles of kQ4StorageN rows in row order
// (AffinePreparation), so rows from a tile boundary on are one range of
// each plane.
std::vector<ops::Projection> contextKvRows(metal::MetalBackend &backend,
                                           const DFlashDraftWeights &weights) {
  const DFlashDraftLayout &layout = weights.layout;
  if (layout.attentionSize >= layout.qkvSize ||
      layout.attentionSize % kQ4StorageN) {
    throw std::invalid_argument(
        "draft key and value rows do not start at a storage tile");
  }
  std::vector<ops::Projection> result;
  result.reserve(weights.layers.size());
  for (const DFlashDraftLayerWeights &layer : weights.layers) {
    const ops::AffineWeights &fused = layer.qkvProjection.affine();
    const auto rows = [&](const metal::MetalBuffer &plane) {
      const uint64_t rowBytes = plane.sizeBytes() / layout.qkvSize;
      return backend.view(plane, uint64_t{layout.attentionSize} * rowBytes,
                          uint64_t{layout.contextKvSize()} * rowBytes);
    };
    result.emplace_back(layout.contextKvSize(), layout.hiddenSize,
                        ops::AffineWeights{rows(fused.weights),
                                           rows(fused.scales),
                                           rows(fused.biases)});
  }
  return result;
}

// SPLASH_DRAFT_HEAD_IDS / SPLASH_DRAFT_HEAD_ROWS: restricted head rows, 0 when unset.
uint32_t configuredHeadRows() {
  const char *path = std::getenv("SPLASH_DRAFT_HEAD_IDS");
  if (!path || !*path) return 0;
  const char *rowsValue = std::getenv("SPLASH_DRAFT_HEAD_ROWS");
  return rowsValue ? static_cast<uint32_t>(std::strtoul(rowsValue, nullptr, 10)) : 98304u;
}

} // namespace

uint64_t DFlashDraft::restrictedHeadPlannedBytes(const DFlashDraftLayout &layout) {
  // draftHead(): rows x (Q4 weights, bf16 scales, bf16 biases, u32 ids) over the hidden width.
  const uint64_t rows = configuredHeadRows();
  const auto pages = [](uint64_t bytes) { return (bytes + 16383) & ~uint64_t{16383}; };
  const uint64_t metadata = pages(rows * (layout.hiddenSize / 64) * 2);
  return rows ? pages(rows * layout.hiddenSize / 2) + 2 * metadata + pages(rows * 4) : 0;
}

bool DFlashDraft::gathersRestrictedHead(const DFlashDraftLayout &layout,
                                        const ops::Projection &target) noexcept {
  return target.outputSize == layout.vocabularySize &&
         target.layout() == ops::WeightLayout::Affine64 && target.inputSize % 64 == 0;
}

void DFlashDraft::loadRestrictedHead(const ops::Projection &target) const {
  if (!headIds_.empty())
    static_cast<void>(draftHead(target));
}

void DFlashDraft::disableRestrictedHead() noexcept {
  headIds_.clear();
  headKeep_.clear();
  headFiller_.clear();
}

uint64_t DFlashDraft::restrictedHeadAllocatedBytes() const noexcept {
  if (!restrictedHead_) return 0;
  const ops::AffineWeights &planes = restrictedHead_->affine();
  return planes.weights.allocatedBytes() + planes.scales.allocatedBytes() +
         planes.biases.allocatedBytes() + headIdMap_.allocatedBytes();
}

DFlashDraftRing::DFlashDraftRing(
    metal::MetalBackend &backend, std::shared_ptr<StateAllocationTracker> tracker,
    DraftStateLayout layout, std::string_view label)
    : tracker_(std::move(tracker)), layers_(layout.layers) {
  if (!tracker_)
    throw std::invalid_argument("draft state allocation tracker is empty");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  const metal::MetalBuffer base = backend.allocateBuffer(
      layout.ringBytes(), metal::BufferStorage::Shared, label);
  uint64_t cursor = 0;
  for (DFlashDraftRingLayer &layer : layers_) {
    layer.keys = backend.view(base, cursor, layout.tensorBytes());
    cursor += layout.tensorBytes();
    layer.values = backend.view(base, cursor, layout.tensorBytes());
    cursor += layout.tensorBytes();
  }
  if (cursor != layout.ringBytes())
    throw std::logic_error("draft ring accounting mismatch");
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (actualAllocatedBytes_ < layout.ringBytes())
    throw std::logic_error("draft ring allocation is below declared bytes");
  tracker_->bytes.fetch_add(actualAllocatedBytes_, std::memory_order_relaxed);
}

DFlashDraftRing::~DFlashDraftRing() {
  tracker_->bytes.fetch_sub(actualAllocatedBytes_, std::memory_order_relaxed);
}

DFlashDraft::DFlashDraft(const DFlashDraftWeights &weights,
                         metal::MetalBackend &backend,
                         const ops::ExecutionPlans &operators)
    : weights_(weights), backend_(backend), operators_(operators),
      selector_(weights.layout.vocabularySize),
      contextKvProjections_(contextKvRows(backend, weights)) {
  // Explicit path, off when unset: the ranked ids are data for one tokenizer.
  if (const char *path = std::getenv("SPLASH_DRAFT_HEAD_IDS"); path && *path) {
    // Ranked ids (most frequent first, unique). The first rows - 256 form the
    // static head (sorted ascending); the rest fill each request's segment.
    std::ifstream file(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(file)), {});
    const uint32_t vocabulary = weights_.layout.vocabularySize;
    const uint32_t rows = configuredHeadRows();
    const size_t count = bytes.size() / 4;
    if (bytes.size() % 4 || rows % 256 || rows <= kHeadSegmentRows || rows >= vocabulary ||
        count < rows + kHeadSegmentRows)
      throw std::invalid_argument("draft head ids must rank at least SPLASH_DRAFT_HEAD_ROWS + 256 ids; rows % 256 == 0");
    std::vector<uint32_t> ranked(count);
    std::memcpy(ranked.data(), bytes.data(), count * 4);
    headKeep_.assign(vocabulary, 0);
    std::vector<uint8_t> seen(vocabulary, 0);
    for (const uint32_t id : ranked) {
      if (id >= vocabulary || seen[id])
        throw std::invalid_argument("draft head ids must be unique vocabulary ids");
      seen[id] = 1;
    }
    const size_t staticRows = rows - kHeadSegmentRows;
    headIds_.assign(ranked.begin(), ranked.begin() + staticRows);
    std::sort(headIds_.begin(), headIds_.end());
    for (const uint32_t id : headIds_) headKeep_[id] = 1;
    headFiller_.assign(ranked.begin() + staticRows, ranked.end());
  }
}

std::vector<uint32_t>
DFlashDraft::promptSegment(std::span<const uint32_t> prompt, uint32_t *promptIds) const {
  if (headIds_.empty() || prompt.empty())
    return {};
  std::vector<uint32_t> segment;
  std::vector<uint8_t> taken(headKeep_.size(), 0);
  for (const uint32_t token : prompt) {
    if (token >= headKeep_.size() || headKeep_[token] || taken[token])
      continue;
    if (segment.size() == kHeadSegmentRows)
      return {};  // too many rare prompt tokens: keep the full head
    taken[token] = 1;
    segment.push_back(token);
  }
  // Prompts where over 1% of tokens are rare (other languages) keep the full
  // head: their answers use rare tokens the prompt does not show yet.
  size_t outside = 0;
  for (const uint32_t token : prompt)
    outside += token < headKeep_.size() && !headKeep_[token];
  if (outside * 100 > prompt.size())
    return {};
  if (promptIds) *promptIds = static_cast<uint32_t>(segment.size());
  for (size_t i = 0; segment.size() < kHeadSegmentRows && i < headFiller_.size(); ++i)
    if (!taken[headFiller_[i]]) segment.push_back(headFiller_[i]);
  return segment;
}

void DFlashDraft::copyHeadRows(const ops::Projection &target,
                               std::span<const uint32_t> ids,
                               uint32_t firstRow) const {
  // Affine planes: weights [N/256][K/64][256][32 B]; scales and biases
  // [N/256][K/64][256] bf16 (AffinePreparation). Rows move independently,
  // bytes unchanged.
  const ops::AffineWeights &source = target.affine();
  const ops::AffineWeights &destination = restrictedHead_->affine();
  const auto *srcWeights = static_cast<const uint8_t *>(source.weights.contents());
  const auto *srcScales = static_cast<const uint16_t *>(source.scales.contents());
  const auto *srcBiases = static_cast<const uint16_t *>(source.biases.contents());
  auto *weights = static_cast<uint8_t *>(destination.weights.contents());
  auto *scales = static_cast<uint16_t *>(destination.scales.contents());
  auto *biases = static_cast<uint16_t *>(destination.biases.contents());
  const uint64_t groups = target.inputSize / 64;
  for (uint32_t i = 0; i < ids.size(); ++i) {
    const uint64_t from0 = ids[i], row = firstRow + i;
    for (uint64_t group = 0; group < groups; ++group) {
      const uint64_t from = ((from0 / kQ4StorageN) * groups + group) * kQ4StorageN + from0 % kQ4StorageN;
      const uint64_t to = ((row / kQ4StorageN) * groups + group) * kQ4StorageN + row % kQ4StorageN;
      std::memcpy(weights + to * 32, srcWeights + from * 32, 32);
      scales[to] = srcScales[from];
      biases[to] = srcBiases[from];
    }
  }
  std::memcpy(static_cast<uint32_t *>(headIdMap_.contents()) + firstRow, ids.data(), ids.size() * 4);
}

void DFlashDraft::useHeadSegment(uint64_t owner, uint64_t version,
                                 std::span<const uint32_t> segment,
                                 const ops::Projection &target) const {
  if (segment.size() != kHeadSegmentRows)
    throw std::invalid_argument("draft head segment must have 256 ids");
  static_cast<void>(draftHead(target));
  if (owner == segmentOwner_ && version == segmentVersion_)
    return;
  // No command reads the head now: the previous cycle's ticket was consumed,
  // and a draft-ahead block reading other rows retires first. Only rows whose
  // id changed are copied (one row per new rare token).
  const uint32_t first = static_cast<uint32_t>(headIds_.size());
  for (uint32_t i = 0; i < kHeadSegmentRows; ++i)
    if (loadedSegment_[i] != segment[i]) {
      copyHeadRows(target, segment.subspan(i, 1), first + i);
      loadedSegment_[i] = segment[i];
    }
  segmentOwner_ = owner;
  segmentVersion_ = version;
}

const ops::Projection &DFlashDraft::draftHead(const ops::Projection &target) const {
  if (restrictedHead_)
    return *restrictedHead_;
  if (headIds_.empty() || !gathersRestrictedHead(weights_.layout, target))
    throw std::logic_error("restricted draft head needs SPLASH_DRAFT_HEAD_IDS and the full affine target head");
  const ops::AffineWeights &source = target.affine();
  if (!source.weights.contents() || !source.scales.contents() || !source.biases.contents())
    throw std::runtime_error("target head is not CPU-visible Q4 for draft head gathering");
  const uint32_t count = static_cast<uint32_t>(headIds_.size()) + kHeadSegmentRows;
  const uint64_t groups = target.inputSize / 64;
  ops::Projection head(count, target.inputSize,
                       ops::AffineWeights{
                           backend_.allocateBuffer(uint64_t{count} * target.inputSize / 2,
                                                   metal::BufferStorage::Shared, "draft-head-weights"),
                           backend_.allocateBuffer(uint64_t{count} * groups * 2, metal::BufferStorage::Shared,
                                                   "draft-head-scales"),
                           backend_.allocateBuffer(uint64_t{count} * groups * 2, metal::BufferStorage::Shared,
                                                   "draft-head-biases")});
  head.destination = target.destination;
  restrictedHead_ = std::move(head);
  headIdMap_ = backend_.allocateBuffer(uint64_t{count} * 4, metal::BufferStorage::Shared, "draft-head-ids");
  copyHeadRows(target, headIds_, 0);
  copyHeadRows(target, std::span(headFiller_).first(kHeadSegmentRows), static_cast<uint32_t>(headIds_.size()));
  loadedSegment_.assign(headFiller_.begin(), headFiller_.begin() + kHeadSegmentRows);
  segmentOwner_ = 0;
  return *restrictedHead_;
}

void DFlashDraft::addSelection(
    metal::CommandGraph &graph, ops::DraftSelectorBuffers buffers,
    std::span<const uint32_t> anchors,
    std::span<const ops::SamplingPolicy> policies, bool restrictedHead) const {
  if (restrictedHead && !restrictedHead_)
    throw std::logic_error("restricted draft head selected before it was gathered");
  selector_.add(graph, buffers,
                {weights_.predecessorCodebook, weights_.successorCodebook},
                anchors, policies,
                restrictedHead ? ops::DraftHeadMap{headIdMap_, restrictedHead_->outputSize}
                               : ops::DraftHeadMap{});
}

void DFlashDraft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
    std::span<const DFlashPrefillSpan> spans) const {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget || spans.empty())
    throw std::invalid_argument("invalid draft context prefill");
  const DFlashDraftLayout &layout = weights_.layout;
  for (const DFlashPrefillSpan &span : spans) {
    if (span.ring.size() != layout.layers)
      throw std::invalid_argument("draft prefill ring layer mismatch");
  }
  operators_.linear().addPrefillSums(graph, buffers.capturedTargetHidden, buffers.projectionSums,
                                     weights_.contextProjection, rows);
  operators_.linear().addPrefill(graph, buffers.capturedTargetHidden, weights_.contextProjection,
                                 buffers.projected, buffers.projectionSums, rows);
  ops::Normalization::addRmsWithQ4Sums(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden,
      buffers.projectionSums, layout.hiddenSize, rows);

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    operators_.linear().addPrefill(graph, buffers.hidden,
                      contextKvProjections_[layer],
                      buffers.contextKv, buffers.projectionSums, rows);
    for (const DFlashPrefillSpan &span : spans) {
      const uint64_t kvOffset =
          uint64_t{span.compactRow} * layout.contextKvSize() * sizeof(uint16_t);
      const uint64_t ropeOffset =
          uint64_t{span.compactRow} * (layout.attentionHeadDimension / 2) * sizeof(float);
      ops::DraftAttention::addContextPrefill(
          graph,
          backend_.view(buffers.contextKv, kvOffset,
                        uint64_t{span.rows} * layout.contextKvSize() *
                            sizeof(uint16_t)),
          weights_.layers[layer].keyNorm,
          backend_.view(buffers.ropeCos, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          backend_.view(buffers.ropeSin, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          span.ring[layer].keys, span.ring[layer].values, span.rows,
          span.startPosition, layout.attentionShape());
    }
  }
}

void DFlashDraft::addDecode(
    metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
    const ops::Projection &vocabularyProjection,
    std::span<const uint32_t> cacheLengths, bool restrictedHead,
    metal::MetalBuffer deviceAttentionParams) const {
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft decode batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::draftQueryRows;
  const auto attentionPlan =
      operators_.draftAttention(layout.attentionShape(), lanes);
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const DFlashDraftLayerWeights &weights = weights_.layers[layer];
    const ops::LinearPlan attentionDynamicPlan = linear.decodePlan(weights.attentionDynamic, lanes);
    const ops::PreparedInput attentionNormalized = ops::Normalization::addRms(
        graph, buffers.hidden[current], weights.inputNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, attentionDynamicPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.dynamic, .scratch = scratch,
                .prepared = attentionNormalized},
               weights.attentionDynamic, attentionDynamicPlan);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    linear.add(graph, {.input = buffers.convolved, .output = buffers.proposalQkv, .scratch = scratch},
               weights.qkvProjection, linear.decodePlan(weights.qkvProjection, lanes));
    ops::DraftAttention::addPrepare(
        graph,
        {buffers.proposalQkv, buffers.attention, weights.queryNorm,
         weights.keyNorm, buffers.ropeCos, buffers.ropeSin, buffers.queryKeys,
         buffers.queryValues},
        attentionPlan);
    ops::DraftAttention::addDecode(
        graph,
        {buffers.attention, buffers.persistentKeys[layer],
         buffers.persistentValues[layer], buffers.queryKeys,
         buffers.queryValues},
        cacheLengths, attentionPlan, deviceAttentionParams);
    ops::DraftAttention::addReorder(graph, buffers.attention,
                                    buffers.proposalQkv, attentionPlan);
    linear.add(graph, {.input = buffers.proposalQkv, .output = buffers.projected, .scratch = scratch},
               weights.outputProjection, linear.decodePlan(weights.outputProjection, lanes));
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.residual},
        attentionPlan, ops::DraftConvolutionStage::Residual);
    const ops::LinearPlan mlpDynamicPlan = linear.decodePlan(weights.mlpDynamic, lanes);
    const ops::PreparedInput mlpNormalized = ops::Normalization::addRms(
        graph, buffers.residual, weights.postAttentionNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, mlpDynamicPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.dynamic, .scratch = scratch,
                .prepared = mlpNormalized},
               weights.mlpDynamic, mlpDynamicPlan);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    linear.add(graph,
               {.input = buffers.convolved, .output = buffers.intermediate, .gateScratch = buffers.gateScratch,
                .scratch = scratch},
               weights.upProjection,
               linear.decodePlan(weights.upProjection, lanes, ops::LinearEpilogue::GateUp, &weights.gateProjection),
               &weights.gateProjection);
    linear.add(graph, {.input = buffers.intermediate, .output = buffers.projected, .scratch = scratch},
               weights.downProjection, linear.decodePlan(weights.downProjection, lanes));
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.hidden[next]},
        attentionPlan, ops::DraftConvolutionStage::Residual);
  }

  // The final norm feeds the shared vocabulary head and then the selector;
  // the selector reuses whatever table the head leaves when the layouts match.
  if (restrictedHead && lanes != 1)
    throw std::invalid_argument("the restricted draft head serves one lane");
  const ops::Projection &head = restrictedHead ? draftHead(vocabularyProjection) : vocabularyProjection;
  const ops::LinearPlan headPlan = linear.decodePlan(head, lanes);
  const ops::PreparedInput finalHidden = ops::Normalization::addRms(
      graph, buffers.hidden[weights_.layout.layers & 1], weights_.finalNorm,
      buffers.finalHidden, layout.hiddenSize, rows, scratch, headPlan.input());
  const ops::PreparedInput afterHead = linear.add(
      graph, {.input = buffers.finalHidden, .output = buffers.logits, .scratch = scratch, .prepared = finalHidden},
      head, headPlan);
  linear.add(graph,
             {.input = buffers.finalHidden, .output = buffers.selectorHidden, .scratch = scratch,
              .prepared = afterHead},
             weights_.selectorProjection, linear.decodePlan(weights_.selectorProjection, lanes));
}

void DFlashDraft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions) const {
  const uint32_t lanes = static_cast<uint32_t>(startPositions.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft context batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  linear.add(graph, {.input = buffers.capturedTargetHidden, .output = buffers.projected, .scratch = scratch},
             weights_.contextProjection, linear.decodePlan(weights_.contextProjection, lanes));
  // Every layer's key and value projection reads the same normalized rows,
  // prepared for the first layer's plan.
  ops::LinearPlan kvPlan = linear.decodePlan(contextKvProjections_[0], lanes);
  ops::PreparedInput hidden = ops::Normalization::addRms(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden, layout.hiddenSize, rows,
      scratch, kvPlan.input());

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const ops::Projection &projection = contextKvProjections_[layer];
    if (layer) kvPlan = linear.decodePlan(projection, lanes);
    hidden = linear.add(graph,
                        {.input = buffers.hidden, .output = buffers.contextKv, .scratch = scratch,
                         .prepared = hidden},
                        projection, kvPlan);
    ops::DraftAttention::addContextCommit(
        graph, buffers.contextKv, weights_.layers[layer].keyNorm, buffers.ropeCos,
        buffers.ropeSin, buffers.persistentKeys[layer],
        buffers.persistentValues[layer], buffers.retainedCounts,
        startPositions, layout.attentionShape());
  }
}

namespace {

// Reads a draft's files in their section order: each layer, then model.bin.
template <class Files>
DFlashDraftWeights readDraft(metal::MetalBackend &backend, Files &files,
                             const DFlashDraftLayout &layout) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  DFlashDraftWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const uint64_t convolutionBytes = checkedMultiply<WeightStoreError>(
      checkedMultiply<WeightStoreError>(SPLASH_DRAFT_CONVOLUTION_STAGES * SPLASH_DRAFT_CONVOLUTION_TAPS,
                                        layout.hiddenSize, "draft convolution elements"),
      kBFloat16Bytes, "draft convolution bytes");
  const uint64_t headNormBytes = checkedMultiply<WeightStoreError>(
      layout.attentionHeadDimension, kBFloat16Bytes,
      "draft head norm bytes");

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    WeightFile file = files.layer(layerIndex);
    DFlashDraftLayerWeights layer;
    layer.inputNorm = readNorm(file, layout.hiddenSize, false, "input-norm");
    layer.attentionConvolution =
        file.section(convolutionBytes, "attention-convolution");
    layer.attentionDynamic = readAffineProjection(
        file, layout.dynamicSize, layout.hiddenSize,
        "attention-dynamic");
    layer.qkvProjection = readAffineProjection(
        file, layout.qkvSize, layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readAffineProjection(
        file, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        readNorm(file, layout.hiddenSize, false, "post-attention-norm");
    layer.mlpConvolution = file.section(convolutionBytes, "mlp-convolution");
    layer.mlpDynamic = readAffineProjection(
        file, layout.dynamicSize, layout.hiddenSize, "mlp-dynamic");
    layer.gateProjection = readAffineProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = readAffineProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = readAffineProjection(
        file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
    file.finish();
    result.files.push_back(file.record());
    result.layers.push_back(std::move(layer));
  }

  {
    WeightFile file = files.model();
    result.contextProjection = readAffineProjection(
        file, layout.hiddenSize, layout.targetHiddenSize,
        "context-projection");
    result.hiddenNorm = readNorm(file, layout.hiddenSize, false, "hidden-norm");
    result.finalNorm = readNorm(file, layout.hiddenSize, false, "final-norm");
    result.selectorProjection = readAffineProjection(
        file, layout.selectorRank, layout.hiddenSize, "selector");
    const uint64_t codebookBytes = checkedMultiply<WeightStoreError>(
        checkedMultiply<WeightStoreError>(layout.vocabularySize,
                                          layout.selectorRank,
                                          "draft codebook elements"),
        kBFloat16Bytes, "draft codebook bytes");
    result.predecessorCodebook =
        file.section(codebookBytes, "predecessor-codebook");
    result.successorCodebook =
        file.section(codebookBytes, "successor-codebook");
    file.finish();
    result.files.push_back(file.record());
  }

  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace

WeightFile PackageDraftFiles::layer(uint32_t index) const {
  const std::string filename = "layer-" + std::to_string(index) + ".bin";
  return images.load(packageImage(directory / filename, "draft/" + filename,
                                  kDFlashLayerMagic, index, 0));
}

WeightFile PackageDraftFiles::model() const {
  return images.load(packageImage(directory / "model.bin", "draft/model.bin",
                                  kDFlashLayerMagic, layout.layers, 1));
}

DFlashDraftWeights loadDFlashDraftWeights(metal::MetalBackend &backend,
                                          const DraftFiles &files,
                                          DFlashDraftLayout layout) {
  requireLayout(layout);
  if (const auto *checkpoint =
          std::get_if<std::reference_wrapper<DraftCheckpointLoader>>(&files))
    return readDraft(backend, checkpoint->get(), layout);
  return readDraft(backend, std::get<PackageDraftFiles>(files), layout);
}

} // namespace splash::model
