// Modified by meowkernels.
#pragma once

#include "Model.hpp"
#include "StateLayout.hpp"
#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <filesystem>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace splash::model {

struct DFlashDraftRingLayer final {
  // K is [head][ring_position][dimension].
  metal::MetalBuffer keys;
  // V is [head][dimension][ring_position].
  metal::MetalBuffer values;
};

// Draft-owned persistent context, paired with target state in composite caches.
class DFlashDraftRing final {
public:
  DFlashDraftRing(metal::MetalBackend &backend,
                  std::shared_ptr<StateAllocationTracker> tracker,
                  DraftStateLayout layout,
                  std::string_view label);
  ~DFlashDraftRing();
  DFlashDraftRing(const DFlashDraftRing &) = delete;
  DFlashDraftRing &operator=(const DFlashDraftRing &) = delete;

  [[nodiscard]] const std::vector<DFlashDraftRingLayer> &layers() const noexcept {
    return layers_;
  }

private:
  std::shared_ptr<StateAllocationTracker> tracker_;
  std::vector<DFlashDraftRingLayer> layers_;
  uint64_t actualAllocatedBytes_ = 0;
};

// The dimensions of a DFlash2 draft. Each target's draft is defined beside
// the target (Qwen3_8.hpp, Qwen3_6Moe.hpp).
struct DFlashDraftLayout final {
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t dynamicSize = 0;
  uint32_t qkvSize = 0;
  uint32_t attentionSize = 0;
  uint32_t intermediateSize = 0;
  uint32_t attentionHeadDimension = 0;
  float rotaryTheta = 0.0F;
  uint32_t targetHiddenSize = 0;
  uint32_t selectorRank = 0;
  uint32_t kvHeads = 0;

  [[nodiscard]] constexpr DraftStateLayout stateLayout() const noexcept {
    return {layers, kvHeads, attentionHeadDimension};
  }
  [[nodiscard]] constexpr ops::DraftAttentionShape attentionShape() const noexcept {
    return {hiddenSize, dynamicSize, qkvSize, attentionSize,
            attentionSize / attentionHeadDimension, kvHeads,
            attentionHeadDimension};
  }
  // The key and value columns of the fused QKV projection, all the context
  // writers read.
  [[nodiscard]] constexpr uint32_t contextKvSize() const noexcept {
    return qkvSize - attentionSize;
  }

  bool operator==(const DFlashDraftLayout &) const = default;
};

struct DFlashDecodeBuffers final {
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer dynamic;
  metal::MetalBuffer convolved;
  metal::MetalBuffer proposalQkv;
  metal::MetalBuffer attention;
  metal::MetalBuffer projected;
  metal::MetalBuffer residual;
  metal::MetalBuffer intermediate;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer queryKeys;
  metal::MetalBuffer queryValues;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer gateScratch;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
};

struct DFlashContextBuffers final {
  ops::LinearScratch linearScratch{};
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer contextKv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer retainedCounts;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
};

struct DFlashPrefillSpan final {
  uint32_t compactRow = 0;
  uint32_t rows = 0;
  uint32_t startPosition = 0;
  std::span<const DFlashDraftRingLayer> ring;
};

struct DFlashPrefillBuffers final {
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer projectionSums;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer contextKv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
};

struct DFlashDraftLayerWeights final {
  ops::NormWeights inputNorm;
  metal::MetalBuffer attentionConvolution;
  ops::Projection attentionDynamic;
  ops::Projection qkvProjection;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  ops::Projection outputProjection;
  ops::NormWeights postAttentionNorm;
  metal::MetalBuffer mlpConvolution;
  ops::Projection mlpDynamic;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

struct DFlashDraftWeights final {
  DFlashDraftLayout layout;
  std::vector<DFlashDraftLayerWeights> layers;
  ops::Projection contextProjection;
  ops::NormWeights hiddenNorm;
  ops::NormWeights finalNorm;
  ops::Projection selectorProjection;
  metal::MetalBuffer predecessorCodebook;
  metal::MetalBuffer successorCodebook;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

inline constexpr std::string_view kDFlashLayerMagic = "MDFD0004";

// A Splash package's draft files: layer-<N>.bin and model.bin.
struct PackageDraftFiles final {
  WeightImages &images;
  std::filesystem::path directory;
  const DFlashDraftLayout &layout;
  [[nodiscard]] WeightFile layer(uint32_t index) const;
  [[nodiscard]] WeightFile model() const;
};

class DraftCheckpointLoader;

// The files a draft is read from: a package's files, or the images
// DraftCheckpointLoader writes from a DFlash2 checkpoint.
using DraftFiles = std::variant<PackageDraftFiles, std::reference_wrapper<DraftCheckpointLoader>>;

[[nodiscard]] DFlashDraftWeights
loadDFlashDraftWeights(metal::MetalBackend &backend, const DraftFiles &files,
                       DFlashDraftLayout layout);

// Builds the draft layer graph and its proposal selection
// (ops::DraftSelector) from batch buffers and persistent context; the
// target's sampling and acceptance policy remain outside the model.
class DFlashDraft final {
public:
  DFlashDraft(const DFlashDraftWeights &weights, metal::MetalBackend &backend,
               const ops::ExecutionPlans &operators);

  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans) const;

  // restrictedHead (SPLASH_DRAFT_HEAD_IDS, one lane): the head reads only the
  // restricted rows, so the logits are [rows][head rows] and the selection
  // maps their columns through the head's id map (ops::DraftHeadMap).
  // deviceAttentionParams (SPLASH_DRAFT_AHEAD): a GPU-written
  // attentionParams() bound in place of the host bytes.
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                 const ops::Projection &vocabularyProjection,
                 std::span<const uint32_t> cacheLengths,
                 bool restrictedHead = false,
                 metal::MetalBuffer deviceAttentionParams = {}) const;
  [[nodiscard]] static DraftAttentionBatchParams
  attentionParams(std::span<const uint32_t> cacheLengths) {
    return ops::DraftAttention::decodeParams(cacheLengths);
  }
  [[nodiscard]] SelectorBatchParams
  selectionParams(std::span<const uint32_t> anchors,
                  std::span<const ops::SamplingPolicy> policies) const {
    return selector_.params(anchors, policies);
  }
  void addSelection(metal::CommandGraph &graph,
                    ops::DraftSelectorBuffers buffers,
                    std::span<const uint32_t> anchors,
                    std::span<const ops::SamplingPolicy> policies,
                    bool restrictedHead = false) const;

  // Frequency-ranked restricted draft head (SPLASH_DRAFT_HEAD_IDS=path of
  // ranked u32 ids, SPLASH_DRAFT_HEAD_ROWS rows, default 98304; off when
  // unset). The first rows - 256 ranked ids form the static head; each request
  // fills the last 256 rows (its segment) with its rare prompt tokens, then
  // filler. A gathered copy of the target head's rows, bytes unchanged.
  static constexpr uint32_t kHeadSegmentRows = 256;
  // Bytes the restricted head allocates (0 when unset), each buffer rounded
  // to 16 KiB pages; the memory plan counts them with the draft weights.
  [[nodiscard]] static uint64_t restrictedHeadPlannedBytes(const DFlashDraftLayout &layout);
  // Whether the restricted head can gather its rows from `target`, the
  // target's vocabulary head: only from the full affine Q4 head, not from a
  // GGUF target's block-quantized one. Startup plans no restricted head
  // otherwise, and every request drafts with the full head.
  [[nodiscard]] static bool gathersRestrictedHead(const DFlashDraftLayout &layout,
                                                  const ops::Projection &target) noexcept;
  // Gathers the restricted head at startup (it is planned), before warmup.
  void loadRestrictedHead(const ops::Projection &target) const;
  // The memory plan does not carry it (the target head cannot be gathered or
  // the copy did not fit): every request drafts with the full head.
  void disableRestrictedHead() noexcept;
  [[nodiscard]] uint64_t restrictedHeadAllocatedBytes() const noexcept;
  [[nodiscard]] bool hasRestrictedHead() const noexcept { return restrictedHead_.has_value(); }
  // The 256 ids a request adds to the static rows, or empty when unconfigured
  // or the prompt has too many rare tokens (full head). *promptIds receives
  // how many leading ids came from the prompt; the rest is filler that rare
  // output tokens may replace.
  [[nodiscard]] std::vector<uint32_t> promptSegment(std::span<const uint32_t> prompt,
                                                    uint32_t *promptIds = nullptr) const;
  [[nodiscard]] bool staticHeadHas(uint32_t token) const noexcept {
    return token < headKeep_.size() && headKeep_[token];
  }
  // Loads a request's segment (owner, version) into the head's last 256 rows.
  void useHeadSegment(uint64_t owner, uint64_t version, std::span<const uint32_t> segment,
                      const ops::Projection &target) const;
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions) const;

private:
  const ops::Projection &draftHead(const ops::Projection &target) const;
  void copyHeadRows(const ops::Projection &target, std::span<const uint32_t> ids,
                    uint32_t firstRow) const;

  const DFlashDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  ops::DraftSelector selector_;
  // Each layer's key and value rows of its QKV projection, views of its
  // planes, which the context writers project with.
  std::vector<ops::Projection> contextKvProjections_;
  // SPLASH_DRAFT_HEAD_IDS: static ids (ascending), membership, filler ids.
  std::vector<uint32_t> headIds_;
  std::vector<uint8_t> headKeep_;
  std::vector<uint32_t> headFiller_;
  mutable uint64_t segmentOwner_ = 0;
  mutable uint64_t segmentVersion_ = 0;
  mutable std::vector<uint32_t> loadedSegment_;
  mutable std::optional<ops::Projection> restrictedHead_;
  mutable metal::MetalBuffer headIdMap_;
};

} // namespace splash::model
