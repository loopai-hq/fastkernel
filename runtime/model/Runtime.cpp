// Modified by Pulsar.
#include "model/Runtime.hpp"
#include "metal/EnvSwitch.hpp"
#include "AwakeClock.hpp"
#include "model/QwenState.hpp"
#include "model/QwenTarget.hpp"
#include "model/PromptLookup.hpp"
#include "model/RuntimeArenas.hpp"

#include "metal/CommandGraph.hpp"
#include "metal/EnvSwitch.hpp"
#include "ops/AneFfn.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"
#include "ops/RoPE.hpp"
#include "ops/RowCopy.hpp"
#include "ops/Sampling.hpp"
#include "ops/Vision.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

using metal::BufferStorage;
using metal::CommandGraph;
using metal::CommandTicket;
using metal::CommandTiming;
using metal::MetalBackend;
using metal::MetalBuffer;

class DeferredMetalTicket final : public ModelBatchTicket {
public:
  // Runs once the command has completed, and may add work of its own to its
  // timing, as a rerun of the command does.
  using Completion = std::function<std::vector<ModelStepResult>(CommandTiming &)>;

  DeferredMetalTicket(CommandTicket ticket, Completion completion,
                      bool representativePrefillTiming = true)
      : ticket_(std::move(ticket)), completion_(std::move(completion)),
        representativePrefillTiming_(representativePrefillTiming) {}

  bool ready() const noexcept override { return ticket_.ready(); }

  std::vector<ModelStepResult> wait() override {
    if (!completion_) {
      throw std::logic_error("Metal ticket was already consumed");
    }
    CommandTiming timing = ticket_.wait();
    const double commandSeconds = timing.wallSeconds;
    Completion completion = std::move(completion_);
    std::vector<ModelStepResult> results = completion(timing);
    wallMilliseconds_ = timing.wallSeconds * 1000.0;
    // Work the completion added takes wall time, but is no sample of the
    // command's rows.
    if (timing.wallSeconds != commandSeconds)
      representativePrefillTiming_ = false;
    return results;
  }

  double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }
  bool prefillTimingIsRepresentative() const noexcept override {
    return representativePrefillTiming_;
  }

private:
  CommandTicket ticket_;
  Completion completion_;
  double wallMilliseconds_ = 0.0;
  bool representativePrefillTiming_;
};

using kv::ChunkedPrefillParams;

bool isStopToken(const RuntimeGeometry &geometry, uint32_t token) noexcept {
  return token == geometry.target.stopTokens[0] ||
         token == geometry.target.stopTokens[1];
}

void requireShared(const MetalBuffer &buffer, std::string_view label) {
  if (!buffer || buffer.storage() != BufferStorage::Shared ||
      !buffer.contents()) {
    throw std::logic_error(std::string(label) + " is not CPU-visible");
  }
}

template <class T>
T *contents(const MetalBuffer &buffer, std::string_view label) {
  requireShared(buffer, label);
  return static_cast<T *>(buffer.contents());
}

void validatePlan(const BatchPlan &plan, std::span<const ModelBatchItem> items,
                  WorkKind expected) {
  if (plan.kind != expected || plan.empty() || plan.width() > kLaneCount ||
      items.size() != plan.items.size()) {
    throw std::invalid_argument("model runtime received an invalid batch plan");
  }
  for (size_t index = 0; index < items.size(); ++index) {
    if (items[index].requestId != plan.items[index].requestId ||
        (expected == WorkKind::Prefill &&
         (!plan.items[index].tokenCount ||
          plan.items[index].tokenCount != items[index].tokenCount ||
          items[index].inputTokens.size() != items[index].tokenCount)) ||
        (expected == WorkKind::Decode &&
         (plan.items[index].tokenCount || items[index].tokenCount ||
          !items[index].inputTokens.empty()))) {
      throw std::invalid_argument("batch items do not match explicit plan");
    }
  }
}

// The lane a start's admission gave it, or the cause of its refusal.
StateAdmission laneAdmission(uint32_t lane, const metal::AllocationResult &result) {
  if (result)
    return {lane, StateFailure::None};
  return {{}, StateFailure::MemoryPressure, result.failure};
}

// Any unassigned lane works: its buffers come from the storage's pool, and
// the governor is asked only for what the pool lacks.
template <class Activate>
StateAdmission admitIdleLane(const QwenStateStorage &states,
                             Activate activate) {
  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    if (!states.metadata(lane).assigned())
      return activate(lane);
  }
  return {{}, StateFailure::ConcurrencyLimit};
}

// A prefill arena's tensors as the target's prefill reads them.
QwenTargetPrefillBuffers prefillBuffers(const PrefillArena &arena) {
  QwenTargetPrefillBuffers buffers;
  // Prefill plans read plain bf16 rows, so there is no input table or sums.
  buffers.linearScratch = {.partials = arena.get(PrefillTensor::LinearPartials),
                           .counters = arena.get(PrefillTensor::LinearCounters),
                           .rotated = arena.get(PrefillTensor::LinearRotated)};
  buffers.hidden = {arena.get(PrefillTensor::Hidden0), arena.get(PrefillTensor::Hidden1)};
  buffers.normalized = arena.get(PrefillTensor::Normalized);
  buffers.captured = arena.get(PrefillTensor::Captured);
  buffers.gdnPacked = arena.get(PrefillTensor::GdnPacked);
  buffers.gdnQueries = arena.get(PrefillTensor::GdnQueries);
  buffers.gdnKeys = arena.get(PrefillTensor::GdnKeys);
  buffers.gdnValues = arena.get(PrefillTensor::GdnValues);
  buffers.gdnDecay = arena.get(PrefillTensor::GdnDecay);
  buffers.gdnBeta = arena.get(PrefillTensor::GdnBeta);
  buffers.recurrent = arena.get(PrefillTensor::Recurrent);
  buffers.gdnHidden = arena.get(PrefillTensor::GdnHidden);
  buffers.gdnOutput = arena.get(PrefillTensor::GdnOutput);
  buffers.denseGateScratch = arena.get(PrefillTensor::GateIntermediate);
  buffers.denseIntermediate = arena.get(PrefillTensor::Intermediate);
  buffers.fullPacked = arena.get(PrefillTensor::FullPacked);
  buffers.fullQueries = arena.get(PrefillTensor::FullQueries);
  buffers.fullAttention = arena.get(PrefillTensor::FullAttention);
  buffers.attentionPartials = arena.get(PrefillTensor::AttentionPartials);
  buffers.attentionStatistics = arena.get(PrefillTensor::AttentionStatistics);
  buffers.attentionHidden = arena.get(PrefillTensor::AttentionHidden);
  buffers.attentionOutput = arena.get(PrefillTensor::AttentionOutput);
  buffers.projectionSums = arena.get(PrefillTensor::ProjectionSums);
  buffers.downProjectionSums = arena.get(PrefillTensor::DownProjectionSums);
  buffers.ropeCos = arena.get(PrefillTensor::RopeCos);
  buffers.ropeSin = arena.get(PrefillTensor::RopeSin);
  buffers.chunkKeys = arena.get(PrefillTensor::ChunkKeys);
  buffers.chunkValues = arena.get(PrefillTensor::ChunkValues);
  buffers.moe = arena.moeScratch();
  return buffers;
}

} // namespace

struct Runtime::Impl {
  // An image by content: the fields a placement's span identifies it by.
  struct ImageKey final {
    uint64_t digestLo = 0;
    uint64_t digestHi = 0;
    uint32_t gridHeight = 0;
    uint32_t gridWidth = 0;

    bool operator==(const ImageKey &) const = default;
  };
  struct ImageKeyHash final {
    // The digest is already a content hash.
    size_t operator()(const ImageKey &key) const noexcept {
      return static_cast<size_t>(key.digestLo ^ key.digestHi);
    }
  };

  // One image's encoded rows, shared by every placement that still has rows
  // to inject (repeated placements and concurrent requests alike) and by
  // the embedding cache. Whichever placement's chunk reaches the image
  // first encodes it and the others inject after it; the pixels go once
  // the encode has completed.
  struct ImageRows final {
    ImageKey key;
    MetalBuffer pixels;
    MetalBuffer embeddings;
    bool encoding = false;
    bool encoded = false;
    // Its entry in the embedding cache while the cache holds it.
    std::optional<std::list<std::shared_ptr<ImageRows>>::iterator> cached;
  };

  // A placement keeps its rows until its last row is injected; its span
  // stays, because rotary positions after it depend on its grid.
  struct ImageState final {
    ImageSpan span;
    std::shared_ptr<ImageRows> rows;
  };

  struct Request final {
    uint64_t id = 0;
    uint32_t stateLane = 0;
    // SPLASH_GDN_DEFER: this cycle defers its recurrent GDN commit (decided
    // before its graph), with its rows at verify scratch lane slot gdnDeferSlot.
    bool gdnDeferCycle = false;
    uint32_t gdnDeferSlot = 0;
    bool resident = false;
    bool promptComplete = false;
    // Rebuild state from already-emitted tokens without sampling an initial
    // anchor, consuming RNG, or replaying output to the caller.
    bool replayingGeneration = false;
    uint32_t promptTokens = 0;
    uint32_t maxNewTokens = 0;
    uint32_t generatedTokens = 0;
    SamplingParameters sampling;
    ConstraintMode constraint = ConstraintMode::None;
    // RequestFlag bits.
    uint32_t flags = 0;
    std::optional<uint32_t> pendingToken;
    // A constrained request's final prompt row, held from its prompt's end
    // until its first token is selected under its first mask, suspensions
    // included. Composite cache state never stores it; every cache hit
    // replays one input token and regenerates this value.
    std::vector<uint16_t> finalTargetHidden;
    std::array<float, kSamplingUniformCount> cycleUniforms{};
    // Nonempty selects score-only mode: the final prefill chunk computes raw
    // logits at these token ids instead of selecting an anchor.
    std::vector<uint32_t> scoreTokens;
    std::vector<uint32_t> maskWords;
    // Set only while the current scheduler-owned ticket overlaps grammar-mask
    // computation with target verification. This is model runtime state, not a
    // scheduler decode stage.
    bool verifyMaskInFlight = false;
    uint64_t rngCounter = 0;
    DecodeStage decodeStage = DecodeStage::Regular;
    std::optional<DraftContextPlan> draftContextPlan;
    std::vector<ImageState> images;
    // What its activation took from a cached state: the images that end
    // there were left out (ModelRequest::restoredTokens).
    uint32_t restoredTokens = 0;
    // Pulsar prompt lookup (SPLASH_PROMPT_LOOKUP): the prompt's last
    // kPromptLookupHistory tokens, then every emitted token; empty when the
    // request does not look up. Per-request counters for end()'s report and
    // SPLASH_LOOKUP_ADAPTIVE's match threshold.
    std::vector<uint32_t> lookupHistory;
    uint64_t lookupCycles = 0;
    uint64_t lookupHits = 0;
    uint64_t lookupRetained = 0;
    size_t lookupMatch = 16;
    // SPLASH_WIDE_PROMPT_LOOKUP: this cycle's verify rows (8, or 16/32 for a
    // wide lookup), its wide cycles, whether it may verify 32 rows
    // (SPLASH_WIDE_LOOKUP32) and whether its last cycle was a fully accepted
    // wide lookup, and its wide GDN route (SPLASH_WIDE_GDN_SINGLE).
    uint32_t verifyRows = kDecodeRows;
    uint64_t wideLookupHits = 0;
    uint64_t wideLookupRetained = 0;
    uint64_t wide32Hits = 0;
    bool wide32 = false;
    bool wideFull = false;
    ops::WideGdn wideGdn = ops::WideGdn::Chain;
    uint64_t wideGdnCycles = 0;
    // SPLASH_DRAFT_HEAD_IDS: the request's 256 restricted-head rows (empty:
    // full head), how many lead with prompt/output ids (the rest is filler),
    // and a version bumped on every change.
    std::vector<uint32_t> draftHeadSegment;
    uint32_t draftHeadSegmentUsed = 0;
    uint64_t draftHeadSegmentVersion = 0;
    // SPLASH_DRAFT_AHEAD blocks launched behind / adopted by its cycles, and
    // its cycles since its last prompt-lookup cycle (none yet: max).
    uint64_t aheadLaunched = 0;
    uint64_t aheadAdopted = 0;
    uint32_t cyclesSinceLookup = std::numeric_limits<uint32_t>::max();
  };

  struct DecodeLaneResult final {
    Request *request = nullptr;
    uint32_t retained = 0;
    uint32_t accepted = 0;
    uint32_t currentAnchor = 0;
    uint32_t maximumRetained = 0;
    // Why the lane's selection is unusable (invalidSelection), found before
    // any lane commits.
    std::string failure;
    // The cycle verified a prompt lookup's proposals instead of the draft's,
    // as a wide lookup over wideTiles aliased 8-row lanes (2 or 4).
    bool promptLookup = false;
    bool wideLookup = false;
    uint32_t wideTiles = 0;
  };

  // What a lane's GPU table was last written from. Its entries stay valid
  // while the revision does: KvPool never releases the extent of a page a
  // request holds (PageStorage::releaseExtent).
  struct PageTableBinding final {
    uint64_t requestId = 0;
    uint64_t revision = 0;
  };

  MetalBackend &backend;
  const LoadedModel &model;
  const RuntimeGeometry geometry;
  const ops::ExecutionPlans &operators;
  kv::PageStorage &kvPages;
  QwenStateStorage &states;
  std::unique_ptr<PrefillArena> prefillArena;
  std::unique_ptr<DecodeArena> decodeArena;
  // Pulsar SPLASH_STREAMED_SUBMIT=N (default 48; 0 = off; read per
  // cycle): a decode command's first N dispatches commit as soon as they are
  // built, while the host builds and encodes the rest. Exact: building a
  // decode graph writes no buffer (page tables, uniforms and lane buffers
  // are written before it; masks behind the grammar chain's wait).
  static size_t streamChunk() noexcept {
    const char *value = std::getenv("SPLASH_STREAMED_SUBMIT");
    return value ? std::strtoul(value, nullptr, 10) : 48;
  }
  // Pulsar SPLASH_GRAMMAR_CHAIN: the event a chained constrained cycle
  // waits on, and the values taken from it so far.
  metal::SharedEvent chainEvent;
  uint64_t chainValues = 0;
  // Abandons, on unwind, a head streamed from a graph whose command never
  // came (its build threw); a no-op once the command was submitted.
  struct StreamedHeadGuard final {
    MetalBackend &backend;
    ~StreamedHeadGuard() { backend.abandonStreamedHead(); }
  };
  // Every state lane's penalty words, bound whole: a batch lane reads the row
  // of its request's state lane, which need not be its own.
  MetalBuffer penaltyTable;
  std::unordered_map<uint64_t, Request> requests;
  // SPLASH_LOOKUP_MIN_MATCH (default 8): the shortest repeated suffix that may
  // replace the draft with a lookup proposal. With SPLASH_LOOKUP_ADAPTIVE
  // (default on) it is the floor of each request's threshold, which starts at
  // 16 (PromptLookup.hpp adaptLookupMatch); texts 18/18 identical (1.0.0).
  static size_t lookupMinMatchFromEnv(const char *name, size_t fallback) {
    const char *value = std::getenv(name);
    return value ? std::max<size_t>(2, std::strtoul(value, nullptr, 10)) : fallback;
  }
  const size_t lookupMinMatch = lookupMinMatchFromEnv("SPLASH_LOOKUP_MIN_MATCH", 8);
  const bool adaptiveLookup = metal::envSwitch("SPLASH_LOOKUP_ADAPTIVE");
  // SPLASH_PROMPT_LOOKUP (default on): a single-request cycle may propose the
  // seven tokens that followed the latest earlier occurrence of its context's
  // suffix instead of running the draft. The proposal is a point mass (q = 1),
  // so greedy and sampled acceptance keep the target's distribution exactly;
  // with 16 rows: rewrite +52%, edit +24% (Pulsar 1.0.0, measured).
  const bool wideLookupEnabled = widePromptLookupEnabled();
  const bool promptLookupEnabled = wideLookupEnabled || metal::envSwitch("SPLASH_PROMPT_LOOKUP");
  // SPLASH_WIDE_LOOKUP_MIN_MATCH (default 16): the shortest suffix a wide
  // lookup takes without SPLASH_LOOKUP_ADAPTIVE.
  const size_t wideLookupMinMatch = lookupMinMatchFromEnv("SPLASH_WIDE_LOOKUP_MIN_MATCH", 16);
  const char *const wide32Mode = wideLookup32Mode();
  uint64_t wide32Admitted = 0;  // =alt: requests admitted so far
  // SPLASH_WIDE_GDN_SINGLE (default parts): a wide lookup's GDN as one dispatch
  // per layer (=1) or as that pass in four value parts plus a finalize
  // (=parts, VH48 layers only); =alt / =alt-parts give it to every other
  // request (in-run A/B); any other value (e.g. 0) keeps the per-tile chain.
  // parts vs the chain: code-edit +2.39%, tool-copy +2.74%, exact (1.0.0).
  const std::string_view wideGdnMode = [] {
    const char *value = std::getenv("SPLASH_WIDE_GDN_SINGLE");
    return value ? std::string_view(value) : std::string_view("parts");
  }();
  const ops::WideGdn wideGdnRoute =
      wideGdnMode == "parts" || wideGdnMode == "alt-parts" ? ops::WideGdn::SingleParts
      : wideGdnMode == "1" || wideGdnMode == "alt"         ? ops::WideGdn::Single
                                                            : ops::WideGdn::Chain;
  const bool wideGdnAlternate = wideGdnMode == "alt" || wideGdnMode == "alt-parts";
  uint64_t wideGdnAdmitted = 0;
  // SPLASH_ROW_HASH=1 (diagnostic, default off): finalizeDecode prints a hash
  // of the final hidden and logits of every accepted-path verify row (rows
  // 0..accepted, whose inputs are committed tokens), keyed by position, so
  // runs with different cycle schedules can be compared row by row.
  const bool rowHash = [] {
    const char *value = std::getenv("SPLASH_ROW_HASH");
    return value && std::string_view(value) == "1";
  }();
  // SPLASH_REQUEST_STATS=1 (diagnostic, default off): Runtime::end prints a
  // request's draft-ahead, prompt-lookup and wide-lookup counters, and the
  // runtime's start its wide lookup's row-stable verify rows.
  const bool requestStats = [] {
    const char *value = std::getenv("SPLASH_REQUEST_STATS");
    return value && std::string_view(value) == "1";
  }();
  // Allocated for images that need an encode, sized for the largest one the
  // start that built it staged, and reclaimable once no image waits for one
  // and no refused start holds it. Injecting already encoded rows needs no
  // vision arena.
  std::shared_ptr<ops::Vision> vision;
  // Every image's rows while anything holds them, so that a placement of
  // the same image anywhere shares them. Entries of rows nothing holds any
  // more go when a lookup or a walk finds them.
  std::unordered_map<ImageKey, std::weak_ptr<ImageRows>, ImageKeyHash> imageRows;
  // Encoded rows kept for reuse once no placement has rows of them left to
  // inject, including prefix hits that land inside an image and still need
  // its remaining rows. Most recently used first, bounded by bytes; the
  // memory reclaimer drops the least recently used entry nothing else holds.
  static constexpr uint64_t kEmbeddingCacheBytes = 512ULL * 1024 * 1024;
  std::list<std::shared_ptr<ImageRows>> embeddingCache;
  uint64_t embeddingCacheBytes = 0;
  // A state in RAM that resumes inside an image, with the rows it needs: its
  // boundary lies less than a page before the image's end, so a restore
  // there injects the image's last rows. The pointer the cache keeps owns
  // both, so the rows go with the state's RAM copy, which the cache drops
  // when it evicts the state or writes it to disk. Held rows and the
  // embedding cache together keep at most kEmbeddingCacheBytes of rows.
  struct HeldState final {
    std::shared_ptr<const CompositeState> state;
    std::shared_ptr<ImageRows> rows;
  };
  std::vector<std::weak_ptr<const HeldState>> stateHolds;
  std::array<PageTableBinding, kLaneCount> pageTableBindings{};
  ModelTelemetry counters;
  ops::Sampling sampling;
  QwenTarget targetModel;
  DFlashDraft draftModel;
  ops::AneFfn *aneFfn;
  // The widest wide lookup this device keeps row-exact (8: none).
  uint32_t stableVerifyRows = kDecodeRows;
  // SPLASH_GDN_DEFER (Pulsar; default on; read per cycle): a one-lane value-parts
  // verify commits only the convolution carry and leaves its recurrent rows
  // pending; the request's next such verify replays them in its scan, from the
  // state before them (the lane's next cell) into its current cell, and scans
  // on from there. Anything else that reads the state first lands them with a
  // flush command of its own (flushGdn): a step that is not that request's
  // next one-lane verify, a prefill, a snapshot. A pending request's rows wait
  // in the verify scratch's lane slot 0 or 1 of each layer, alternating, so
  // the next scan's prologue writes the other slot.
  struct GdnPending final {
    uint64_t requestId = 0;
    uint32_t stateLane = 0;
    uint32_t rows = 0;
    uint32_t slot = 0;
  };
  std::optional<GdnPending> gdnPending;
  std::optional<bool> gdnDeferRoute;
  // SPLASH_DRAFT_AHEAD: the block in flight behind the last decode command.
  struct DraftAhead final {
    uint32_t lanes = 0;
    std::array<uint64_t, kLaneCount> requestIds{};
    // Logical positions of the launching cycle; + retained once finalized.
    std::array<uint64_t, kLaneCount> positions{};
    // Each lane's pending token once finalized.
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<uint64_t, kLaneCount> headVersions{};
    // A sampled lane's next-cycle uniforms (zero for a greedy lane).
    std::array<std::array<float, kSamplingUniformCount>, kLaneCount> uniforms{};
    bool restrictedHead = false;
    bool finalized = false;
  };
  std::optional<DraftAhead> ahead;
  CommandGraph aheadGraph;
  // The arena's DraftAheadParams (empty when SPLASH_DRAFT_AHEAD was off at startup).
  MetalBuffer aheadParams;
  // SPLASH_DRAFT_AHEAD_LOOKUP_QUIET=K (default 8): launch a block only after K
  // cycles without a prompt-lookup cycle. Lookup hits cluster, and a block a
  // lookup cycle drops costs a wait for it against ~0.3 ms an adopted one
  // saves (Pulsar 1.0.0).
  const uint32_t aheadLookupQuiet = [] {
    const char *value = std::getenv("SPLASH_DRAFT_AHEAD_LOOKUP_QUIET");
    return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 8u;
  }();
  explicit Impl(RuntimeContext value)
      : backend(value.backend),
        model(value.model),
        geometry(RuntimeGeometry::from(value.model, value.kvPages.layout().format)),
        operators(value.operators),
        kvPages(value.kvPages),
        states(value.stateStorage),
        sampling(geometry.target.vocabularySize),
        targetModel(std::visit(
                        [&](const auto &weights) {
                          return QwenTarget(weights, geometry.target,
                                            value.backend, operators);
                        },
                        value.model.target)),
        draftModel(value.model.draft, value.backend, operators),
        aneFfn(value.aneFfn) {
    if (states.layout() != model.stateLayout() ||
        kvPages.layout() != model.targetKvLayout(kvPages.layout().format)) {
      throw std::invalid_argument(
          "model runtime resources do not match the loaded model");
    }
    prefillArena = std::make_unique<PrefillArena>(backend, geometry, operators);
    decodeArena = std::make_unique<DecodeArena>(backend, geometry, operators);
    penaltyTable =
        decodeArena->batchSlice(DecodeTensor::PenaltyState, kLaneCount);
    aheadParams = decodeArena->get(0, DecodeTensor::DraftAheadParams);
    preparePolicyPipelines();
    if (wideLookupEnabled && decodeArena->wideConvolutionScratch()) {
      stableVerifyRows = targetModel.rowStableVerifyRows();
      if (requestStats)
        std::fprintf(stderr, "row-stable verify rows: %u\n", stableVerifyRows);
    }
    // SPLASH_DRAFT_HEAD_IDS: the memory plan carries the restricted draft head
    // with the draft weights, so it is gathered now, before warmup and the
    // memory audit.
    if (value.restrictedDraftHead)
      draftModel.loadRestrictedHead(targetModel.vocabularyProjection());
    else
      draftModel.disableRestrictedHead();
    // Its selection's head-id map (draft_map_head_ids) compiles now, not
    // inside the first restricted-head request.
    if (draftModel.hasRestrictedHead()) {
      auto d = [&](DecodeTensor tensor) { return decodeArena->batchSlice(tensor, 1); };
      CommandGraph graph;
      const std::array<uint32_t, 1> anchor{0};
      const std::array<ops::SamplingPolicy, 1> policy{};
      draftModel.addSelection(
          graph,
          {d(DecodeTensor::Logits), d(DecodeTensor::TopPartialIds),
           d(DecodeTensor::TopPartialValues), d(DecodeTensor::Candidates),
           d(DecodeTensor::Unary), d(DecodeTensor::SelectorHidden),
           d(DecodeTensor::SamplingUniforms), d(DecodeTensor::ProposedTokens),
           d(DecodeTensor::ProposalProbs)},
          anchor, policy, true);
      backend.preparePipelines(graph.dispatches());
    }
  }

  // Warmup selects greedily, so the first sampled, penalized or constrained
  // request would compile the policy's kernels inside its TTFT and stall the
  // engine meanwhile; compile them now. A sampled, penalized and constrained
  // lane and a greedy one reach every kernel the first-token and verify
  // selections dispatch.
  void preparePolicyPipelines() const {
    const ops::SamplingPolicy sampled{.topK = 0,
                                      .temperature = 1.0F,
                                      .topP = 0.95F,
                                      .constrained = true,
                                      .penalties = {1.1F, 0.5F, 0.5F},
                                      .minP = 0.05F};
    const std::array<ops::SamplingPolicy, 2> policies{sampled, {}};
    const std::array<uint32_t, 2> stateLanes{0, 1};
    const ops::PenaltyTable penalties{penaltyTable, stateLanes};
    CommandGraph graph;
    sampling.addInitial(graph, policies, samplingBuffers(2), 0,
                        geometry.target.stopTokens[0],
                        geometry.target.stopTokens[1], penalties);
    sampling.addVerify(graph, policies, samplingBuffers(2),
                       geometry.target.stopTokens[0],
                       geometry.target.stopTokens[1], penalties);
    // SPLASH_BLOCK_VERIFY: a verify batch whose lanes all sample draws,
    // accepts and corrects with the block kernels.
    const auto blockPolicies = std::span(policies).first(1);
    const std::array<uint32_t, 1> retained{kDecodeRows};
    sampling.addVerify(graph, blockPolicies, samplingBuffers(1),
                       geometry.target.stopTokens[0],
                       geometry.target.stopTokens[1],
                       {penaltyTable, std::span(stateLanes).first(1)});
    sampling.addAcceptance(graph, acceptanceBuffers(1), retained, blockPolicies,
                           geometry.target.stopTokens[0],
                           geometry.target.stopTokens[1]);
    // SPLASH_SAMPLER_TOPK32: a sampled lane with a top-k of at most 32 and
    // no min-p searches its shards' top tokens.
    const std::array<ops::SamplingPolicy, 1> topTokens{
        {{.topK = 20, .temperature = 1.0F, .topP = 0.95F}}};
    sampling.addInitial(graph, topTokens, samplingBuffers(1), 0,
                        geometry.target.stopTokens[0],
                        geometry.target.stopTokens[1],
                        {penaltyTable, std::span(stateLanes).first(1)});
    backend.preparePipelines(graph.dispatches());
  }

  Request &request(uint64_t id) {
    auto found = requests.find(id);
    if (found == requests.end())
      throw std::out_of_range("unknown request");
    return found->second;
  }

  static bool samplingEnabled(const Request &entry) noexcept {
    return entry.sampling.temperature > 0.0F;
  }

  // Qwen3.5 M-RoPE: text rows advance one counter shared by all three axes;
  // an image's rows spread over (t, h, w) from the counter at the image start
  // and the counter then advances by max(merged height, merged width).
  static std::array<uint32_t, 3> ropePosition(const Request &entry,
                                              uint64_t logical) {
    int64_t delta = 0;
    for (const ImageState &image : entry.images) {
      const ImageSpan &span = image.span;
      if (logical < span.offset)
        break;
      const uint32_t mergedHeight = span.gridHeight / 2;
      const uint32_t mergedWidth = span.gridWidth / 2;
      const uint32_t start =
          static_cast<uint32_t>(static_cast<int64_t>(span.offset) + delta);
      if (logical < span.end()) {
        const uint32_t local = static_cast<uint32_t>(logical - span.offset);
        return {start, start + local / mergedWidth,
                start + local % mergedWidth};
      }
      delta += static_cast<int64_t>(std::max(mergedHeight, mergedWidth)) -
               static_cast<int64_t>(span.tokens);
    }
    const uint32_t position =
        static_cast<uint32_t>(static_cast<int64_t>(logical) + delta);
    return {position, position, position};
  }

  uint64_t embeddingBytes(const ImageSpan &span) const {
    return uint64_t{ops::Vision::embeddingRows(span.grid())} *
           geometry.target.hiddenSize * sizeof(uint16_t);
  }

  static ImageKey imageKey(const ImageSpan &span) noexcept {
    return {span.digestLo, span.digestHi, span.gridHeight, span.gridWidth};
  }

  // The rows of an identical image that something still holds, moved to the
  // front of the embedding cache when it is there; null otherwise.
  std::shared_ptr<ImageRows> findRows(const ImageSpan &span) {
    const auto found = imageRows.find(imageKey(span));
    if (found == imageRows.end())
      return {};
    std::shared_ptr<ImageRows> rows = found->second.lock();
    if (!rows) {
      imageRows.erase(found);
      return {};
    }
    if (rows->cached)
      embeddingCache.splice(embeddingCache.begin(), embeddingCache, *rows->cached);
    return rows;
  }

  // Keeps encoded rows for reuse as the most recently used, dropping the
  // least recently used while the rows kept for reuse exceed the cache's
  // bytes.
  void retain(const std::shared_ptr<ImageRows> &rows) {
    if (rows->cached) {
      embeddingCache.splice(embeddingCache.begin(), embeddingCache, *rows->cached);
      return;
    }
    const uint64_t bytes = rows->embeddings.sizeBytes();
    embeddingCache.push_front(rows);
    rows->cached = embeddingCache.begin();
    embeddingCacheBytes += bytes;
    while (!embeddingCache.empty() &&
           embeddingCacheBytes + heldRowsBytes(true) > kEmbeddingCacheBytes)
      static_cast<void>(uncache(std::prev(embeddingCache.end())));
  }

  // The bytes of the distinct rows states in RAM hold: all of them, or only
  // those the embedding cache does not hold as well.
  [[nodiscard]] uint64_t heldRowsBytes(bool uncachedOnly) const noexcept {
    uint64_t bytes = 0;
    for (auto hold = stateHolds.begin(); hold != stateHolds.end(); ++hold) {
      const std::shared_ptr<const HeldState> held = hold->lock();
      if (!held || (uncachedOnly && held->rows->cached))
        continue;
      const bool counted = std::any_of(
          stateHolds.begin(), hold, [&](const std::weak_ptr<const HeldState> &earlier) {
            const std::shared_ptr<const HeldState> other = earlier.lock();
            return other && other->rows == held->rows;
          });
      if (!counted)
        bytes += held->rows->embeddings.sizeBytes();
    }
    return bytes;
  }

  // A state in RAM whose boundary lies inside an image, less than a page
  // before its end, holds the image's encoded rows, unless that would take
  // the rows kept for reuse past the cache's bytes: the state returned owns
  // them. Boundaries deeper inside an image keep only the embedding cache.
  std::shared_ptr<const CompositeState>
  holdStraddledRows(const Request &entry, std::shared_ptr<const CompositeState> state) {
    std::erase_if(stateHolds, [](const std::weak_ptr<const HeldState> &hold) {
      return hold.expired();
    });
    const uint64_t boundary = states.metadata(entry.stateLane).lengths.targetTokens;
    for (const ImageState &image : entry.images) {
      // The chunk that ended at the boundary encoded the image it reached.
      if (image.span.offset >= boundary || image.span.end() <= boundary)
        continue;
      if (image.span.end() - boundary >= kv::kPageTokens)
        break;
      const bool kept =
          image.rows->cached ||
          std::ranges::any_of(stateHolds, [&](const std::weak_ptr<const HeldState> &hold) {
            const std::shared_ptr<const HeldState> held = hold.lock();
            return held && held->rows == image.rows;
          });
      const uint64_t added = kept ? 0 : image.rows->embeddings.sizeBytes();
      if (embeddingCacheBytes + heldRowsBytes(true) + added > kEmbeddingCacheBytes)
        break;
      auto held = std::make_shared<const HeldState>(HeldState{std::move(state), image.rows});
      stateHolds.push_back(held);
      return {held, held->state.get()};
    }
    return state;
  }

  // Drops one entry of the embedding cache and returns the bytes it held.
  uint64_t uncache(std::list<std::shared_ptr<ImageRows>>::iterator entry) noexcept {
    const uint64_t bytes = (*entry)->embeddings.sizeBytes();
    (*entry)->cached.reset();
    embeddingCache.erase(entry);
    embeddingCacheBytes -= bytes;
    return bytes;
  }

  // A request lets go of its images; the encoded ones stay in the cache.
  void releaseImages(Request &entry) {
    for (const ImageState &image : entry.images) {
      if (image.rows && image.rows->encoded)
        retain(image.rows);
    }
    entry.images.clear();
  }

  // Frees one cache that can be rebuilt and returns its bytes. The vision
  // arena goes first, when no image waits for its encode and nothing holds
  // it: an image whose rows are encoded never needs it, and the next start
  // that does builds one sized for its own images. Then the least recently
  // used embedding entry nothing else holds, one at a time, since only an
  // encode rebuilds it. An entry something else holds is skipped, since
  // dropping it frees nothing.
  uint64_t releaseOneCache() noexcept {
    if (vision && vision.use_count() == 1 && visionIdle()) {
      const uint64_t bytes = vision->arenaBytes();
      vision.reset();
      return bytes;
    }
    for (auto entry = embeddingCache.end(); entry != embeddingCache.begin();) {
      if ((--entry)->use_count() == 1)
        return uncache(entry);
    }
    return 0;
  }

  // Puts back the encoder a start replaced, or drops the one it built,
  // unless the start completes: an admission granted it, but a later step of
  // the start threw.
  struct VisionRollback final {
    Impl &runtime;
    std::shared_ptr<ops::Vision> previous;
    bool committed = false;
    ~VisionRollback() {
      if (!committed)
        runtime.vision = std::move(previous);
    }
  };

  // What a refused start matched (StateAdmission::held): the rows it would
  // share and, when an image still needs its encode and the live encoder
  // covers it, that encoder.
  struct Matched final {
    std::vector<std::shared_ptr<ImageRows>> rows;
    std::shared_ptr<ops::Vision> encoder;
  };

  // A request's lane with everything else its start allocates, in one
  // admission: the pixel and embedding buffers of the images nothing holds
  // yet and, when an image still needs an encode that the live encoder does
  // not cover, a vision scratch sized for the largest such image. That
  // encoder replaces the live one, which covers fewer patches, so every
  // image waiting on the old one fits the new; a command in flight keeps the
  // old arena until it completes. Rows something holds are shared, encoded
  // or not. Images the restored prefix covers are left out: only their
  // spans are kept. At the budget the engine retries a denied start after
  // each reclaim step, and a denial builds nothing, so no encoder arena,
  // image buffer or lane state is built and dropped every time. The refusal
  // keeps its cause and holds what it matched, so the reclaim before the
  // retry spares it; a grant hands the request's images to `images` and
  // counts the rows it shares as reuses, each once.
  StateAdmission activate(const ModelRequest &request, uint32_t stateLane,
                          std::vector<ImageState> &images) {
    if (request.images.empty())
      return laneAdmission(stateLane, states.tryActivateLane(stateLane, request.id));
    // The engine rejects image requests at submission when there is no vision.
    if (!model.descriptor.hasVision())
      throw std::logic_error("image request reached a model without vision");
    std::vector<ImageState> staged;
    staged.reserve(request.images.size());
    std::vector<std::shared_ptr<ImageRows>> shared;
    uint64_t bytes = 0;
    // The patches of the largest staged image that still needs its encode.
    uint32_t encodePatches = 0;
    for (const ImageSpan &span : request.images) {
      if (span.end() <= request.restoredTokens) {
        staged.push_back({span, nullptr});
        continue;
      }
      // New rows enter the registry now, so a repeated placement shares
      // them; they have no buffers until the admission allocates them.
      std::shared_ptr<ImageRows> rows = findRows(span);
      if (!rows) {
        rows = std::make_shared<ImageRows>();
        rows->key = imageKey(span);
        imageRows.insert_or_assign(rows->key, rows);
        bytes += span.pixelBytes() + embeddingBytes(span);
      } else if (rows->embeddings && std::ranges::find(shared, rows) == shared.end()) {
        shared.push_back(rows);
      }
      if (!rows->encoded)
        encodePatches = std::max(encodePatches, span.gridHeight * span.gridWidth);
      staged.push_back({span, std::move(rows)});
    }
    const uint64_t encoderBytes =
        encodePatches && !(vision && vision->maximumPatches() >= encodePatches)
            ? ops::Vision::scratchBytes(model.vision.tensors.layout, encodePatches)
            : 0;
    std::shared_ptr<ops::Vision> encoder;
    const uint8_t *pixels = request.imagePixels.data();
    const auto allocate = [&] {
      if (encoderBytes) {
        encoder = std::make_shared<ops::Vision>(
            backend, model.vision.tensors, encodePatches);
      }
      for (ImageState &image : staged) {
        const ImageSpan &span = image.span;
        if (image.rows && !image.rows->embeddings) {
          ImageRows &rows = *image.rows;
          rows.pixels = backend.allocateBuffer(
              span.pixelBytes(), BufferStorage::Shared, "image pixels");
          std::memcpy(contents<uint8_t>(rows.pixels, "image pixels"), pixels,
                      static_cast<size_t>(span.pixelBytes()));
          rows.embeddings = backend.allocateBuffer(
              embeddingBytes(span), BufferStorage::Private, "image embeddings");
        }
        pixels += span.pixelBytes();
      }
    };
    StateAdmission admission = laneAdmission(
        stateLane, states.tryActivateLane(stateLane, request.id, encoderBytes + bytes, allocate));
    if (!admission.granted()) {
      admission.held = std::make_shared<const Matched>(
          Matched{std::move(shared), encodePatches && !encoderBytes ? vision : nullptr});
      return admission;
    }
    if (encoder)
      vision = std::move(encoder);
    counters.imageEmbeddingReuses += shared.size();
    images = std::move(staged);
    return admission;
  }

  // No image waits for its encode, so the vision arena can go.
  [[nodiscard]] bool visionIdle() noexcept {
    for (auto entry = imageRows.begin(); entry != imageRows.end();) {
      const std::shared_ptr<ImageRows> rows = entry->second.lock();
      if (!rows) {
        entry = imageRows.erase(entry);
        continue;
      }
      if (!rows->encoded)
        return false;
      ++entry;
    }
    return true;
  }

  // Encodes every image whose rows first appear in this chunk and overwrites
  // the chunk's placeholder embedding rows with the image rows. Text-only
  // requests add no dispatches.
  void addImageRows(CommandGraph &graph, Request &entry,
                    const ModelBatchItem &item, uint32_t rowBegin) {
    const uint64_t chunkBegin = item.logicalPosition;
    const uint64_t chunkEnd = chunkBegin + item.tokenCount;
    for (ImageState &image : entry.images) {
      const uint64_t begin = std::max<uint64_t>(chunkBegin, image.span.offset);
      const uint64_t end = std::min<uint64_t>(chunkEnd, image.span.end());
      if (begin >= end)
        continue;
      if (!image.rows)
        throw std::logic_error("prefill reached an image its activation left out");
      ImageRows &rows = *image.rows;
      if (!rows.encoded && !rows.encoding) {
        if (!vision)
          throw std::logic_error("image request has no vision encoder");
        vision->encode(graph, image.span.grid(), rows.pixels, rows.embeddings);
        rows.encoding = true;
        ++counters.imageEncodes;
      }
      const uint32_t width = model.vision.tensors.layout.outputHiddenSize;
      ops::RowCopy::add(
          graph, rows.embeddings,
          {static_cast<uint32_t>(begin - image.span.offset), width, 0},
          prefillArena->get(PrefillTensor::Hidden0),
          {rowBegin + static_cast<uint32_t>(begin - chunkBegin), width, 0},
          static_cast<uint32_t>(end - begin), width);
    }
  }

  static float uniformAt(uint64_t seed, uint64_t counter) noexcept {
    uint64_t value = seed + counter * 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return float(value >> 40) * 0x1p-24F;
  }

  static float nextUniform(Request &entry) noexcept {
    return uniformAt(entry.sampling.seed, ++entry.rngCounter);
  }

  // SPLASH_DRAFT_AHEAD: the uniforms stageSamplingCycle will stage next,
  // without drawing them.
  static std::array<float, kSamplingUniformCount>
  nextCycleUniforms(const Request &entry) noexcept {
    std::array<float, kSamplingUniformCount> result{};
    for (uint32_t index = SPLASH_UNIFORM_PROPOSALS;
         index < SPLASH_SAMPLING_UNIFORMS; ++index)
      result[index] = uniformAt(entry.sampling.seed,
                                entry.rngCounter + index - SPLASH_UNIFORM_PROPOSALS + 1);
    return result;
  }

  static void stageSamplingCycle(Request &entry) noexcept {
    entry.cycleUniforms.fill(0.0F);
    for (uint32_t index = SPLASH_UNIFORM_PROPOSALS;
         index < SPLASH_SAMPLING_UNIFORMS; ++index) {
      entry.cycleUniforms[index] = nextUniform(entry);
    }
  }

  [[nodiscard]] MetalBuffer synchronizedPageTable(Request &entry,
                                                  const ModelBatchItem &item) {
    if (entry.stateLane >= pageTableBindings.size())
      throw std::out_of_range("request state lane is outside page tables");
    if (item.pageTable.empty() ||
        item.pageTable.size() > kMaximumPageTableEntries) {
      throw std::invalid_argument("request page table has invalid length");
    }
    if (!item.pageTableRevision)
      throw std::invalid_argument("request page table has no revision");
    PageTableBinding &binding = pageTableBindings[entry.stateLane];
    MetalBuffer destination =
        decodeArena->get(entry.stateLane, DecodeTensor::PageTable);
    // Rewrite only what changed since the table was written: nothing at the
    // same revision, the entries from the first changed page on at the next
    // one, and everything after two changes or for another request.
    const auto size = static_cast<uint32_t>(item.pageTable.size());
    uint32_t first = 0;
    if (binding.requestId == entry.id) {
      if (binding.revision == item.pageTableRevision)
        first = size;
      else if (binding.revision + 1 == item.pageTableRevision)
        first = std::min(item.pageTableFirstChanged, size);
    }
    if (first < size)
      kvPages.writeEntries(item.pageTable, first, destination);
    binding = {entry.id, item.pageTableRevision};
    return destination;
  }

  void addRopeTables(CommandGraph &graph, MetalBuffer targetPositions,
                     uint32_t targetRows, MetalBuffer draftPositions,
                     uint32_t draftRows, MetalBuffer targetCos,
                     MetalBuffer targetSin, MetalBuffer draftCos,
                     MetalBuffer draftSin) const {
    ops::RoPE::addTables(
        graph, std::move(targetPositions), std::move(draftPositions),
        prefillArena->get(PrefillTensor::TargetInverseFrequencies),
        prefillArena->get(PrefillTensor::DraftInverseFrequencies),
        std::move(targetCos), std::move(targetSin), std::move(draftCos),
        std::move(draftSin), {targetRows, draftRows}, kPrefillRows);
  }

  // A constrained lane keeps its final prompt row, which prefill leaves at
  // row 0 of its Hidden0 block, until its first mask arrives.
  void captureFinalHidden(Request &entry, uint32_t lane) const {
    const uint16_t *source =
        contents<uint16_t>(decodeArena->get(lane, DecodeTensor::Hidden0),
                           "target final hidden source");
    entry.finalTargetHidden.assign(source,
                                   source + geometry.target.hiddenSize);
  }

  static DispatchDraftCapturePlan
  activeDraftCaptures(const Request &entry, const ModelBatchItem &item) {
    if (!entry.draftContextPlan) {
      throw std::logic_error("prefill request has no draft context plan");
    }
    const uint64_t next = item.logicalPosition + item.tokenCount;
    return draftCaptureSpansForDispatch(
        *entry.draftContextPlan, static_cast<uint32_t>(item.logicalPosition),
        static_cast<uint32_t>(next));
  }

  static uint32_t captureRows(const DispatchDraftCapturePlan &captures) {
    uint32_t rows = 0;
    for (const auto &capture : captures)
      rows += capture.absoluteEnd - capture.absoluteBegin;
    return rows;
  }

  // The lengths after the draft ring takes rows [begin, end) at target
  // length targetTokens. Unless `reset` starts a new window there, the rows
  // continue the ring, which must hold rows ending at `begin`.
  static QwenLogicalLengths
  advanceDraftContext(const QwenLogicalLengths &previous, uint64_t targetTokens,
                      uint64_t begin, uint64_t end, bool reset) {
    if (!reset && (!previous.draftLength || previous.draftEnd() != begin))
      throw std::logic_error("draft capture does not continue the draft ring");
    const uint64_t combined = (reset ? 0 : previous.draftLength) + (end - begin);
    QwenLogicalLengths next = previous;
    next.targetTokens = targetTokens;
    next.draftLength =
        static_cast<uint32_t>(std::min<uint64_t>(combined, kDraftCacheStride));
    next.draftBase = end - next.draftLength;
    return next;
  }

  // Only a sampled lane's draws read its cycle's uniforms.
  void uploadSamplingUniforms(const Request &entry, uint32_t lane) const {
    std::copy(entry.cycleUniforms.begin(), entry.cycleUniforms.end(),
              contents<float>(
                  decodeArena->get(lane, DecodeTensor::SamplingUniforms),
                  "sampling uniforms"));
  }

  // Only a constrained lane's selections read its mask rows.
  std::span<uint32_t> constraintMasks(uint32_t lane) const {
    const MetalBuffer masks =
        decodeArena->get(lane, DecodeTensor::ConstraintMasks);
    return {contents<uint32_t>(masks, "constraint masks"),
            masks.sizeBytes() / sizeof(uint32_t)};
  }

  void uploadConstraintMasks(uint32_t lane,
                             std::span<const uint32_t> masks) const {
    const std::span<uint32_t> rows = constraintMasks(lane);
    if (masks.size() > rows.size())
      throw std::invalid_argument("constraint mask exceeds decode arena");
    std::ranges::copy(masks, rows.begin());
  }

  // A constrained lane whose mask was abandoned admits every token.
  void admitEveryToken(uint32_t lane) const {
    std::ranges::fill(constraintMasks(lane),
                      std::numeric_limits<uint32_t>::max());
  }

  static ops::SamplingPenalties samplingPenalties(const Request &entry) noexcept {
    return {entry.sampling.repetitionPenalty, entry.sampling.presencePenalty,
            entry.sampling.frequencyPenalty};
  }

  static ops::SamplingPolicy samplingPolicy(const Request &entry) noexcept {
    return {entry.sampling.topK, entry.sampling.temperature,
            entry.sampling.topP, entry.constraint == ConstraintMode::TokenMask,
            (entry.flags & RequestIgnoreEndOfSequence) != 0,
            samplingPenalties(entry), entry.sampling.minP};
  }

  ops::SamplingBuffers samplingBuffers(uint32_t lanes) const {
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, lanes);
    };
    return {d(DecodeTensor::Logits),
            d(DecodeTensor::TargetPartialMasses),
            d(DecodeTensor::TargetVocabularyRows),
            d(DecodeTensor::SamplingUniforms),
            d(DecodeTensor::ConstraintMasks),
            d(DecodeTensor::OutputTokens),
            d(DecodeTensor::ArgmaxValues),
            d(DecodeTensor::ArgmaxIndices),
            d(DecodeTensor::InputTokens),
            d(DecodeTensor::Candidates),
            d(DecodeTensor::ProposalProbs),
            d(DecodeTensor::TargetVocabularyRanges),
            d(DecodeTensor::TargetVocabularyArrivals),
            d(DecodeTensor::TargetCandidateRows)};
  }

  ops::AcceptanceBuffers acceptanceBuffers(uint32_t lanes) const {
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, lanes);
    };
    return {d(DecodeTensor::ProposedTokens),
            d(DecodeTensor::Candidates),
            d(DecodeTensor::ProposalProbs),
            d(DecodeTensor::TargetVocabularyRows),
            d(DecodeTensor::SamplingUniforms),
            d(DecodeTensor::OutputTokens),
            d(DecodeTensor::RetainedCount),
            d(DecodeTensor::AcceptedCount),
            samplingBuffers(lanes),
            d(DecodeTensor::BlockCorrection)};
  }

  std::span<uint32_t> penaltyWords(uint32_t stateLane) const {
    return {contents<uint32_t>(
                decodeArena->get(stateLane, DecodeTensor::PenaltyState),
                "penalty words"),
            geometry.target.vocabularySize};
  }

  // Rebuilds a penalized request's penalty words when it takes a state lane,
  // at activation and at resume, from the history the lane's prefill
  // consumes. No command reads the lane's words yet.
  void bindPenalties(const Request &entry,
                     std::span<const uint32_t> history) const {
    const ops::SamplingPenalties penalties = samplingPenalties(entry);
    if (!penalties.active())
      return;
    ops::Sampling::rebuildPenaltyWords(penaltyWords(entry.stateLane), history,
                                       entry.generatedTokens,
                                       entry.pendingToken,
                                       penalties.repetition != 1.0F);
  }

  // The one place a token the target selected becomes the pending anchor:
  // tokens are one step's selections in order, the new anchor last. The
  // command that selected them has completed, and the next one that reads
  // the lane's words is encoded after this.
  void commitSelected(Request &entry, std::span<const uint32_t> tokens) {
    if (tokens.empty())
      throw std::logic_error("no selected token to commit");
    if (samplingPenalties(entry).active())
      ops::Sampling::countPenaltyTokens(penaltyWords(entry.stateLane), tokens);
    entry.pendingToken = tokens.back();
  }

  // A selection outside the vocabulary is the sampling kernels' sentinel for a
  // non-finite logit row: a numerical outcome of this request, which it reports
  // as its lane failure (ModelStepResult::failure) so the batch survives.
  [[nodiscard]] std::string invalidSelection(std::span<const uint32_t> tokens) const {
    const auto found = std::find_if(tokens.begin(), tokens.end(), [&](uint32_t token) {
      return token >= geometry.target.vocabularySize;
    });
    if (found == tokens.end())
      return {};
    return "target selected out-of-vocabulary token " + std::to_string(*found) +
           " from a non-finite logit row";
  }

  // A lane of an initial selection, whose final prompt row is at row 0 of
  // its Hidden0 block: one that selects its first token, or a score lane,
  // which needs only the logits.
  struct InitialSelection final {
    Request *entry = nullptr;
    uint32_t lane = 0;
    bool select = false;
  };

  // A sampled lane draws its first token with the first uniform of a fresh
  // cycle.
  void uploadInitialUniform(Request &entry, uint32_t lane) const {
    entry.cycleUniforms.fill(0.0F);
    entry.cycleUniforms[SPLASH_UNIFORM_INITIAL] = nextUniform(entry);
    uploadSamplingUniforms(entry, lane);
  }

  // One LM head over the batch's `width` lanes, then one selection of the
  // first token of every selecting lane from its logits row 0, which
  // initialToken() reads. The head computes, and nothing reads, the other
  // rows of each lane and the lanes not listed; a lane that does not select
  // takes the argmax of its row. Sampled lanes' uniforms and constrained
  // lanes' masks are uploaded first.
  void encodeInitialSelections(CommandGraph &graph,
                               std::span<const InitialSelection> lanes,
                               uint32_t width) {
    const uint32_t storage = targetModel.decodeStorageLanes(width);
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, storage);
    };
    targetModel.addHeadBatch(graph, d(DecodeTensor::Hidden0),
                             d(DecodeTensor::FinalHidden),
                             d(DecodeTensor::Logits), width,
                             decodeArena->linearScratch());
    if (std::ranges::none_of(lanes, &InitialSelection::select))
      return;
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    std::array<uint32_t, kLaneCount> stateLanes{};
    for (const InitialSelection &lane : lanes) {
      if (!lane.select)
        continue;
      policies[lane.lane] = samplingPolicy(*lane.entry);
      stateLanes[lane.lane] = lane.entry->stateLane;
    }
    sampling.addInitial(graph, std::span(policies).first(width),
                        samplingBuffers(width), 0,
                        geometry.target.stopTokens[0],
                        geometry.target.stopTokens[1],
                        {penaltyTable, std::span(stateLanes).first(width)});
  }

  // The first token encodeInitialSelections() selected for a batch lane: a
  // selection writes one output token per lane, in lane order.
  uint32_t initialToken(uint32_t lane) const {
    return contents<uint32_t>(
        decodeArena->batchSlice(DecodeTensor::OutputTokens, lane + 1),
        "initial tokens")[lane];
  }

  // Selects the first token of each constrained lane of the plan under the
  // mask it was given, from its final prompt row, in one command. The lanes
  // draft and verify from their next plan on.
  std::unique_ptr<ModelBatchTicket>
  submitInitialSelection(std::span<const ModelBatchItem> items,
                         std::function<void()> completion) {
    const uint32_t width = static_cast<uint32_t>(items.size());
    std::array<InitialSelection, kLaneCount> selections{};
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = request(items[lane].requestId);
      if (entry.decodeStage != DecodeStage::ApplyInitialMask ||
          entry.pendingToken ||
          entry.maskWords.size() != geometry.maskWords() ||
          entry.finalTargetHidden.size() != geometry.target.hiddenSize) {
        throw std::logic_error("initial selection state is invalid");
      }
      std::ranges::copy(entry.finalTargetHidden,
                        contents<uint16_t>(
                            decodeArena->get(lane, DecodeTensor::Hidden0),
                            "final prompt hidden"));
      if (samplingEnabled(entry))
        uploadInitialUniform(entry, lane);
      uploadConstraintMasks(lane, entry.maskWords);
      selections[lane] = {&entry, lane, true};
    }
    CommandGraph graph;
    encodeInitialSelections(graph, std::span(selections).first(width), width);
    CommandTicket command =
        backend.submitCommandAsync(graph.dispatches(), std::move(completion));
    auto finish = [this, selections, width](CommandTiming) {
      std::vector<ModelStepResult> results;
      results.reserve(width);
      for (const InitialSelection &selection :
           std::span(selections).first(width)) {
        Request &entry = *selection.entry;
        ModelStepResult &result = results.emplace_back();
        result.requestId = entry.id;
        const uint32_t token = initialToken(selection.lane);
        // A failed selection leaves no anchor: the engine ends the request
        // before any output, and the other lanes go on.
        result.failure = invalidSelection({&token, 1});
        if (!result.failure.empty())
          continue;
        commitSelected(entry, {&token, 1});
        entry.maskWords.clear();
        entry.finalTargetHidden.clear();
        entry.decodeStage = DecodeStage::Regular;
        emitTerminalAnchor(entry, result);
      }
      return results;
    };
    return std::make_unique<DeferredMetalTicket>(std::move(command),
                                                 std::move(finish));
  }

  ChunkedPrefillParams chunkParams(uint64_t logicalPosition,
                                   uint32_t chunkTokens, uint32_t chunkStride,
                                   std::span<const uint32_t> pages) const {
    return ops::PagedAttention::prefillParams(
        logicalPosition, chunkTokens, chunkStride,
        static_cast<uint32_t>(pages.size()));
  }

  struct RaggedPrefillSequence final {
    Request *entry = nullptr;
    const ModelBatchItem *item = nullptr;
    uint32_t lane = 0;
    uint32_t rowBegin = 0;
    uint32_t attentionStride = 0;
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    uint32_t captureBegin = 0;
    ChunkedPrefillParams chunk;
    MetalBuffer pageTable;
    DispatchDraftCapturePlan captures;
  };

  struct RaggedPrefillBatch final {
    std::vector<RaggedPrefillSequence> sequences;
    uint32_t rows = 0;
    uint32_t capturedRows = 0;
  };

  MetalBuffer prefillU16(const MetalBuffer &tensor, uint32_t begin,
                         uint32_t rows, uint32_t width) const {
    return backend.view(tensor, bytesFor<uint16_t>(uint64_t{begin} * width),
                        bytesFor<uint16_t>(uint64_t{rows} * width));
  }

  RaggedPrefillBatch
  prepareRaggedPrefill(std::span<const ModelBatchItem> items,
                       std::array<Request *, kLaneCount> &entries) {
    RaggedPrefillBatch batch;
    batch.sequences.reserve(items.size());
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      const ModelBatchItem &item = items[lane];
      Request &entry = request(item.requestId);
      if (item.tokenCount > kPrefillRows ||
          item.logicalPosition > entry.promptTokens ||
          item.tokenCount > entry.promptTokens - item.logicalPosition ||
          !entry.resident) {
        throw std::invalid_argument("invalid ragged Qwen prefill item");
      }
      const QwenLaneMetadata &metadata = states.metadata(entry.stateLane);
      if (metadata.requestId != entry.id ||
          metadata.lengths.targetTokens != item.logicalPosition) {
        throw std::logic_error("ragged prefill state length is not exact");
      }
      if (item.logicalPosition == 0)
        states.clearForColdStart(entry.stateLane);
      if (item.tokenCount > kPrefillRows - batch.rows) {
        throw std::invalid_argument("ragged prefill exceeds actual-row budget");
      }
      auto captures = activeDraftCaptures(entry, item);
      const uint32_t capturedRows = captureRows(captures);
      const uint32_t attentionStride =
          ((item.tokenCount + kTileRows - 1) / kTileRows) * kTileRows;
      const ChunkedPrefillParams chunk =
          chunkParams(item.logicalPosition, item.tokenCount, attentionStride,
                      item.pageTable);
      MetalBuffer pageTable = synchronizedPageTable(entry, item);
      batch.sequences.push_back({&entry, &item, lane, batch.rows,
                                 attentionStride, queryOffset, kvOffset,
                                 batch.capturedRows, chunk, std::move(pageTable),
                                 std::move(captures)});
      entries[lane] = &entry;
      batch.rows += item.tokenCount;
      batch.capturedRows += capturedRows;
      queryOffset += bytesFor<uint16_t>(
          uint64_t{geometry.target.attentionQueryHeads} * attentionStride *
          geometry.target.attentionHeadDimension);
      kvOffset += bytesFor<uint16_t>(
          uint64_t{geometry.target.attentionKvHeads} * attentionStride *
          geometry.target.attentionHeadDimension);
    }
    if (!batch.rows ||
        queryOffset >
            prefillArena->get(PrefillTensor::FullQueries).sizeBytes() ||
        kvOffset > prefillArena->get(PrefillTensor::ChunkKeys).sizeBytes()) {
      throw std::logic_error("ragged prefill scratch geometry overflowed");
    }

    auto *input =
        contents<uint32_t>(prefillArena->get(PrefillTensor::InputTokens),
                           "ragged prefill input tokens");
    auto *targetPositions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::TargetPositions),
                           "target RoPE positions");
    auto *draftPositions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::DraftPositions),
                           "draft RoPE positions");
    for (const RaggedPrefillSequence &sequence : batch.sequences) {
      const ModelBatchItem &item = *sequence.item;
      std::copy(item.inputTokens.begin(), item.inputTokens.end(),
                input + sequence.rowBegin);
      for (uint32_t localRow = 0; localRow < item.tokenCount; ++localRow) {
        const uint32_t row = sequence.rowBegin + localRow;
        if (input[row] >= geometry.target.vocabularySize) {
          throw std::invalid_argument("prompt token is out of vocabulary");
        }
        const std::array<uint32_t, 3> rotary =
            ropePosition(*sequence.entry, item.logicalPosition + localRow);
        std::copy(rotary.begin(), rotary.end(), targetPositions + row * 3);
      }
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        for (uint32_t row = capture.absoluteBegin; row < capture.absoluteEnd;
             ++row) {
          const uint32_t compactRow = sequence.captureBegin +
                                      capture.compactDestinationRow + row -
                                      capture.absoluteBegin;
          draftPositions[compactRow] = row;
        }
      }
    }
    return batch;
  }

  void addRaggedDraftContext(CommandGraph &graph,
                             const RaggedPrefillBatch &batch) {
    if (!batch.capturedRows)
      return;
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor); };
    std::array<DFlashPrefillSpan, kLaneCount * 2> spans{};
    uint32_t spanCount = 0;
    for (const RaggedPrefillSequence &sequence : batch.sequences) {
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        DFlashPrefillSpan &span = spans.at(spanCount++);
        span.compactRow = sequence.captureBegin + capture.compactDestinationRow;
        span.rows = capture.absoluteEnd - capture.absoluteBegin;
        span.startPosition = capture.absoluteBegin;
        span.ring = states.draft(sequence.entry->stateLane);
      }
    }
    draftModel.addContextPrefill(
        graph,
        {p(PrefillTensor::Captured), p(PrefillTensor::ProjectionSums),
         p(PrefillTensor::ContextProjected), p(PrefillTensor::ContextHidden),
         p(PrefillTensor::ContextKv), p(PrefillTensor::DraftRopeCos),
         p(PrefillTensor::DraftRopeSin)},
        batch.capturedRows, std::span(spans).first(spanCount));
  }

  // Returns each lane's draft captures, indexed like `entries`.
  std::array<DispatchDraftCapturePlan, kLaneCount>
  encodeRaggedPrefillGraph(CommandGraph &graph,
                           std::span<const ModelBatchItem> items,
                           std::array<Request *, kLaneCount> &entries) {
    RaggedPrefillBatch batch = prepareRaggedPrefill(items, entries);
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor); };

    addRopeTables(graph, p(PrefillTensor::TargetPositions), batch.rows,
                  p(PrefillTensor::DraftPositions), batch.capturedRows,
                  p(PrefillTensor::RopeCos), p(PrefillTensor::RopeSin),
                  p(PrefillTensor::DraftRopeCos),
                  p(PrefillTensor::DraftRopeSin));

    targetModel.addEmbedding(graph, p(PrefillTensor::InputTokens),
                             p(PrefillTensor::Hidden0), batch.rows);
    for (const RaggedPrefillSequence &sequence : batch.sequences) {
      addImageRows(graph, *sequence.entry, *sequence.item, sequence.rowBegin);
    }

    std::array<QwenTargetPrefillSequence, kLaneCount> modelSequences{};
    const uint32_t modelSequenceCount =
        static_cast<uint32_t>(batch.sequences.size());
    const uint64_t stateBindingCount = uint64_t{modelSequenceCount} *
                                       geometry.target.stateLayout.layers;
    std::vector<MetalBuffer> convolutionIn(stateBindingCount);
    std::vector<MetalBuffer> convolutionOut(stateBindingCount);
    std::vector<MetalBuffer> recurrentIn(stateBindingCount);
    std::vector<MetalBuffer> recurrentOut(stateBindingCount);
    for (uint32_t lane = 0; lane < batch.sequences.size(); ++lane) {
      const RaggedPrefillSequence &sequence = batch.sequences[lane];
      QwenTargetPrefillSequence &destination = modelSequences[lane];
      destination.rowBegin = sequence.rowBegin;
      destination.rows = sequence.item->tokenCount;
      destination.attentionStride = sequence.attentionStride;
      destination.queryOffset = sequence.queryOffset;
      destination.kvOffset = sequence.kvOffset;
      destination.chunk = sequence.chunk;
      destination.pageTable = sequence.pageTable;
      const uint32_t gdnLayers = geometry.target.stateLayout.layers;
      const uint64_t stateBegin = uint64_t{lane} * gdnLayers;
      destination.convolutionIn =
          std::span(convolutionIn).subspan(stateBegin, gdnLayers);
      destination.convolutionOut =
          std::span(convolutionOut).subspan(stateBegin, gdnLayers);
      destination.recurrentIn =
          std::span(recurrentIn).subspan(stateBegin, gdnLayers);
      destination.recurrentOut =
          std::span(recurrentOut).subspan(stateBegin, gdnLayers);
      const GdnParityBuffers &in = states.current(sequence.entry->stateLane);
      const GdnParityBuffers &out = states.next(sequence.entry->stateLane);
      for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
        convolutionIn[stateBegin + layer] = in.convolutionLayers[layer];
        convolutionOut[stateBegin + layer] = out.convolutionLayers[layer];
        recurrentIn[stateBegin + layer] = in.recurrentLayers[layer];
        recurrentOut[stateBegin + layer] = out.recurrentLayers[layer];
      }
      destination.captureCount = sequence.captures.size();
      for (uint32_t index = 0; index < sequence.captures.size(); ++index) {
        const DispatchDraftCaptureSpan &capture = sequence.captures[index];
        destination.captures[index] = {
            sequence.rowBegin +
                static_cast<uint32_t>(capture.absoluteBegin -
                                      sequence.item->logicalPosition),
            sequence.captureBegin + capture.compactDestinationRow,
            capture.absoluteEnd - capture.absoluteBegin};
      }
    }
    QwenTargetPrefillBuffers buffers = prefillBuffers(*prefillArena);
    const MetalBuffer finalHidden = targetModel.addPrefill(
        graph, std::move(buffers),
        std::span(modelSequences).first(batch.sequences.size()), batch.rows,
        kvPages.layers(), aneFfn);
    addRaggedDraftContext(graph, batch);

    // A lane that finishes its prompt copies the prompt's last row to row 0
    // of its Hidden0 block. A constrained lane's completion captures that
    // row into finalTargetHidden (captureFinalHidden), which holds it until
    // the first mask. The others share one head: a score lane reads raw
    // logits at the final prompt position, and a policy lane selects its
    // first token.
    std::array<InitialSelection, kLaneCount> selections{};
    uint32_t selectionCount = 0;
    for (const RaggedPrefillSequence &sequence : batch.sequences) {
      Request &entry = *sequence.entry;
      const ModelBatchItem &item = *sequence.item;
      if (entry.replayingGeneration ||
          item.logicalPosition + item.tokenCount != entry.promptTokens)
        continue;
      const uint32_t hidden = geometry.target.hiddenSize;
      ops::RowCopy::add(
          graph,
          prefillU16(finalHidden, sequence.rowBegin, item.tokenCount, hidden),
          {item.tokenCount - 1, hidden, 0},
          decodeArena->get(sequence.lane, DecodeTensor::Hidden0),
          {0, hidden, 0}, 1, hidden);
      if (entry.constraint != ConstraintMode::None)
        continue;
      const bool scoring = !entry.scoreTokens.empty();
      if (!scoring && samplingEnabled(entry))
        uploadInitialUniform(entry, sequence.lane);
      selections[selectionCount++] = {&entry, sequence.lane, !scoring};
    }
    if (selectionCount) {
      encodeInitialSelections(
          graph, std::span(selections).first(selectionCount),
          static_cast<uint32_t>(batch.sequences.size()));
    }
    std::array<DispatchDraftCapturePlan, kLaneCount> captures{};
    for (const RaggedPrefillSequence &sequence : batch.sequences)
      captures[sequence.lane] = sequence.captures;
    return captures;
  }

  // draftInputs false: an adopted SPLASH_DRAFT_AHEAD block already wrote the
  // draft-side inputs on the GPU (the same values).
  void prepareDecodeLane(Request &entry, const ModelBatchItem &item,
                         uint32_t lane, bool draftInputs = true) {
    if (!entry.resident || !entry.promptComplete || !entry.pendingToken) {
      throw std::logic_error("decode request is not ready");
    }
    entry.verifyRows = kDecodeRows;
    const QwenLaneMetadata &metadata = states.metadata(entry.stateLane);
    if (metadata.lengths.targetTokens != item.logicalPosition ||
        !metadata.lengths.hasCompleteDraftWindow(kDraftCacheStride)) {
      throw std::logic_error("decode state length is not exact");
    }
    static_cast<void>(synchronizedPageTable(entry, item));
    if (draftInputs) {
      auto *draftInput = contents<uint32_t>(
          decodeArena->get(lane, DecodeTensor::DraftInputTokens),
          "draft input tokens");
      draftInput[0] = *entry.pendingToken;
      std::fill(draftInput + 1, draftInput + kDecodeRows,
                geometry.target.maskToken);
    }

    auto *positions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::Positions),
                           "decode RoPE positions");
    auto *draftPositions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::DraftPositions),
                           "decode draft RoPE positions");
    for (uint32_t row = 0; row < kDecodeRows; ++row) {
      const std::array<uint32_t, 3> rotary =
          ropePosition(entry, item.logicalPosition + row);
      std::copy(rotary.begin(), rotary.end(), positions + row * 3);
      // The draft is a text model over logical positions.
      if (draftInputs)
        draftPositions[row] = static_cast<uint32_t>(item.logicalPosition + row);
    }
  }

  // A prompt-lookup proposal for the request's next cycle: the history ends
  // with the pending anchor.
  std::optional<std::array<uint32_t, kDraftProposalTokens>>
  promptLookupProposal(Request &entry) const {
    if (entry.lookupHistory.empty())
      return std::nullopt;
    entry.lookupHistory.push_back(*entry.pendingToken);
    const auto proposal = model::promptLookup<kDraftProposalTokens>(
        entry.lookupHistory, adaptiveLookup ? entry.lookupMatch : lookupMinMatch);
    entry.lookupHistory.pop_back();
    return proposal;
  }

  // Writes a lookup's proposals where the draft's selector would, as lane
  // 0's: each position's candidates hold the proposed token alone with
  // probability 1, so a sampled verify row draws its correction from the
  // target distribution without that token and acceptance keeps it with the
  // target's probability (sampling.metal vocabulary_draw, decode_accept_dflash).
  void preparePromptLookup(const std::array<uint32_t, kDraftProposalTokens> &proposal) const {
    constexpr uint32_t candidates = SPLASH_DRAFT_CANDIDATES;
    std::ranges::copy(proposal, contents<uint32_t>(
        decodeArena->get(0, DecodeTensor::ProposedTokens), "lookup proposals"));
    auto *ids = contents<uint32_t>(decodeArena->get(0, DecodeTensor::Candidates), "lookup candidates");
    auto *probabilities = contents<float>(
        decodeArena->get(0, DecodeTensor::ProposalProbs), "lookup probabilities");
    std::fill_n(ids, kDraftProposalTokens * candidates, std::numeric_limits<uint32_t>::max());
    std::fill_n(probabilities, kDraftProposalTokens * candidates, 0.0F);
    for (uint32_t row = 0; row < kDraftProposalTokens; ++row) {
      ids[row * candidates] = proposal[row];
      probabilities[row * candidates] = 1.0F;
    }
  }

  struct WideProposal final {
    std::array<uint32_t, kWideLookupProposals> tokens{};  // the first tiles * 8 - 1 are used
    uint32_t tiles = 2;
  };

  // SPLASH_WIDE_PROMPT_LOOKUP: a 16-row lookup proposal (32 rows, with
  // SPLASH_WIDE_LOOKUP32, right after a fully accepted wide lookup) for a
  // dense target whose wide rows stay row-exact. A penalized request is
  // refused: its penalty words would not count the earlier rows' proposals.
  std::optional<WideProposal> lookup16Proposal(Request &entry, const ModelBatchItem &item) const {
    // ponytail: only already-admitted pages; fall back to fewer rows when an
    // extra page would be needed instead of changing engine admission.
    const auto fits = [&](uint32_t rows) {
      return entry.maxNewTokens - entry.generatedTokens >= rows &&
             item.logicalPosition + rows <= kv::kMaximumPhysicalTokens &&
             (item.logicalPosition + rows + kTileRows - 1) / kTileRows <= item.pageTable.size();
    };
    if (stableVerifyRows < 2 * kDecodeRows || entry.lookupHistory.empty() ||
        samplingPenalties(entry).active() || !fits(2 * kDecodeRows))
      return std::nullopt;
    const size_t minMatch = adaptiveLookup ? entry.lookupMatch : wideLookupMinMatch;
    entry.lookupHistory.push_back(*entry.pendingToken);
    std::optional<WideProposal> result;
    // A 32-row verify costs ~16.5 ms more than 16 rows, so it pays only inside
    // long copy runs: code-edit +12.7%, tool-copy +27.4%, chat -0.7% (n.s.) (1.0.0).
    if (entry.wide32 && entry.wideFull && stableVerifyRows >= 4 * kDecodeRows && fits(4 * kDecodeRows)) {
      if (const auto proposal = model::promptLookup<4 * kDecodeRows - 1>(entry.lookupHistory, minMatch))
        result = WideProposal{*proposal, 4};
    }
    if (!result) {
      if (const auto proposal = model::promptLookup<2 * kDecodeRows - 1>(entry.lookupHistory, minMatch)) {
        result.emplace();
        std::ranges::copy(*proposal, result->tokens.begin());
      }
    }
    entry.lookupHistory.pop_back();
    return result;
  }

  // A wide lookup's inputs: the anchor and proposals over the aliased lanes'
  // verify input rows, each tile's positions (prepareDecodeLane wrote tile 0's)
  // and, for a sampled request, the acceptance uniforms after lane 0's and each
  // drafted row's point-mass candidate (Sampling::addLookupVerify).
  void prepareLookup16(Request &entry, const ModelBatchItem &item, const WideProposal &proposal) {
    const uint32_t tiles = proposal.tiles, rows = tiles * kDecodeRows;
    auto *tokens = contents<uint32_t>(decodeArena->batchSlice(DecodeTensor::InputTokens, tiles),
                                      "wide lookup input");
    tokens[0] = *entry.pendingToken;
    std::copy_n(proposal.tokens.begin(), rows - 1, tokens + 1);
    for (uint32_t tile = 1; tile < tiles; ++tile) {
      auto *positions = contents<uint32_t>(decodeArena->get(tile, DecodeTensor::Positions), "wide positions");
      auto *draftPositions =
          contents<uint32_t>(decodeArena->get(tile, DecodeTensor::DraftPositions), "wide draft positions");
      for (uint32_t row = 0; row < kDecodeRows; ++row) {
        const uint64_t position = item.logicalPosition + tile * kDecodeRows + row;
        std::ranges::copy(ropePosition(entry, position), positions + row * 3);
        draftPositions[row] = static_cast<uint32_t>(position);
      }
    }
    entry.verifyRows = rows;
    if (!samplingEnabled(entry))
      return;
    float *uniforms = contents<float>(decodeArena->batchSlice(DecodeTensor::SamplingUniforms, tiles),
                                      "wide acceptance uniforms") + kSamplingUniformCount;
    for (uint32_t row = 0; row + 1 < rows; ++row)
      uniforms[row] = nextUniform(entry);
    constexpr uint32_t candidates = SPLASH_DRAFT_CANDIDATES;
    auto *ids = contents<uint32_t>(decodeArena->get(0, DecodeTensor::LookupCandidates), "wide candidates");
    auto *probabilities =
        contents<float>(decodeArena->get(0, DecodeTensor::LookupProbabilities), "wide candidate probabilities");
    std::fill_n(ids, (rows - 1) * candidates, std::numeric_limits<uint32_t>::max());
    std::fill_n(probabilities, (rows - 1) * candidates, 0.0F);
    for (uint32_t row = 0; row + 1 < rows; ++row) {
      ids[row * candidates] = tokens[row + 1];
      probabilities[row * candidates] = 1.0F;
    }
  }

  // A wide lookup's physical lanes: one request, tile t at position + 8t.
  struct WideLanes final {
    std::array<Request *, kLaneCount> entries{};
    std::array<ModelBatchItem, kLaneCount> items{};
    uint32_t tiles = 0;
    std::span<Request *const> entrySpan() const { return {entries.data(), tiles}; }
    std::span<const ModelBatchItem> itemSpan() const { return {items.data(), tiles}; }
  };

  static WideLanes wideLanes(Request &entry, const ModelBatchItem &item) {
    WideLanes lanes;
    lanes.tiles = entry.verifyRows / kDecodeRows;
    for (uint32_t tile = 0; tile < lanes.tiles; ++tile) {
      lanes.entries[tile] = &entry;
      lanes.items[tile] = item;
      lanes.items[tile].logicalPosition += tile * kDecodeRows;
    }
    return lanes;
  }

  void encodeLookup16Forward(CommandGraph &graph, Request &entry, const ModelBatchItem &item) {
    const WideLanes lanes = wideLanes(entry, item);
    encodeBatchEmbedding(graph, DecodeTensor::InputTokens, DecodeTensor::Hidden0, lanes.tiles);
    encodeTargetVerifyBatchForward(graph, lanes.entrySpan(), lanes.itemSpan(), true);
    if (entry.wideGdn != ops::WideGdn::Chain)
      ++entry.wideGdnCycles;
  }

  // Policy, acceptance and commits of a wide lookup's rows.
  void encodeLookup16Commit(CommandGraph &graph, Request &entry, const ModelBatchItem &item,
                            uint32_t maximumRetained) {
    const WideLanes lanes = wideLanes(entry, item);
    const uint32_t tiles = lanes.tiles, rows = tiles * kDecodeRows;
    const ops::SamplingPolicy policy = samplingPolicy(entry);
    ops::SamplingBuffers buffers = samplingBuffers(tiles);
    buffers.draftCandidates = decodeArena->get(0, DecodeTensor::LookupCandidates);
    buffers.draftProbabilities = decodeArena->get(0, DecodeTensor::LookupProbabilities);
    sampling.addLookupVerify(graph, policy, buffers, rows, geometry.target.stopTokens[0],
                             geometry.target.stopTokens[1]);
    const MetalBuffer tileRetained = decodeArena->get(0, DecodeTensor::LookupRetainedHalves);
    sampling.addLookupAcceptance(
        graph,
        {decodeArena->batchSlice(DecodeTensor::InputTokens, tiles),
         decodeArena->batchSlice(DecodeTensor::TargetVocabularyRows, tiles),
         decodeArena->batchSlice(DecodeTensor::SamplingUniforms, tiles),
         decodeArena->batchSlice(DecodeTensor::OutputTokens, tiles),
         decodeArena->get(0, DecodeTensor::RetainedCount), decodeArena->get(0, DecodeTensor::AcceptedCount),
         tileRetained},
        rows, maximumRetained, policy, geometry.target.stopTokens[0], geometry.target.stopTokens[1]);
    encodeBatchGdnCommit(graph, lanes.entrySpan(), true);
    encodeDraftStateCommitBatch(graph, lanes.entrySpan(), lanes.itemSpan(), tileRetained);
  }

  // A wide lookup's verify masks: rows + 1 rows from lane 0's, every token for
  // an abandoned mask.
  void uploadWideConstraintMasks(uint32_t tiles, std::span<const uint32_t> masks) const {
    const MetalBuffer buffer = decodeArena->batchSlice(DecodeTensor::ConstraintMasks, tiles);
    const std::span<uint32_t> rows{contents<uint32_t>(buffer, "wide constraint masks"),
                                   buffer.sizeBytes() / sizeof(uint32_t)};
    if (masks.size() > rows.size())
      throw std::invalid_argument("wide constraint mask exceeds decode arena");
    if (masks.empty())
      std::ranges::fill(rows, std::numeric_limits<uint32_t>::max());
    else
      std::ranges::copy(masks, rows.begin());
  }

  // After a decode step: the request's lookup counters, SPLASH_LOOKUP_ADAPTIVE's
  // threshold, and the emitted tokens appended to its history.
  void recordLookup(Request &entry, const DecodeLaneResult &lane,
                    std::span<const uint32_t> emitted) const {
    entry.cyclesSinceLookup = lane.promptLookup
        ? 0 : entry.cyclesSinceLookup + (entry.cyclesSinceLookup < std::numeric_limits<uint32_t>::max());
    if (entry.lookupHistory.empty())
      return;
    ++entry.lookupCycles;
    if (lane.promptLookup) {
      ++entry.lookupHits;
      entry.lookupRetained += lane.retained;
      if (adaptiveLookup)
        entry.lookupMatch = model::adaptLookupMatch(entry.lookupMatch, lane.retained, lookupMinMatch);
    }
    if (lane.wideLookup) {
      ++entry.wideLookupHits;
      entry.wideLookupRetained += lane.retained;
      entry.wide32Hits += lane.wideTiles == 4;
    }
    entry.wideFull = lane.wideLookup && lane.accepted == lane.wideTiles * kDecodeRows - 1;
    entry.lookupHistory.insert(entry.lookupHistory.end(), emitted.begin(), emitted.end());
    if (entry.lookupHistory.size() > kPromptLookupHistory)
      entry.lookupHistory.erase(entry.lookupHistory.begin(),
                                entry.lookupHistory.end() - kPromptLookupHistory);
  }

  // Batch lanes beyond the active width replay the last active request so
  // every padded M32 lane binds valid state.
  static Request &laneEntry(std::span<Request *const> entries, uint32_t lane) {
    Request *entry = entries[std::min<size_t>(lane, entries.size() - 1)];
    if (!entry)
      throw std::invalid_argument("empty decode batch lane");
    return *entry;
  }

  void bindDraftRings(
      std::span<Request *const> entries,
      std::vector<std::array<MetalBuffer, kLaneCount>> &keys,
      std::vector<std::array<MetalBuffer, kLaneCount>> &values) const {
    keys.resize(geometry.draft.layers);
    values.resize(geometry.draft.layers);
    for (uint32_t layer = 0; layer < geometry.draft.layers; ++layer) {
      for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
        const auto &ring =
            states.draft(laneEntry(entries, lane).stateLane)[layer];
        keys[layer][lane] = ring.keys;
        values[layer][lane] = ring.values;
      }
    }
  }

  // The device params (SPLASH_DRAFT_AHEAD) replace the host attention and
  // selector params, whose cache lengths and anchors are then only templates.
  void encodeDraftBatchGraph(CommandGraph &graph,
                             std::span<Request *const> entries,
                             std::span<const uint64_t> logicalPositions,
                             MetalBuffer deviceAttentionParams = {},
                             MetalBuffer deviceSelectorParams = {}) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != logicalPositions.size()) {
      throw std::invalid_argument("invalid draft decode batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    // The draft shares the target's vocabulary head and its storage rows.
    const uint32_t storage = targetModel.decodeStorageLanes(lanes);
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, storage);
    };
    std::array<uint32_t, kLaneCount> cacheLengths{};
    for (uint32_t lane = 0; lane < lanes; ++lane)
      cacheLengths[lane] = static_cast<uint32_t>(logicalPositions[lane]);

    DFlashDecodeBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    for (uint32_t hidden = 0; hidden < buffers.hidden.size(); ++hidden) {
      buffers.hidden[hidden] = d(static_cast<DecodeTensor>(
          static_cast<uint32_t>(DecodeTensor::DraftHidden0) + hidden));
    }
    buffers.normalized = d(DecodeTensor::DraftNormalized);
    buffers.dynamic = d(DecodeTensor::DraftDynamic);
    buffers.convolved = d(DecodeTensor::DraftConvolved);
    buffers.proposalQkv = d(DecodeTensor::DraftProposalQkv);
    buffers.attention = d(DecodeTensor::DraftAttention);
    buffers.projected = d(DecodeTensor::DraftProjected);
    buffers.residual = d(DecodeTensor::DraftResidual);
    buffers.intermediate = d(DecodeTensor::DraftIntermediate);
    buffers.finalHidden = d(DecodeTensor::DraftFinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    buffers.selectorHidden = d(DecodeTensor::SelectorHidden);
    buffers.queryKeys = d(DecodeTensor::DraftQueryKeys);
    buffers.queryValues = d(DecodeTensor::DraftQueryValues);
    buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
    buffers.gateScratch = decodeArena->gateScratch();
    bindDraftRings(entries, buffers.persistentKeys, buffers.persistentValues);
    // One lane only: the head's segment rows belong to one request at a time.
    const bool restrictedHead =
        lanes == 1 && !laneEntry(entries, 0).draftHeadSegment.empty();
    if (restrictedHead)
      draftModel.useHeadSegment(laneEntry(entries, 0).id,
                                laneEntry(entries, 0).draftHeadSegmentVersion,
                                laneEntry(entries, 0).draftHeadSegment,
                                targetModel.vocabularyProjection());
    draftModel.addDecode(graph, std::move(buffers),
                         targetModel.vocabularyProjection(),
                         std::span(cacheLengths).first(lanes), restrictedHead,
                         std::move(deviceAttentionParams));
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      Request &entry = laneEntry(entries, lane);
      if (!entry.pendingToken)
        throw std::invalid_argument("draft batch lane has no anchor");
      anchors[lane] = *entry.pendingToken;
      policies[lane] = samplingPolicy(entry);
    }
    draftModel.addSelection(
        graph,
        {d(DecodeTensor::Logits), d(DecodeTensor::TopPartialIds),
         d(DecodeTensor::TopPartialValues), d(DecodeTensor::Candidates),
         d(DecodeTensor::Unary), d(DecodeTensor::SelectorHidden),
         d(DecodeTensor::SamplingUniforms), d(DecodeTensor::ProposedTokens),
         d(DecodeTensor::ProposalProbs), std::move(deviceSelectorParams)},
        std::span(anchors).first(lanes), std::span(policies).first(lanes),
        restrictedHead);
  }

  // wide: one request's aliased lanes (SPLASH_WIDE_PROMPT_LOOKUP).
  void encodeTargetVerifyBatchForward(CommandGraph &graph,
                                      std::span<Request *const> entries,
                                      std::span<const ModelBatchItem> items,
                                      bool wide = false) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size()) {
      throw std::invalid_argument("invalid target verify batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    const uint32_t storage = targetModel.decodeStorageLanes(lanes);
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, storage);
    };

    std::array<ChunkedPrefillParams, kLaneCount> chunks{};
    const uint32_t gdnLayers = geometry.target.stateLayout.layers;
    const uint32_t attentionLayers =
        geometry.target.kvLayout.attentionLayers;
    std::vector<MetalBuffer> gdnPacked(gdnLayers);
    std::vector<MetalBuffer> gdnMixed(gdnLayers);
    std::vector<MetalBuffer> gdnDecay(gdnLayers);
    std::vector<MetalBuffer> gdnBeta(gdnLayers);
    std::vector<MetalBuffer> chunkKeys(attentionLayers);
    std::vector<MetalBuffer> chunkValues(attentionLayers);
    QwenTargetVerifyBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    buffers.hidden = {d(DecodeTensor::Hidden0), d(DecodeTensor::Hidden1)};
    buffers.normalized = d(DecodeTensor::Normalized);
    buffers.gdnHidden = d(DecodeTensor::GdnHidden);
    buffers.gdnOutput = d(DecodeTensor::GdnOutput);
    buffers.denseIntermediate = d(DecodeTensor::Intermediate);
    buffers.fullPacked = d(DecodeTensor::FullPacked);
    buffers.fullQueries = d(DecodeTensor::FullQueries);
    buffers.attentionPartials = d(DecodeTensor::AttentionPartials);
    buffers.attentionStatistics = d(DecodeTensor::AttentionStatistics);
    buffers.fullAttention = d(DecodeTensor::FullAttention);
    buffers.attentionHidden = d(DecodeTensor::AttentionHidden);
    buffers.attentionOutput = d(DecodeTensor::AttentionOutput);
    buffers.ropeCos = d(DecodeTensor::RopeCos);
    buffers.ropeSin = d(DecodeTensor::RopeSin);
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.finalHidden = d(DecodeTensor::FinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    buffers.denseGateScratch = decodeArena->gateScratch();
    buffers.gdnPacked = gdnPacked;
    buffers.gdnMixed = gdnMixed;
    buffers.gdnDecay = gdnDecay;
    buffers.gdnBeta = gdnBeta;
    buffers.chunkKeys = chunkKeys;
    buffers.chunkValues = chunkValues;
    buffers.moe = decodeArena->moeScratch(storage);
    // SPLASH_M24_PAD3 (QwenTarget decides per encode): a three-lane verify
    // may run its input RMS and projections over the arena's fourth lane,
    // which nothing in this command reads. A command that keeps work in
    // flight on that lane must not pass these views.
    std::vector<MetalBuffer> gdnPackedPadded;
    if (lanes == 3 && storage == 3 && kLaneCount == 4) {
      buffers.hiddenPadded = {decodeArena->batchSlice(DecodeTensor::Hidden0, 4),
                              decodeArena->batchSlice(DecodeTensor::Hidden1, 4)};
      buffers.normalizedPadded = decodeArena->batchSlice(DecodeTensor::Normalized, 4);
      buffers.fullPackedPadded = decodeArena->batchSlice(DecodeTensor::FullPacked, 4);
      gdnPackedPadded.resize(gdnLayers);
      for (uint32_t layer = 0; layer < gdnLayers; ++layer)
        gdnPackedPadded[layer] = decodeArena->gdnBatchSlice(DecodeTensor::VerifyPackedBase, layer, 4);
      buffers.gdnPackedPadded = gdnPackedPadded;
    }
    for (uint32_t lane = 0; lane < lanes; ++lane)
      chunks[lane] = ops::PagedAttention::verifyParams(
          items[lane].logicalPosition,
          static_cast<uint32_t>(items[lane].pageTable.size()));
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      Request &entry = laneEntry(entries, lane);
      buffers.pageTables[lane] =
          decodeArena->get(entry.stateLane, DecodeTensor::PageTable);
      buffers.currentGdnStates[lane] = states.current(entry.stateLane).stateBase;
      buffers.nextGdnStates[lane] = states.next(entry.stateLane).stateBase;
    }
    for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
      gdnPacked[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyPackedBase, layer, storage);
      gdnMixed[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyMixedBase, layer, storage);
      gdnDecay[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyDecayBase, layer, storage);
      gdnBeta[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyBetaBase, layer, storage);
    }
    // SPLASH_GDN_DEFER: this cycle's rows go to the lane slot the pending
    // rows (if any) do not use; the scans replay those from the next cell.
    std::vector<MetalBuffer> pendingMixed, pendingDecay, pendingBeta;
    if (!wide && lanes == 1 && entries[0]->gdnDeferCycle) {
      Request &entry = *entries[0];
      if (gdnPending && gdnPending->requestId != entry.id)
        throw std::logic_error("another request's GDN commit is pending");
      const uint32_t rows = gdnPending ? gdnPending->rows : 0;
      const uint32_t pendingSlot = gdnPending ? gdnPending->slot : 1;
      entry.gdnDeferSlot = 1 - pendingSlot;
      pendingMixed.resize(gdnLayers);
      pendingDecay.resize(gdnLayers);
      pendingBeta.resize(gdnLayers);
      for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
        gdnMixed[layer] = decodeArena->gdnLaneSlice(DecodeTensor::VerifyMixedBase, layer, entry.gdnDeferSlot);
        gdnDecay[layer] = decodeArena->gdnLaneSlice(DecodeTensor::VerifyDecayBase, layer, entry.gdnDeferSlot);
        gdnBeta[layer] = decodeArena->gdnLaneSlice(DecodeTensor::VerifyBetaBase, layer, entry.gdnDeferSlot);
        pendingMixed[layer] = decodeArena->gdnLaneSlice(DecodeTensor::VerifyMixedBase, layer, pendingSlot);
        pendingDecay[layer] = decodeArena->gdnLaneSlice(DecodeTensor::VerifyDecayBase, layer, pendingSlot);
        pendingBeta[layer] = decodeArena->gdnLaneSlice(DecodeTensor::VerifyBetaBase, layer, pendingSlot);
      }
      buffers.gdnDefer = true;
      buffers.gdnPendingRows = rows;
      buffers.gdnDeferBase = rows ? states.next(entry.stateLane).stateBase : states.current(entry.stateLane).stateBase;
      buffers.gdnPendingMixed = pendingMixed;
      buffers.gdnPendingDecay = pendingDecay;
      buffers.gdnPendingBeta = pendingBeta;
    } else if (gdnPending) {
      throw std::logic_error("a pending GDN commit was not flushed before its state was read");
    }
    for (uint32_t layer = 0; layer < attentionLayers; ++layer) {
      chunkKeys[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkKeysBase, layer, storage);
      chunkValues[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkValuesBase, layer, storage);
    }
    if (wide) {
      if ((lanes != 2 && lanes != 4) ||
          std::any_of(entries.begin(), entries.end(), [&](Request *entry) { return entry != entries[0]; }))
        throw std::logic_error("wide lookup must be one aliased request");
      targetModel.addVerify16(graph, std::move(buffers), kvPages.layers(), std::span(chunks).first(lanes),
                              lanes, decodeArena->wideConvolutionScratch(), entries[0]->wideGdn);
      return;
    }
    targetModel.addVerify(graph, std::move(buffers), kvPages.layers(),
                          std::span(chunks).first(lanes), lanes);
  }

  void encodeTargetVerifyBatchPolicy(CommandGraph &graph,
                                     std::span<Request *const> entries) {
    if (entries.empty() || entries.size() > kLaneCount)
      throw std::invalid_argument("invalid target policy batch");
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    std::array<uint32_t, kLaneCount> stateLanes{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (!entries[lane])
        throw std::invalid_argument("empty target policy lane");
      policies[lane] = samplingPolicy(*entries[lane]);
      stateLanes[lane] = entries[lane]->stateLane;
    }
    sampling.addVerify(graph, std::span(policies).first(lanes),
                       samplingBuffers(lanes), geometry.target.stopTokens[0],
                       geometry.target.stopTokens[1],
                       {penaltyTable, std::span(stateLanes).first(lanes)});
  }

  // retainedOverride: each aliased lane's retained rows (a wide lookup).
  void encodeDraftStateCommitBatch(CommandGraph &graph,
                                   std::span<Request *const> entries,
                                   std::span<const ModelBatchItem> items,
                                   MetalBuffer retainedOverride = {}) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size()) {
      throw std::invalid_argument("invalid draft state commit batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, lanes);
    };

    std::array<uint32_t, kLaneCount> startPositions{};
    for (uint32_t lane = 0; lane < lanes; ++lane)
      startPositions[lane] = static_cast<uint32_t>(items[lane].logicalPosition);
    DFlashContextBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.projected = d(DecodeTensor::ContextProjected);
    buffers.hidden = d(DecodeTensor::ContextHidden);
    buffers.contextKv = d(DecodeTensor::ContextKv);
    buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
    buffers.retainedCounts = retainedOverride ? retainedOverride : d(DecodeTensor::RetainedCount);
    bindDraftRings(entries, buffers.persistentKeys, buffers.persistentValues);
    draftModel.addContextCommit(graph, std::move(buffers),
                                std::span(startPositions).first(lanes));
  }

  void encodeBatchAcceptance(CommandGraph &graph,
                             std::span<Request *const> lanes,
                             std::span<const uint32_t> maximumRetained) {
    if (lanes.empty() || lanes.size() > kLaneCount ||
        lanes.size() != maximumRetained.size()) {
      throw std::invalid_argument("invalid DFlash acceptance batch");
    }
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
      if (!lanes[lane] || !maximumRetained[lane] ||
          maximumRetained[lane] > kDecodeRows) {
        throw std::invalid_argument("invalid DFlash acceptance lane");
      }
      policies[lane] = samplingPolicy(*lanes[lane]);
    }
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    sampling.addAcceptance(
        graph, acceptanceBuffers(width),
        maximumRetained, std::span(policies).first(width),
        geometry.target.stopTokens[0], geometry.target.stopTokens[1]);
  }

  void encodeBatchEmbedding(CommandGraph &graph, DecodeTensor tokens,
                            DecodeTensor output, uint32_t lanes) {
    if (!lanes || lanes > kLaneCount)
      throw std::invalid_argument("invalid embedding batch width");
    const uint32_t rows = lanes * kDecodeRows;
    targetModel.addEmbedding(graph, decodeArena->batchSlice(tokens, lanes),
                             decodeArena->batchSlice(output, lanes), rows);
  }

  void encodeBatchVerifyInput(CommandGraph &graph, uint32_t lanes) {
    if (!lanes || lanes > kLaneCount)
      throw std::invalid_argument("invalid verify-input batch width");
    targetModel.addVerifyInput(
        graph, decodeArena->batchSlice(DecodeTensor::DraftInputTokens, lanes),
        decodeArena->batchSlice(DecodeTensor::ProposedTokens, lanes),
        decodeArena->batchSlice(DecodeTensor::InputTokens, lanes), lanes);
  }

  // SPLASH_GDN_DEFER (default on; =0 off): whether a step of `width` lanes
  // defers its GDN commit.
  bool gdnDeferAllowed(uint32_t width, bool wide) {
    if (width != 1 || wide || !metal::envSwitch("SPLASH_GDN_DEFER"))
      return false;
    if (!gdnDeferRoute)
      gdnDeferRoute = targetModel.gdnDeferSupported();
    return *gdnDeferRoute;
  }

  // SPLASH_GDN_DEFER: the pending recurrent commit as a command of its own,
  // completed before this returns (and before anything is encoded after it).
  void flushGdn() {
    if (!gdnPending)
      return;
    CommandGraph graph;
    targetModel.addStateFlush(
        graph,
        {decodeArena->gdnStorage(DecodeTensor::VerifyMixedBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyDecayBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyBetaBase), states.next(gdnPending->stateLane).stateBase,
         states.current(gdnPending->stateLane).stateBase},
        gdnPending->rows, gdnPending->slot);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches(), {}).wait());
    gdnPending.reset();
  }

  // wide: one request's aliased lanes, whose total retained count lane 0 holds.
  void encodeBatchGdnCommit(CommandGraph &graph,
                            std::span<Request *const> lanes, bool wide = false) {
    if (lanes.empty() || lanes.size() > kLaneCount)
      throw std::invalid_argument("invalid GDN commit batch");
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    std::array<MetalBuffer, kLaneCount> currentStates;
    std::array<MetalBuffer, kLaneCount> nextStates;
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      Request *entry = lanes[std::min(lane, width - 1)];
      if (!entry)
        throw std::invalid_argument("empty GDN commit lane");
      currentStates[lane] = states.current(entry->stateLane).stateBase;
      nextStates[lane] = states.next(entry->stateLane).stateBase;
    }
    QwenTargetCommitBuffers buffers{
        decodeArena->gdnStorage(DecodeTensor::VerifyPackedBase),
        decodeArena->gdnStorage(DecodeTensor::VerifyMixedBase),
        decodeArena->gdnStorage(DecodeTensor::VerifyDecayBase),
        decodeArena->gdnStorage(DecodeTensor::VerifyBetaBase), currentStates,
        nextStates,
        decodeArena->batchSlice(DecodeTensor::RetainedCount, width)};
    if (wide)
      targetModel.addStateCommit16(graph, std::move(buffers), decodeArena->wideConvolutionScratch(), width);
    else if (width == 1 && lanes[0]->gdnDeferCycle)  // SPLASH_GDN_DEFER: the carry only
      targetModel.addStateCommitConv(graph, std::move(buffers));
    else
      targetModel.addStateCommit(graph, std::move(buffers), width);
  }

  // SPLASH_DRAFT_HEAD_IDS: rare output tokens (e.g. another language) join the
  // request's head segment; once it is full the request uses the full head.
  void noteDraftHeadTokens(Request &entry, std::span<const uint32_t> output) const {
    auto &segment = entry.draftHeadSegment;
    if (segment.empty())
      return;
    for (const uint32_t token : output) {
      if (token >= geometry.target.vocabularySize || draftModel.staticHeadHas(token))
        continue;
      const size_t used = entry.draftHeadSegmentUsed;
      const size_t at = std::find(segment.begin(), segment.end(), token) - segment.begin();
      if (at < used)
        continue;  // already protected
      if (at < segment.size()) {
        std::swap(segment[at], segment[used]);  // filler id seen in output: protect it
      } else if (used == segment.size()) {
        segment.clear();  // more rare ids than rows: full head from now on
        return;
      } else {
        segment[used] = token;  // evicts a filler id
      }
      ++entry.draftHeadSegmentUsed;
      ++entry.draftHeadSegmentVersion;
    }
  }

  // A stop token or the last budgeted token needs no target work of its own:
  // the next cycle would only echo it as output. Emitting it as soon as it is
  // selected saves that cycle; the engine is told it has no KV row.
  bool emitTerminalAnchor(Request &entry, ModelStepResult &result) const {
    const bool stop = isStopToken(geometry, *entry.pendingToken);
    if (!stop && entry.maxNewTokens - entry.generatedTokens != 1)
      return false;
    result.outputTokens.push_back(*entry.pendingToken);
    result.outputTokensWithoutKv = 1;
    result.finished = stop;
    ++entry.generatedTokens;
    return true;
  }

  // SPLASH_ROW_HASH: a wide lookup's rows span its aliased lanes; a
  // multi-request batch hashes each lane's own eight rows.
  void printRowHashes(const DecodeLaneResult &laneResult, uint32_t lane, const ModelBatchItem &item,
                      size_t batchLanes) const {
    const uint32_t tiles = laneResult.wideLookup ? laneResult.wideTiles : 1;
    const auto rowsOf = [&](DecodeTensor tensor) {
      return laneResult.wideLookup ? decodeArena->batchSlice(tensor, tiles) : decodeArena->get(lane, tensor);
    };
    const MetalBuffer hidden = rowsOf(DecodeTensor::FinalHidden);
    const MetalBuffer logits = rowsOf(DecodeTensor::Logits);
    const auto *hiddenBytes = contents<char>(hidden, "row hash");
    const auto *logitBytes = contents<char>(logits, "row hash");
    const auto *tokens = contents<uint32_t>(rowsOf(DecodeTensor::InputTokens), "row hash tokens");
    const uint32_t rows = tiles * kDecodeRows;
    const size_t hiddenRow = hidden.sizeBytes() / rows, logitRow = logits.sizeBytes() / rows;
    const std::hash<std::string_view> hash;
    for (uint32_t row = 0; row <= laneResult.accepted && row < laneResult.retained; ++row)
      std::fprintf(stderr,
                   "row_hash request=%llu position=%llu token=%u wide=%u row=%u hidden=%zx logits=%zx lanes=%zu\n",
                   static_cast<unsigned long long>(item.requestId),
                   static_cast<unsigned long long>(item.logicalPosition + row), tokens[row], tiles - 1, row,
                   hash(std::string_view(hiddenBytes + row * hiddenRow, hiddenRow)),
                   hash(std::string_view(logitBytes + row * logitRow, logitRow)), batchLanes);
  }

  // A decode lane's selected tokens: a wide lookup's over its aliased lanes.
  MetalBuffer outputTokens(const DecodeLaneResult &lane, uint32_t index) const {
    return lane.wideLookup ? decodeArena->batchSlice(DecodeTensor::OutputTokens, lane.wideTiles)
                           : decodeArena->get(index, DecodeTensor::OutputTokens);
  }

  std::vector<ModelStepResult> finalizeDecode(
      std::span<DecodeLaneResult> lanes, std::span<const ModelBatchItem> items,
      CommandTiming timing) {
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      auto d = [&](DecodeTensor tensor) {
        return decodeArena->get(lane, tensor);
      };
      laneResult.retained = *contents<uint32_t>(d(DecodeTensor::RetainedCount),
                                                "GPU retained token count");
      laneResult.accepted = *contents<uint32_t>(d(DecodeTensor::AcceptedCount),
                                                "GPU accepted draft count");
      // A wide lookup's rows span its aliased lanes (lane 0 holds the counts).
      const uint32_t verifyRows = laneResult.wideLookup ? laneResult.wideTiles * kDecodeRows : kDecodeRows;
      if (!laneResult.retained || laneResult.retained > verifyRows)
        throw std::runtime_error("target policy produced invalid retention");
      if (laneResult.accepted > verifyRows - 1)
        throw std::runtime_error(
            "target accepted more than the draft proposed");
      // The retained target tokens end with the next anchor. A non-finite
      // target row can also accept a sentinel draft proposal as an interior
      // token, so every retained token is checked.
      const uint32_t *targetTokens = contents<uint32_t>(outputTokens(laneResult, lane), "target output tokens");
      laneResult.failure = invalidSelection({targetTokens, laneResult.retained});
      if (rowHash)
        printRowHashes(laneResult, lane, items[lane], items.size());
    }

    std::vector<ModelStepResult> results;
    results.reserve(items.size());
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      Request &entry = *laneResult.request;
      if (!laneResult.failure.empty()) {
        // The cycle's state and tokens are not committed; the engine ends
        // the request (Runtime::end drops its pending GDN commit).
        entry.gdnDeferCycle = false;
        entry.maskWords.clear();
        entry.verifyMaskInFlight = false;
        results.push_back({.requestId = entry.id,
                           .failure = std::move(laneResult.failure)});
        continue;
      }
      const uint32_t *targetTokens =
          contents<uint32_t>(outputTokens(laneResult, lane), "target output tokens");
      std::vector<uint32_t> output;
      output.reserve(laneResult.retained);
      output.push_back(laneResult.currentAnchor);
      output.insert(output.end(), targetTokens,
                    targetTokens + (laneResult.retained - 1));
      noteDraftHeadTokens(entry, output);

      states.swapParity(entry.stateLane);
      // SPLASH_GDN_DEFER: its recurrent rows wait for the next scan or a flush,
      // replayed from the cell that is now next.
      if (std::exchange(entry.gdnDeferCycle, false))
        gdnPending = GdnPending{entry.id, entry.stateLane, laneResult.retained, entry.gdnDeferSlot};
      const uint64_t nextLength =
          items[lane].logicalPosition + laneResult.retained;
      states.updateLengths(
          entry.stateLane,
          advanceDraftContext(states.metadata(entry.stateLane).lengths,
                              nextLength, items[lane].logicalPosition,
                              nextLength, false));
      entry.generatedTokens += laneResult.retained;
      commitSelected(entry, {targetTokens, laneResult.retained});
      if (ahead && !ahead->finalized && lane < ahead->lanes &&
          ahead->requestIds[lane] == entry.id) {
        ahead->positions[lane] += laneResult.retained;
        ahead->anchors[lane] = *entry.pendingToken;
      }
      entry.maskWords.clear();
      entry.verifyMaskInFlight = false;
      entry.verifyRows = kDecodeRows;
      results.push_back({entry.id,
                         0,
                         std::move(output),
                         false,
                         DecodeStage::Regular,
                         laneResult.wideLookup ? laneResult.wideTiles * kDecodeRows - 1 : kDraftProposalTokens,
                         std::min(laneResult.accepted, laneResult.retained - 1)});
      ModelStepResult &result = results.back();
      if (entry.generatedTokens < entry.maxNewTokens)
        emitTerminalAnchor(entry, result);
      recordLookup(entry, laneResult, result.outputTokens);
    }

    if (ahead)
      ahead->finalized = true;
    counters.lastDecodeGpuSeconds = timing.gpuSeconds;
    counters.totalDecodeGpuSeconds += timing.gpuSeconds;
    counters.lastDecodeWallSeconds = timing.wallSeconds;
    counters.totalDecodeWallSeconds += timing.wallSeconds;
    return results;
  }

  // SPLASH_DRAFT_AHEAD (default on): after a drafter cycle, the next cycle's
  // draft block is committed behind the cycle's last command and runs while
  // the host reads the result; draft_ahead_prepare derives its anchor,
  // positions and params from the device acceptance. The next decodeAsync
  // adopts it only when an inline draft would compute the same block (same
  // requests, lanes, positions, anchors, uniforms and head rows); otherwise
  // the host waits for it before writing any lane buffer and drafts inline.
  // Pulsar 1.0.0: serving -0.257 ms/cycle, outputs byte-identical.
  // SPLASH_DRAFT_AHEAD_GRAMMAR (default on, with it): constrained cycles
  // launch and adopt blocks too. Their proposals are unconstrained (the host
  // simulates the grammar over them and masks only the target), so an adopted
  // block feeds the simulation exactly what the inline draft would.
  // Read per cycle so an in-process A/B can flip them; the arena holds the
  // device params only when SPLASH_DRAFT_AHEAD was on at startup.
  bool draftAheadOn() const noexcept {
    return aheadParams && metal::envSwitch("SPLASH_DRAFT_AHEAD");
  }
  bool grammarAheadOn() const noexcept {
    return draftAheadOn() && metal::envSwitch("SPLASH_DRAFT_AHEAD_GRAMMAR");
  }

  void retireAhead() {
    if (!ahead)
      return;
    ahead.reset();
    backend.awaitTrailing();
  }

  // A released request's ring must not stay bound to a running block.
  void retireAheadFor(uint64_t requestId) {
    if (ahead && std::find(ahead->requestIds.begin(),
                           ahead->requestIds.begin() + ahead->lanes,
                           requestId) != ahead->requestIds.begin() + ahead->lanes)
      retireAhead();
  }

  // Cheap checks before any lane buffer is written: the same requests in the
  // same lanes at the positions and anchors the block was built for, an
  // unchanged head, and the stage and cohort that would draft inline.
  bool aheadMayMatch(const BatchPlan &plan,
                     std::span<const ModelBatchItem> items) {
    if (!ahead->finalized || plan.decodeStage != DecodeStage::Regular ||
        !(plan.constrained ? grammarAheadOn() : draftAheadOn()) ||
        items.size() != ahead->lanes)
      return false;
    for (uint32_t lane = 0; lane < ahead->lanes; ++lane) {
      if (items[lane].requestId != ahead->requestIds[lane] ||
          items[lane].logicalPosition != ahead->positions[lane])
        return false;
      const Request &entry = request(items[lane].requestId);
      if (entry.draftHeadSegmentVersion != ahead->headVersions[lane] ||
          !entry.pendingToken || *entry.pendingToken != ahead->anchors[lane])
        return false;
    }
    return ahead->restrictedHead ==
           (ahead->lanes == 1 &&
            !request(items[0].requestId).draftHeadSegment.empty());
  }

  // Launch rule for a block behind a cycle's last command: the switch for
  // the cohort, no row hashes, a drafter cycle (not a prompt lookup) with no lookup cycle in
  // its requests' last aheadLookupQuiet cycles, and every resident request in
  // this batch (a prefilling or waiting peer makes the next plan differ).
  bool aheadLaunchAllowed(std::span<const ModelBatchItem> items,
                          bool constrained, bool lookup) const {
    // SPLASH_ROW_HASH reads the verify's logits after the command; a block's
    // draft head writes the same Logits tensor behind it.
    return (constrained ? grammarAheadOn() : draftAheadOn()) && !lookup && !rowHash &&
           std::all_of(items.begin(), items.end(),
                       [&](const ModelBatchItem &item) {
                         return requests.at(item.requestId).cyclesSinceLookup >= aheadLookupQuiet;
                       }) &&
           std::all_of(requests.begin(), requests.end(),
                       [&](const auto &resident) {
                         return !resident.second.resident ||
                                std::any_of(items.begin(), items.end(),
                                            [&](const ModelBatchItem &item) {
                                              return item.requestId == resident.first;
                                            });
                       });
  }

  // Submits a cycle's last command; with `launch`, the next cycle's draft
  // block is built once it is committed and runs right behind it.
  CommandTicket submitWithAhead(const metal::Command &command,
                                std::function<void()> completion,
                                std::span<Request *const> entries,
                                std::span<const ModelBatchItem> items,
                                bool launch) {
    std::optional<DraftAhead> launched;
    const auto build = [&]() -> std::span<const metal::ComputeDispatch> {
      aheadGraph = CommandGraph{};
      launched = encodeDraftAhead(aheadGraph, entries, items);
      return aheadGraph.dispatches();
    };
    bool committed = false;
    CommandTicket ticket = backend.submitCommandAsync(
        command, std::move(completion),
        launch ? MetalBackend::TrailingBuilder(build) : MetalBackend::TrailingBuilder{},
        committed);
    if (committed) {
      ahead = std::move(launched);
      for (Request *entry : entries)
        ++entry->aheadLaunched;
    }
    return ticket;
  }

  // The next cycle's draft block for these lanes, fed on the GPU by this
  // cycle's acceptance. Host cache lengths and anchors are templates.
  // A trailing command: tracked buffers only (arena, draft rings, weights),
  // never a KV page (MetalBackend's commitTrailing relies on it).
  DraftAhead encodeDraftAhead(CommandGraph &graph,
                              std::span<Request *const> entries,
                              std::span<const ModelBatchItem> items) {
    static_assert(sizeof(DraftAttentionBatchParams) <= 64 &&
                  64 + sizeof(SelectorBatchParams) <= 128,
                  "draft-ahead params fit the arena's 128 bytes");
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    DraftAhead next;
    next.lanes = lanes;
    DraftAheadParams params{};
    std::array<uint32_t, kLaneCount> cacheLengths{};
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    std::array<uint64_t, kLaneCount> logicalPositions{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const Request &entry = *entries[lane];
      next.requestIds[lane] = entry.id;
      next.positions[lane] = items[lane].logicalPosition;
      next.headVersions[lane] = entry.draftHeadSegmentVersion;
      if (samplingEnabled(entry)) {
        next.uniforms[lane] = nextCycleUniforms(entry);
        params.uniform_lanes |= uint32_t{1} << lane;
        std::copy(next.uniforms[lane].begin(), next.uniforms[lane].end(),
                  params.uniforms + lane * SPLASH_SAMPLING_UNIFORMS);
      }
      params.start_position[lane] =
          static_cast<uint32_t>(items[lane].logicalPosition);
      cacheLengths[lane] = static_cast<uint32_t>(items[lane].logicalPosition);
      anchors[lane] = *entry.pendingToken;
      policies[lane] = samplingPolicy(entry);
      logicalPositions[lane] = items[lane].logicalPosition;
    }
    params.attention = DFlashDraft::attentionParams(std::span(cacheLengths).first(lanes));
    params.selector = draftModel.selectionParams(
        std::span(anchors).first(lanes), std::span(policies).first(lanes));
    params.mask_token = geometry.target.maskToken;
    next.restrictedHead = lanes == 1 && !entries[0]->draftHeadSegment.empty();

    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, lanes);
    };
    MetalBuffer attention =
        backend.view(aheadParams, 0, sizeof(DraftAttentionBatchParams));
    MetalBuffer selector =
        backend.view(aheadParams, 64, sizeof(SelectorBatchParams));
    ops::DraftAttention::addAheadPrepare(
        graph,
        {d(DecodeTensor::RetainedCount), d(DecodeTensor::OutputTokens),
         d(DecodeTensor::DraftInputTokens), d(DecodeTensor::DraftPositions),
         d(DecodeTensor::SamplingUniforms), attention, selector},
        params);
    const uint32_t rows = lanes * kDecodeRows;
    addRopeTables(graph, d(DecodeTensor::Positions), 0,
                  d(DecodeTensor::DraftPositions), rows, d(DecodeTensor::RopeCos),
                  d(DecodeTensor::RopeSin), d(DecodeTensor::DraftRopeCos),
                  d(DecodeTensor::DraftRopeSin));
    encodeBatchEmbedding(graph, DecodeTensor::DraftInputTokens,
                         DecodeTensor::DraftHidden0, lanes);
    encodeDraftBatchGraph(graph, entries, std::span(logicalPositions).first(lanes),
                          std::move(attention), std::move(selector));
    return next;
  }

  // A constrained DFlash cycle has one host dependency between three Metal
  // commands: draft proposals define the grammar simulation, while the target
  // forward is independent of the resulting mask.  This ticket keeps the
  // scheduler batch (and therefore its DecodeArena lanes) owned across that
  // dependency.  All state transitions run on the engine thread; completion
  // handlers only wake it, so they capture the wake hook and never the ticket.
  class ConstrainedDecodeTicket final : public ModelBatchTicket {
  public:
    // adopted: a SPLASH_DRAFT_AHEAD block already drafted; `draft` holds
    // only the target's RoPE tables.
    ConstrainedDecodeTicket(Impl &impl, std::vector<DecodeLaneResult> lanes,
                            std::span<const ModelBatchItem> items,
                            CommandGraph &draft, bool adopted,
                            std::function<void()> completion)
        : impl_(impl), lanes_(std::move(lanes)),
          items_(items.begin(), items.end()), adopted_(adopted),
          wake_(std::move(completion)) {
      // Pulsar SPLASH_GRAMMAR_CHAIN (default on, read per cycle; one
      // request): the target forward and the commit join the draft's command.
      // The draft's event step signals once the proposals exist; the commit
      // part waits until the host has written the masks. With
      // SPLASH_STREAMED_SUBMIT the draft commits before the rest is built.
      // Scheduling only: the same dispatches in the same order.
      if (lanes_.size() != 1 || !metal::envSwitch("SPLASH_GRAMMAR_CHAIN")) {
        submit(draft);
        return;
      }
      if (!impl_.chainEvent)
        impl_.chainEvent = impl_.backend.newSharedEvent();
      signalValue_ = ++impl_.chainValues;
      waitValue_ = ++impl_.chainValues;
      draft.signal(impl_.chainEvent, signalValue_);
      // The streamed head ends at the signal (a streamed head's event steps
      // follow its dispatches): a prompt lookup cycle's head is its rope
      // tables alone.
      if (Impl::streamChunk())
        static_cast<void>(impl_.backend.streamHead(draft.command()));
      beginVerify();
      encodeForward(draft);
      draft.wait(impl_.chainEvent, waitValue_);
      encodeCommit(draft);
      stage_ = Stage::ChainDraft;
      // The flag outlives the ticket; the callback may come after it.
      impl_.chainEvent.notify(signalValue_, [signaled = signaled_, wake = wake_] {
        signaled->store(true, std::memory_order_release);
        if (wake)
          wake();
      });
      // The commit is part of this command: SPLASH_DRAFT_AHEAD_GRAMMAR's next
      // block trails it.
      submit(draft, true);
    }

    // A chained command still waiting for masks must not wait forever: a
    // cancelled, failed or shut-down batch releases it with this ticket.
    ~ConstrainedDecodeTicket() override {
      if (waitValue_ && stage_ != Stage::Commit && stage_ != Stage::Done)
        impl_.chainEvent.signal(waitValue_);
    }

    std::vector<ModelMaskRequest> takeMaskRequests() override {
      std::vector<ModelMaskRequest> requests;
      // An adopted block's proposals are complete once the chain signals: the
      // command's first encoder writes the arena the block wrote, so Metal
      // ran it after the block. (awaitTrailing here would wait for this
      // command's own trailing block, which waits for these masks.)
      if (stage_ == Stage::ChainDraft &&
          signaled_->load(std::memory_order_acquire)) {
        requests = maskRequests();
        maskWaitStarted_ = AwakeClock::now();
        stage_ = Stage::WaitingMask;
      }
      if (stage_ == Stage::Draft && command_.ready()) {
        addTiming(command_.wait());
        awaitAdopted();
        beginVerify();
        requests = maskRequests();
        CommandGraph target;
        encodeForward(target);
        submit(target);
        stage_ = Stage::TargetForward;
      }

      if (stage_ == Stage::TargetForward && command_.ready()) {
        const CommandTiming forward = command_.wait();
        addTiming(forward);
        targetForwardGpuSeconds_ += forward.gpuSeconds;
        maskWaitStarted_ = AwakeClock::now();
        stage_ = Stage::WaitingMask;
      }

      if (stage_ == Stage::WaitingMask) {
        bool masksReady = true;
        for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
          masksReady = masksReady && (abandoned_[lane] ||
                                      !lanes_[lane].request->maskWords.empty());
        }
        if (masksReady) {
          maskWaitSeconds_ +=
              std::chrono::duration<double>(AwakeClock::now() - *maskWaitStarted_)
                  .count();
          maskWaitStarted_.reset();
          for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
            const DecodeLaneResult &laneResult = lanes_[lane];
            if (laneResult.wideLookup)
              // SPLASH_WIDE_PROMPT_LOOKUP: the masks of every aliased lane.
              impl_.uploadWideConstraintMasks(
                  laneResult.wideTiles,
                  abandoned_[lane] ? std::span<const uint32_t>{}
                                   : std::span<const uint32_t>{laneResult.request->maskWords});
            else if (abandoned_[lane])
              impl_.admitEveryToken(lane);
            else
              impl_.uploadConstraintMasks(lane, laneResult.request->maskWords);
          }
          if (waitValue_) {
            impl_.chainEvent.signal(waitValue_);
          } else {
            CommandGraph commit;
            encodeCommit(commit);
            submit(commit, true);
          }
          stage_ = Stage::Commit;
        }
      }
      return requests;
    }

    bool ownsMaskWait(uint64_t requestId) const noexcept override {
      if (stage_ == Stage::Draft || stage_ == Stage::ChainDraft ||
          stage_ == Stage::Done)
        return false;
      return std::any_of(lanes_.begin(), lanes_.end(),
                         [requestId](const DecodeLaneResult &lane) {
                           return lane.request->id == requestId;
                         });
    }

    void abandonMask(uint64_t requestId) noexcept override {
      if (stage_ == Stage::Done)
        return;
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        if (lanes_[lane].request->id == requestId) {
          abandoned_[lane] = true;
          lanes_[lane].request->maskWords.clear();
          return;
        }
      }
    }

    bool ready() const noexcept override {
      return stage_ == Stage::Commit && command_.ready();
    }

    std::vector<ModelStepResult> wait() override {
      if (!ready())
        throw std::logic_error("constrained decode ticket is not complete");
      addTiming(command_.wait());
      stage_ = Stage::Done;
      ModelTelemetry &counters = impl_.counters;
      ++counters.constrainedMaskOverlapBatches;
      counters.constrainedMaskOverlapRequests += lanes_.size();
      counters.lastConstrainedTargetForwardGpuSeconds =
          targetForwardGpuSeconds_;
      counters.totalConstrainedTargetForwardGpuSeconds +=
          targetForwardGpuSeconds_;
      counters.lastConstrainedMaskWaitSeconds = maskWaitSeconds_;
      counters.totalConstrainedMaskWaitSeconds += maskWaitSeconds_;
      return impl_.finalizeDecode(lanes_, items_, timing_);
    }

    double wallMilliseconds() const noexcept override {
      return timing_.wallSeconds * 1000.0;
    }

  private:
    enum class Stage : uint8_t {
      Draft,
      TargetForward,
      ChainDraft,  // SPLASH_GRAMMAR_CHAIN: one command, proposals pending
      WaitingMask,
      Commit,
      Done
    };

    // The lanes' mask state once their proposals are fixed.
    void beginVerify() {
      for (DecodeLaneResult &laneResult : lanes_) {
        laneResult.request->maskWords.clear();
        laneResult.request->verifyMaskInFlight = true;
      }
    }

    // One grammar simulation per live lane: the anchor and the proposals.
    std::vector<ModelMaskRequest> maskRequests() const {
      std::vector<ModelMaskRequest> requests;
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        if (abandoned_[lane])
          continue;
        const Request &entry = *lanes_[lane].request;
        const uint32_t *proposed = contents<uint32_t>(
            impl_.decodeArena->get(lane, DecodeTensor::ProposedTokens),
            "constrained draft proposals");
        ModelMaskRequest request;
        request.requestId = entry.id;
        if (lanes_[lane].wideLookup) {
          // SPLASH_WIDE_PROMPT_LOOKUP: the anchor and every proposal.
          const uint32_t *input = contents<uint32_t>(
              impl_.decodeArena->batchSlice(DecodeTensor::InputTokens, lanes_[lane].wideTiles),
              "wide mask simulation");
          request.simulationTokens.assign(input, input + entry.verifyRows);
        } else {
          request.simulationTokens.reserve(kDecodeRows);
          request.simulationTokens.push_back(*entry.pendingToken);
          request.simulationTokens.insert(request.simulationTokens.end(),
                                          proposed,
                                          proposed + kDraftProposalTokens);
        }
        requests.push_back(std::move(request));
      }
      return requests;
    }

    std::span<Request *const> entries() {
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane)
        entries_[lane] = lanes_[lane].request;
      return {entries_.data(), lanes_.size()};
    }

    // The target forward is independent of the masks.
    void encodeForward(CommandGraph &graph) {
      if (lanes_[0].wideLookup) {
        impl_.encodeLookup16Forward(graph, *lanes_[0].request, items_[0]);
        return;
      }
      const uint32_t width = static_cast<uint32_t>(lanes_.size());
      impl_.encodeBatchVerifyInput(graph, width);
      impl_.encodeBatchEmbedding(graph, DecodeTensor::InputTokens,
                                 DecodeTensor::Hidden0, width);
      impl_.encodeTargetVerifyBatchForward(graph, entries(), items_);
    }

    // Policy (reads the masks), acceptance and commits.
    void encodeCommit(CommandGraph &graph) {
      if (lanes_[0].wideLookup) {
        impl_.encodeLookup16Commit(graph, *lanes_[0].request, items_[0], lanes_[0].maximumRetained);
        return;
      }
      std::array<uint32_t, kLaneCount> maximumRetained{};
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane)
        maximumRetained[lane] = lanes_[lane].maximumRetained;
      impl_.encodeTargetVerifyBatchPolicy(graph, entries());
      impl_.encodeBatchAcceptance(graph, entries(),
                                  {maximumRetained.data(), lanes_.size()});
      impl_.encodeBatchGdnCommit(graph, entries());
      impl_.encodeDraftStateCommitBatch(graph, entries(), items_);
    }

    // launch: the cycle's last command, which a SPLASH_DRAFT_AHEAD_GRAMMAR
    // block may trail.
    void submit(const CommandGraph &graph, bool launch = false) {
      command_ = launch
          ? impl_.submitWithAhead(graph.command(), wake_, entries(), items_,
                                  impl_.aheadLaunchAllowed(items_, true, lanes_[0].promptLookup))
          : impl_.backend.submitCommandAsync(graph.command(), wake_);
    }

    // An adopted draft-ahead block wrote the proposals the host reads next
    // (three-command path: nothing has trailed this cycle yet).
    void awaitAdopted() {
      if (adopted_)
        impl_.backend.awaitTrailing();
    }

    void addTiming(CommandTiming value) noexcept {
      timing_.gpuSeconds += value.gpuSeconds;
      timing_.wallSeconds += value.wallSeconds;
    }

    Impl &impl_;
    std::vector<DecodeLaneResult> lanes_;
    std::vector<ModelBatchItem> items_;
    bool adopted_ = false;
    Stage stage_ = Stage::Draft;
    CommandTicket command_;
    CommandTiming timing_;
    std::array<bool, kLaneCount> abandoned_{};
    std::array<Request *, kLaneCount> entries_{};
    double targetForwardGpuSeconds_ = 0.0;
    double maskWaitSeconds_ = 0.0;
    std::optional<AwakeClock::time_point> maskWaitStarted_;
    std::function<void()> wake_;
    // SPLASH_GRAMMAR_CHAIN: the event values of a chained cycle (0: none).
    uint64_t signalValue_ = 0;
    uint64_t waitValue_ = 0;
    std::shared_ptr<std::atomic<bool>> signaled_ =
        std::make_shared<std::atomic<bool>>(false);
  };
};

Runtime::Runtime(RuntimeContext context)
    : impl_(std::make_unique<Impl>(context)) {}

Runtime::~Runtime() = default;

void Runtime::checkHealth() { impl_->backend.checkHealth(); }

void Runtime::beginColdRequest(const ModelRequest &request,
                               uint32_t stateLane) {
  if (const StateAdmission admission = beginAt(request, stateLane); !admission.granted()) {
    throw metal::MetalAllocationError(
        std::string("unable to allocate a lane's state: ") +
            metal::allocationFailureName(admission.allocationFailure),
        admission.allocationFailure);
  }
  try {
    setDraftContextPlan(
        request.id,
        planDraftContext(0, static_cast<uint32_t>(request.prompt.size()), {}));
  } catch (...) {
    end(request.id);
    throw;
  }
}

StateAdmission Runtime::begin(const ModelRequest &request) {
  Impl::VisionRollback rollback{*impl_, impl_->vision};
  StateAdmission admission = admitIdleLane(
      impl_->states, [&](uint32_t lane) { return beginAt(request, lane); });
  rollback.committed = admission.granted();
  return admission;
}

void Runtime::suspend(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  impl_->retireAhead();  // SPLASH_DRAFT_AHEAD: its lane and ring may be reused
  if (!entry.resident || entry.verifyMaskInFlight) {
    throw std::logic_error("Qwen request cannot be suspended");
  }
  if (impl_->gdnPending && impl_->gdnPending->requestId == requestId)
    impl_->gdnPending.reset();  // SPLASH_GDN_DEFER: the lane's state goes
  impl_->states.releaseLane(entry.stateLane, requestId);
  impl_->pageTableBindings[entry.stateLane] = {};
  impl_->releaseImages(entry);
  entry.draftContextPlan.reset();
  entry.replayingGeneration |= entry.promptComplete;
  entry.promptComplete = false;
  entry.resident = false;
}

StateAdmission Runtime::resume(const ModelRequest &request) {
  Impl::Request &entry = impl_->request(request.id);
  if (entry.resident) {
    throw std::logic_error("Qwen request is not suspended");
  }
  if (request.prompt.size() < entry.promptTokens) {
    throw std::invalid_argument("recomputed history cannot shorten the prompt");
  }
  Impl::VisionRollback rollback{*impl_, impl_->vision};
  std::vector<Impl::ImageState> images;
  StateAdmission admission = admitIdleLane(impl_->states, [&](uint32_t lane) {
    return impl_->activate(request, lane, images);
  });
  if (admission.granted()) {
    entry.stateLane = *admission.lane;
    entry.resident = true;
    entry.promptTokens = static_cast<uint32_t>(request.prompt.size());
    entry.images = std::move(images);
    entry.restoredTokens = request.restoredTokens;
    impl_->bindPenalties(entry, request.prompt);
  }
  rollback.committed = admission.granted();
  return admission;
}

StateAdmission Runtime::beginAt(const ModelRequest &request, uint32_t stateLane) {
  if (!request.id || stateLane >= kLaneCount || request.prompt.empty()) {
    throw std::invalid_argument("invalid executor request activation");
  }
  if (impl_->requests.contains(request.id)) {
    throw std::logic_error("request is already active");
  }
  Impl::Request entry;
  entry.id = request.id;
  entry.promptTokens = static_cast<uint32_t>(request.prompt.size());
  entry.maxNewTokens = request.maxNewTokens;
  entry.sampling = request.sampling;
  entry.constraint = request.constraint;
  entry.flags = request.flags;
  entry.scoreTokens.assign(request.scoreTokens.begin(),
                           request.scoreTokens.end());
  entry.draftHeadSegment =
      impl_->draftModel.promptSegment(request.prompt, &entry.draftHeadSegmentUsed);
  std::vector<Impl::ImageState> images;
  const StateAdmission admission = impl_->activate(request, stateLane, images);
  if (!admission.granted())
    return admission;
  entry.stateLane = stateLane;
  entry.resident = true;
  entry.images = std::move(images);
  entry.restoredTokens = request.restoredTokens;
  impl_->bindPenalties(entry, request.prompt);
  // SPLASH_PROMPT_LOOKUP: text requests that generate keep a lookup history.
  if (impl_->promptLookupEnabled && request.scoreTokens.empty() && request.images.empty() &&
      request.imagePixels.empty()) {
    const size_t count = std::min(request.prompt.size(), kPromptLookupHistory);
    entry.lookupHistory.reserve(kPromptLookupHistory + kDecodeRows);
    entry.lookupHistory.assign(request.prompt.end() - count, request.prompt.end());
  }
  // SPLASH_WIDE_LOOKUP32=alt / SPLASH_WIDE_GDN_SINGLE=alt*: every other
  // admitted request (in-run A/B).
  entry.wide32 = impl_->wide32Mode &&
                 (std::string_view(impl_->wide32Mode) == "1" || impl_->wide32Admitted++ % 2 == 1);
  entry.wideGdn = !impl_->wideGdnAlternate || impl_->wideGdnAdmitted++ % 2 == 1 ? impl_->wideGdnRoute
                                                                                 : ops::WideGdn::Chain;
  auto [_, inserted] = impl_->requests.emplace(request.id, std::move(entry));
  if (!inserted) {
    throw std::logic_error("request insertion lost uniqueness");
  }
  return admission;
}

std::unique_ptr<StateRestore> Runtime::beginRestore(
    uint64_t requestId, uint32_t boundary,
    std::shared_ptr<const CompositeState> state, bool restoreDraft,
    std::function<void()> completion) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || !state || boundary >= entry.promptTokens)
    throw std::invalid_argument("invalid state restore");
  impl_->retireAhead();  // SPLASH_DRAFT_AHEAD: its ring is rewritten
  return impl_->states.beginRestore(entry.stateLane, *state, restoreDraft,
      std::move(completion), [this, requestId, boundary, restoreDraft] {
        finishRestore(requestId, boundary, restoreDraft);
      });
}

void Runtime::finishRestore(uint64_t requestId, uint32_t restoredPrefixLength,
                            bool restoreDraftState) {
  Impl::Request &entry = impl_->request(requestId);
  // A shorter restore would replay rows of images that were never staged.
  if (restoredPrefixLength < entry.restoredTokens)
    throw std::invalid_argument("restore stops before images its activation left out");
  if (!restoreDraftState)
    ++impl_->counters.draftStateRestoreSkipped;
  const QwenLogicalLengths &lengths =
      impl_->states.metadata(entry.stateLane).lengths;
  if (lengths.targetTokens != restoredPrefixLength ||
      (restoreDraftState &&
       !lengths.hasCompleteDraftWindow(kDraftCacheStride)) ||
      (!restoreDraftState && lengths.draftLength != 0)) {
    throw std::invalid_argument("prefix logical length does not match state");
  }
  // Activation left out the images ModelRequest::restoredTokens covers; a
  // restore further in releases the rest here: warmup and direct callers
  // activate with 0, as does an engine start that let its cache lease go and
  // found a state when it looked up again. Their spans stay because rotary
  // positions after them depend on their grids.
  for (Impl::ImageState &image : entry.images) {
    if (image.span.end() <= restoredPrefixLength) {
      image.rows.reset();
    }
  }
  entry.promptComplete = false;
  if (!entry.replayingGeneration) {
    entry.finalTargetHidden.clear();
    entry.pendingToken.reset();
  }
  entry.draftContextPlan.reset();
}

void Runtime::setDraftContextPlan(uint64_t requestId, DraftContextPlan plan) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || plan.replayEnd != entry.promptTokens) {
    throw std::invalid_argument("draft context plan does not match request");
  }
  const uint64_t current =
      impl_->states.metadata(entry.stateLane).lengths.targetTokens;
  if (plan.replayBegin != current)
    throw std::invalid_argument("draft context plan restore boundary is stale");
  entry.draftContextPlan = std::move(plan);
}

std::vector<ModelStepResult>
Runtime::prefill(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
  return prefillAsync(plan, items, {})->wait();
}

std::unique_ptr<ModelBatchTicket>
Runtime::submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
                std::function<void()> completion) {
  switch (plan.kind) {
  case WorkKind::Prefill:
    return prefillAsync(plan, items, std::move(completion));
  case WorkKind::Decode:
    return decodeAsync(plan, items, std::move(completion));
  }
  throw std::logic_error("unknown model work kind");
}

std::unique_ptr<ModelBatchTicket>
Runtime::prefillAsync(const BatchPlan &plan,
                      std::span<const ModelBatchItem> items,
                      std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Prefill);
  if (plan.decodeStage != DecodeStage::Regular) {
    throw std::invalid_argument("Qwen prefill cannot resume a mask plan");
  }
  impl_->retireAhead();  // SPLASH_DRAFT_AHEAD: prefill writes lane buffers
  impl_->flushGdn();     // SPLASH_GDN_DEFER

  std::array<Impl::Request *, kLaneCount> entries{};
  // Each request's draws before the chunk, which a rerun of it restores.
  std::array<uint64_t, kLaneCount> draws{};
  CommandGraph graph;
  if (impl_->aneFfn) {
    for (uint32_t lane = 0; lane < items.size(); ++lane)
      draws[lane] = impl_->request(items[lane].requestId).rngCounter;
    impl_->aneFfn->begin();
  }
  const auto captures = impl_->encodeRaggedPrefillGraph(graph, items, entries);
  const bool encodesImages = std::any_of(
      entries.begin(), entries.begin() + items.size(), [](const auto *entry) {
        return std::any_of(entry->images.begin(), entry->images.end(),
                           [](const auto &image) {
                             return image.rows && image.rows->encoding;
                           });
      });
  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  CommandTicket command =
      impl_->aneFfn ? impl_->aneFfn->commit(graph, std::move(completion))
                    : impl_->backend.submitCommandAsync(graph.command(), std::move(completion));
  Impl *impl = impl_.get();
  auto finish = [impl, entries, captures, draws,
                 items = std::move(copiedItems)](CommandTiming &timing) mutable {
    // A chunk whose Neural Engine work failed holds no usable outputs, and
    // the split has stopped: the GPU runs the chunk again alone, which
    // computes what it would have the first time. Nothing of the chunk is
    // committed before the code below: state parity, lengths, the selected
    // token and the image rows' state. The rerun writes the same KV and draft
    // rows; clearForColdStart clears the current state again, which neither
    // command writes; synchronizedPageTable writes nothing at the same
    // revision; image rows the first command encoded are still `encoding` and
    // are copied, not encoded again; and the requests' draws restored, a
    // sampled first token draws the same uniform. The first ticket was
    // released before this completion runs (DeferredMetalTicket::wait), so
    // the backend takes the rerun's command, whose time counts in the
    // chunk's.
    if (impl->aneFfn && !impl->aneFfn->finish()) {
      for (uint32_t lane = 0; lane < items.size(); ++lane)
        entries[lane]->rngCounter = draws[lane];
      CommandGraph again;
      static_cast<void>(impl->encodeRaggedPrefillGraph(again, items, entries));
      const CommandTiming rerun = impl->backend.submitCommandAsync(again.command()).wait();
      timing.gpuSeconds += rerun.gpuSeconds;
      timing.wallSeconds += rerun.wallSeconds;
      ++impl->counters.aneFfnReruns;
    }
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      const uint64_t chunkEnd = items[lane].logicalPosition + items[lane].tokenCount;
      for (Impl::ImageState &image : entries[lane]->images) {
        if (!image.rows)
          continue;
        Impl::ImageRows &rows = *image.rows;
        if (rows.encoding) {
          rows.encoding = false;
          rows.encoded = true;
          rows.pixels = MetalBuffer{};
        }
        // Its last row is injected: the cache owns the rows from now on, so
        // reclaim can free them while the request decodes.
        if (image.span.end() <= chunkEnd) {
          impl->retain(image.rows);
          image.rows.reset();
        }
      }
    }

    std::vector<ModelStepResult> results;
    results.reserve(items.size());
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      Impl::Request &entry = *entries[lane];
      const ModelBatchItem &item = items[lane];
      const uint64_t nextLength = item.logicalPosition + item.tokenCount;
      // The anchor this chunk selects when it completes a generation prompt.
      std::optional<uint32_t> selected;
      if (nextLength == entry.promptTokens && !entry.replayingGeneration &&
          entry.scoreTokens.empty() && entry.constraint == ConstraintMode::None) {
        selected = impl->initialToken(lane);
        if (std::string failure = impl->invalidSelection({&*selected, 1});
            !failure.empty()) {
          // The chunk's state is not committed; the engine ends the
          // request.
          results.push_back({.requestId = entry.id,
                             .consumedPromptTokens = item.tokenCount,
                             .failure = std::move(failure)});
          continue;
        }
      }
      impl->states.swapParity(entry.stateLane);
      QwenLogicalLengths lengths = impl->states.metadata(entry.stateLane).lengths;
      lengths.targetTokens = nextLength;
      for (const DispatchDraftCaptureSpan &capture : captures[lane]) {
        lengths = Impl::advanceDraftContext(lengths, nextLength,
                                            capture.absoluteBegin,
                                            capture.absoluteEnd,
                                            capture.resetDraftState);
        impl->counters.draftContextRowsActive += capture.activeRows;
        impl->counters.draftContextRowsMaterialization +=
            capture.materializationRows;
        if (capture.resetDraftState)
          ++impl->counters.draftStateResets;
      }
      impl->counters.targetPrefillRows += item.tokenCount;
      impl->counters.draftContextRowsAvoided +=
          item.tokenCount - Impl::captureRows(captures[lane]);
      impl->states.updateLengths(entry.stateLane, lengths);
      entry.promptComplete = nextLength == entry.promptTokens;
      ModelStepResult result{entry.id, item.tokenCount, {}, false,
                             DecodeStage::Regular, 0, 0};
      if (entry.promptComplete && !entry.replayingGeneration) {
        entry.pendingToken.reset();
        if (!entry.scoreTokens.empty()) {
          // Score-only: read the raw fp32 logits at the final prompt position
          // (the lane's logits row 0) in requested order.
          const float *row = contents<float>(
              impl->decodeArena->get(lane, DecodeTensor::Logits),
              "score logits");
          result.scoreLogits.reserve(entry.scoreTokens.size());
          for (uint32_t token : entry.scoreTokens) {
            const float logit = row[token];
            if (!std::isfinite(logit)) {
              // A numerical outcome for this request, not a broken invariant:
              // report it as a lane failure so the engine drops this request
              // before cache publication or output and the batch survives.
              result.scoreLogits.clear();
              result.failure = "score logit is not finite";
              break;
            }
            result.scoreLogits.push_back(logit);
          }
          result.finished = true;
        } else if (selected) {
          impl->commitSelected(entry, {&*selected, 1});
          impl->emitTerminalAnchor(entry, result);
        } else {
          // The first token waits for the request's first mask. A replay
          // never gets here: it keeps its stage, and a request that holds
          // its mask asks for none.
          impl->captureFinalHidden(entry, lane);
          entry.decodeStage = DecodeStage::ApplyInitialMask;
          result.nextDecodeStage = DecodeStage::ApplyInitialMask;
        }
      }
      if (entry.promptComplete)
        entry.replayingGeneration = false;
      results.push_back(std::move(result));
    }
    impl->counters.lastPrefillWallSeconds = timing.wallSeconds;
    impl->counters.totalPrefillWallSeconds += timing.wallSeconds;
    impl->counters.lastPrefillGpuSeconds = timing.gpuSeconds;
    impl->counters.totalPrefillGpuSeconds += timing.gpuSeconds;
    return results;
  };
  return std::make_unique<DeferredMetalTicket>(
      std::move(command), std::move(finish), !encodesImages);
}

std::vector<ModelStepResult>
Runtime::decode(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
  return decodeAsync(plan, items, {})->wait();
}

std::unique_ptr<ModelBatchTicket>
Runtime::decodeAsync(const BatchPlan &plan,
                     std::span<const ModelBatchItem> items,
                     std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Decode);
  const bool constrained = plan.constrained;
  // SPLASH_DRAFT_AHEAD: a block built for other lanes, positions, anchors,
  // stages or head rows finishes before any lane buffer is written.
  if (impl_->ahead && !impl_->aheadMayMatch(plan, items))
    impl_->retireAhead();
  if (plan.decodeStage != DecodeStage::Regular) {
    if (!constrained) {
      throw std::invalid_argument(
          "only constrained decode uses a specialized decode stage");
    }
    return impl_->submitInitialSelection(items, std::move(completion));
  }

  const uint32_t width = static_cast<uint32_t>(items.size());
  std::vector<Impl::DecodeLaneResult> lanes(width);
  std::array<Impl::Request *, kLaneCount> requests{};
  std::array<uint64_t, kLaneCount> logicalPositions{};
  std::array<uint32_t, kLaneCount> maximumRetained{};
  for (uint32_t lane = 0; lane < width; ++lane) {
    const ModelBatchItem &item = items[lane];
    Impl::Request &entry = impl_->request(item.requestId);
    if ((entry.constraint == ConstraintMode::TokenMask) != constrained) {
      throw std::invalid_argument(
          "request does not belong to the batch's constraint mode");
    }
    if (entry.decodeStage != DecodeStage::Regular) {
      throw std::logic_error("request decode stage does not match decode plan");
    }
    if (!entry.pendingToken)
      throw std::logic_error("decode request has no current anchor");
    const uint32_t remaining = entry.maxNewTokens - entry.generatedTokens;
    if (!remaining)
      throw std::logic_error("completed request was decoded");
    if (isStopToken(impl_->geometry, *entry.pendingToken) || remaining == 1) {
      throw std::logic_error("terminal anchor was not emitted on selection");
    }

    if (constrained && !entry.maskWords.empty())
      throw std::logic_error("constrained request has stale mask state");
    if (Impl::samplingEnabled(entry))
      Impl::stageSamplingCycle(entry);

    // DFlash has one physical graph: anchor + seven proposal rows. A shorter
    // output budget only lowers the token-exact commit count; it never
    // changes the Metal graph shape.
    Impl::DecodeLaneResult &laneResult = lanes[lane];
    laneResult.request = &entry;
    laneResult.currentAnchor = *entry.pendingToken;
    laneResult.maximumRetained = std::min(remaining, kDecodeRows);

    requests[lane] = &entry;
    logicalPositions[lane] = item.logicalPosition;
    maximumRetained[lane] = laneResult.maximumRetained;
  }

  // SPLASH_PROMPT_LOOKUP: a single request whose context repeats verifies the
  // lookup's proposals in place of the draft's (constrained ones too: the
  // grammar simulation reads the proposed tokens either way), over 16 or 32
  // rows with SPLASH_WIDE_PROMPT_LOOKUP. Chosen before any lane write, so an
  // adoptable draft-ahead block never races a host write.
  const auto wide = width == 1 ? impl_->lookup16Proposal(*requests[0], items[0]) : std::nullopt;
  const auto lookup = width == 1 && !wide ? impl_->promptLookupProposal(*requests[0]) : std::nullopt;
  lanes[0].promptLookup = wide || lookup;

  // SPLASH_DRAFT_AHEAD: adopt the block for a drafter cycle when it drew this
  // cycle's uniforms; it then wrote the draft inputs, uniforms and proposals
  // on the GPU, and the host writes none of them.
  bool adopt = false;
  if (impl_->ahead) {
    adopt = !lanes[0].promptLookup;
    for (uint32_t lane = 0; adopt && lane < width; ++lane)
      adopt = !Impl::samplingEnabled(*requests[lane]) ||
              requests[lane]->cycleUniforms == impl_->ahead->uniforms[lane];
    if (adopt) {
      impl_->ahead.reset();
      for (uint32_t lane = 0; lane < width; ++lane)
        ++requests[lane]->aheadAdopted;
    } else {
      impl_->retireAhead();
    }
  }
  for (uint32_t lane = 0; lane < width; ++lane) {
    if (!adopt && Impl::samplingEnabled(*requests[lane]))
      impl_->uploadSamplingUniforms(*requests[lane], lane);
    impl_->prepareDecodeLane(*requests[lane], items[lane], lane, !adopt);
  }

  const std::span<Impl::Request *const> entries(requests.data(), width);
  if (wide) {
    impl_->prepareLookup16(*requests[0], items[0], *wide);
    lanes[0].wideLookup = true;
    lanes[0].wideTiles = wide->tiles;
    lanes[0].maximumRetained = maximumRetained[0] = std::min(
        requests[0]->maxNewTokens - requests[0]->generatedTokens, wide->tiles * kDecodeRows);
  } else if (lookup) {
    impl_->preparePromptLookup(*lookup);
  }
  const uint32_t physicalWidth = wide ? wide->tiles : width;
  const uint32_t ropeRows = physicalWidth * kDecodeRows;
  // SPLASH_GDN_DEFER: decided before the graph; a pending commit this step
  // does not take (another request's, or this one's on another route) lands
  // first.
  const bool gdnDefer = impl_->gdnDeferAllowed(width, wide.has_value());
  if (impl_->gdnPending && !(gdnDefer && impl_->gdnPending->requestId == requests[0]->id))
    impl_->flushGdn();
  for (uint32_t lane = 0; lane < width; ++lane)
    requests[lane]->gdnDeferCycle = gdnDefer;
  CommandGraph commandGraph;
  const Impl::StreamedHeadGuard streamGuard{impl_->backend};
  if (const size_t chunk = Impl::streamChunk(); chunk && !constrained)
    commandGraph.streamAt(chunk, [impl = impl_.get()](const metal::Command &head) {
      static_cast<void>(impl->backend.streamHead(head));
    });
  // An adopted block built the draft tables already.
  impl_->addRopeTables(
      commandGraph,
      impl_->decodeArena->batchSlice(DecodeTensor::Positions, physicalWidth), ropeRows,
      impl_->decodeArena->batchSlice(DecodeTensor::DraftPositions, physicalWidth),
      adopt ? 0 : ropeRows, impl_->decodeArena->batchSlice(DecodeTensor::RopeCos, physicalWidth),
      impl_->decodeArena->batchSlice(DecodeTensor::RopeSin, physicalWidth),
      impl_->decodeArena->batchSlice(DecodeTensor::DraftRopeCos, physicalWidth),
      impl_->decodeArena->batchSlice(DecodeTensor::DraftRopeSin, physicalWidth));
  if (!lanes[0].promptLookup && !adopt) {
    impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::DraftInputTokens,
                                DecodeTensor::DraftHidden0, width);
    impl_->encodeDraftBatchGraph(commandGraph, entries,
                                 {logicalPositions.data(), width});
  }
  if (constrained) {
    return std::make_unique<Impl::ConstrainedDecodeTicket>(
        *impl_, std::move(lanes), items, commandGraph, adopt,
        std::move(completion));
  }
  if (wide) {
    impl_->encodeLookup16Forward(commandGraph, *requests[0], items[0]);
    impl_->encodeLookup16Commit(commandGraph, *requests[0], items[0], maximumRetained[0]);
  } else {
    impl_->encodeBatchVerifyInput(commandGraph, width);
    impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::InputTokens,
                                DecodeTensor::Hidden0, width);
    impl_->encodeTargetVerifyBatchForward(commandGraph, entries, items);
    impl_->encodeTargetVerifyBatchPolicy(commandGraph, entries);
    impl_->encodeBatchAcceptance(commandGraph, entries,
                                 {maximumRetained.data(), width});
    impl_->encodeBatchGdnCommit(commandGraph, entries);
    impl_->encodeDraftStateCommitBatch(commandGraph, entries, items);
  }

  // Read before `finish` takes the lanes.
  const bool launchAhead = impl_->aheadLaunchAllowed(items, false, lanes[0].promptLookup);
  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  Impl *impl = impl_.get();
  auto finish = [impl, lanes = std::move(lanes),
                 items = std::move(copiedItems)](CommandTiming timing) mutable {
    return impl->finalizeDecode(lanes, items, timing);
  };
  CommandTicket command = impl_->submitWithAhead(
      commandGraph.command(), std::move(completion), entries, items, launchAhead);
  return std::make_unique<DeferredMetalTicket>(std::move(command),
                                               std::move(finish));
}

uint32_t Runtime::residentLane(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident)
    throw std::logic_error("request is not resident");
  return entry.stateLane;
}

void Runtime::settleState() { impl_->flushGdn(); }

std::shared_ptr<const CompositeState> Runtime::snapshot(uint64_t requestId) {
  if (impl_->gdnPending && impl_->gdnPending->requestId == requestId)
    impl_->flushGdn();  // SPLASH_GDN_DEFER
  std::shared_ptr<const CompositeState> state =
      impl_->states.snapshot(residentLane(requestId));
  if (!state)
    return state;
  return impl_->holdStraddledRows(impl_->request(requestId), std::move(state));
}

uint64_t Runtime::snapshotBytes() const noexcept {
  return impl_->states.layout().cachedBytes();
}

bool Runtime::canSnapshotToDisk() const noexcept {
  return impl_->states.canSnapshotToDisk();
}

std::unique_ptr<StateOffload>
Runtime::snapshotToDisk(uint64_t requestId, std::function<void()> completion) {
  if (impl_->gdnPending && impl_->gdnPending->requestId == requestId)
    impl_->flushGdn();  // SPLASH_GDN_DEFER
  return impl_->states.snapshotToDisk(residentLane(requestId), std::move(completion));
}

uint32_t Runtime::statesToActivate() const noexcept {
  return impl_->states.statesToActivate();
}

uint64_t Runtime::reclaimIdleState(bool keepLane, IdleMemory scope) noexcept {
  // One unit per call, so a denied allocation frees only what it needs;
  // rebuildable caches go once the pool has nothing more to give.
  if (const uint64_t buffer = impl_->states.releaseOneIdle(keepLane))
    return buffer;
  return scope == IdleMemory::BuffersThenCaches ? impl_->releaseOneCache() : 0;
}

std::optional<std::string>
Runtime::provideMask(uint64_t requestId, std::span<const uint32_t> words) {
  Impl::Request &entry = impl_->request(requestId);
  const bool acceptsMask =
      waitsForMask(entry.decodeStage) || entry.verifyMaskInFlight;
  // Initial-mask replies can race resource preemption. They belong to the
  // host continuation, not the released device state.
  if (entry.constraint != ConstraintMode::TokenMask || !acceptsMask ||
      !entry.maskWords.empty()) {
    throw std::logic_error("request is not waiting for a token mask");
  }
  const uint32_t maskWords = impl_->geometry.maskWords();
  uint64_t expected = entry.verifyMaskInFlight
                          ? uint64_t{entry.verifyRows + 1} * maskWords
                          : maskWords;
  // The native loop matches each response's word count to its request.
  if (words.size() != expected) {
    throw std::logic_error("token mask has the wrong word count");
  }
  const uint32_t rows = static_cast<uint32_t>(words.size() / maskWords);
  for (uint32_t row = 0; row < rows; ++row) {
    auto begin = words.begin() + uint64_t{row} * maskWords;
    if (std::none_of(begin, begin + maskWords,
                     [](uint32_t word) { return word != 0; })) {
      return "token mask row permits no vocabulary token";
    }
  }
  if (entry.verifyMaskInFlight) {
    if (!entry.pendingToken || (words[*entry.pendingToken / 32] &
                                (1U << (*entry.pendingToken % 32))) == 0) {
      return "verify mask is not synchronized to the pending anchor";
    }
  }
  entry.maskWords.assign(words.begin(), words.end());
  return std::nullopt;
}

void Runtime::end(uint64_t requestId) {
  auto found = impl_->requests.find(requestId);
  if (found == impl_->requests.end())
    return;
  impl_->retireAheadFor(requestId);  // SPLASH_DRAFT_AHEAD: its ring is released
  if (impl_->gdnPending && impl_->gdnPending->requestId == requestId)
    impl_->gdnPending.reset();  // SPLASH_GDN_DEFER: the lane's state goes
  if (impl_->requestStats && impl_->draftAheadOn())
    std::fprintf(stderr, "draft_ahead request=%llu launched=%llu adopted=%llu\n",
                 static_cast<unsigned long long>(requestId),
                 static_cast<unsigned long long>(found->second.aheadLaunched),
                 static_cast<unsigned long long>(found->second.aheadAdopted));
  if (const Impl::Request &entry = found->second; impl_->requestStats && !entry.lookupHistory.empty())
    std::fprintf(stderr, "prompt_lookup request=%llu cycles=%llu hits=%llu retained=%llu\n",
                 static_cast<unsigned long long>(requestId),
                 static_cast<unsigned long long>(entry.lookupCycles),
                 static_cast<unsigned long long>(entry.lookupHits),
                 static_cast<unsigned long long>(entry.lookupRetained));
  if (const Impl::Request &entry = found->second;
      impl_->requestStats && !entry.lookupHistory.empty() && impl_->wideLookupEnabled)
    std::fprintf(stderr,
                 "wide_lookup request=%llu hits=%llu retained=%llu wide32=%d hits32=%llu gdn_single=%d "
                 "gdn_cycles=%llu\n",
                 static_cast<unsigned long long>(requestId),
                 static_cast<unsigned long long>(entry.wideLookupHits),
                 static_cast<unsigned long long>(entry.wideLookupRetained), entry.wide32 ? 1 : 0,
                 static_cast<unsigned long long>(entry.wide32Hits), static_cast<int>(entry.wideGdn),
                 static_cast<unsigned long long>(entry.wideGdnCycles));
  impl_->releaseImages(found->second);
  if (found->second.resident) {
    impl_->states.releaseLane(found->second.stateLane, requestId);
    impl_->pageTableBindings[found->second.stateLane] = {};
  }
  impl_->requests.erase(found);
}

namespace {

// A warmup step whose lane failed (a non-finite logit row) fails the warmup
// there, with the lane's reason.
void requireLanesSucceeded(std::span<const ModelStepResult> results) {
  for (const ModelStepResult &result : results)
    if (!result.failure.empty())
      throw std::runtime_error(result.failure);
}

// Warmup runs on the startup runway the engine's KV pool allocated
// (ExecutionLimits::warmupKvPages); it never allocates KV.
void requireRunwayPages(const kv::PageStorage &storage,
                        std::span<const uint32_t> pages) {
  for (uint32_t page : pages) {
    if (page >= ExecutionLimits::warmupKvPages || !storage.isAllocated(page)) {
      throw std::logic_error("warmup KV page " + std::to_string(page) +
                             " is outside the startup runway");
    }
  }
}

// A warmup request's batch item. Each warmup residency keeps one page list,
// so its revision stays 1.
ModelBatchItem warmupItem(uint64_t id, uint64_t position, uint32_t tokens,
                          std::span<const uint32_t> pages) {
  return {.requestId = id,
          .logicalPosition = position,
          .tokenCount = tokens,
          .pageTable = pages,
          .pageTableRevision = 1};
}

} // namespace

void Runtime::prepareWarmupDecode(uint64_t requestId, uint32_t anchor) {
  // Teacher-force a valid input so EOS selected by synthetic prefill cannot
  // prevent the warmup from exercising the real draft/verify/commit graph.
  while (anchor < impl_->geometry.target.vocabularySize &&
         isStopToken(impl_->geometry, anchor))
    ++anchor;
  if (anchor >= impl_->geometry.target.vocabularySize)
    throw std::logic_error("decode warmup has no non-terminal input token");
  auto &entry = impl_->request(requestId);
  entry.pendingToken = anchor;
  entry.generatedTokens = 0;
}

WarmupStepResult Runtime::warmupPrefill(uint32_t rows) {
  if (!rows || rows > kPrefillRows)
    throw std::invalid_argument("invalid prefill warmup row count");
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 100;
  double wallSeconds = 0.0;
  std::vector<WarmupLaneResult> lanes;
  std::vector<uint32_t> warmupPrompt(rows, 0);
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 16;
  beginColdRequest(request, 0);
  try {
    std::vector<uint32_t> pages((rows + kv::kPageTokens - 1) / kv::kPageTokens);
    std::iota(pages.begin(), pages.end(), 0u);
    requireRunwayPages(impl_->kvPages, pages);
    BatchPlan plan{.kind = WorkKind::Prefill,
                   .items = {{id, rows}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item = warmupItem(id, 0, rows, pages);
    item.inputTokens = request.prompt;
    const auto phaseStart = AwakeClock::now();
    auto result = prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    wallSeconds = std::chrono::duration<double>(AwakeClock::now() - phaseStart).count();
    requireLanesSucceeded(result);
    if (result.size() != 1 || result[0].consumedPromptTokens != rows) {
      throw std::runtime_error("prefill warmup result mismatch");
    }
    lanes.push_back({std::move(result[0]), impl_->request(id).pendingToken,
                     impl_->states.metadata(0).lengths.targetTokens});
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  return {"real " + std::to_string(rows) +
              "-row KV target+draft prefill [M32]",
          wallSeconds, std::move(lanes)};
}

WarmupStepResult Runtime::warmupDecodeBatch(uint32_t width) {
  if (!width || width > kLaneCount) {
    throw std::invalid_argument("invalid decode warmup width");
  }
  constexpr uint64_t firstId = std::numeric_limits<uint64_t>::max() - 110;
  // Plan order is deliberately unrelated to state-lane order. DecodeArena
  // lanes follow the explicit BatchPlan, while recurrent and KV state stay
  // addressed by each request's state lane; batching must never assume lanes
  // 0..3.
  constexpr std::array<uint32_t, kLaneCount> stateLaneOrder{2, 0, 3, 1};
  double wallSeconds = 0.0;
  std::vector<WarmupLaneResult> lanes;
  std::array<std::vector<uint32_t>, kLaneCount> pages;
  try {
    for (uint32_t lane = 0; lane < width; ++lane) {
      std::vector<uint32_t> warmupPrompt{lane};
      ModelRequest request;
      request.id = firstId + lane;
      request.prompt = warmupPrompt;
      request.maxNewTokens = 16;
      beginColdRequest(request, stateLaneOrder[lane]);
      pages[lane] = {5 + lane};
      requireRunwayPages(impl_->kvPages, pages[lane]);
      BatchPlan prefillPlan{.kind = WorkKind::Prefill,
                            .items = {{request.id, 1}},
                            .decodeStage = DecodeStage::Regular};
      ModelBatchItem item = warmupItem(request.id, 0, 1, pages[lane]);
      item.inputTokens = request.prompt;
      requireLanesSucceeded(
          prefill(prefillPlan, std::span<const ModelBatchItem>(&item, 1)));
      prepareWarmupDecode(request.id, warmupPrompt.back());
    }
    BatchPlan plan;
    plan.kind = WorkKind::Decode;
    std::vector<ModelBatchItem> items;
    for (uint32_t lane = 0; lane < width; ++lane) {
      plan.items.push_back({firstId + lane, 0});
      items.push_back(warmupItem(firstId + lane, 1, 0, pages[lane]));
    }
    const auto phaseStart = AwakeClock::now();
    auto decoded = decode(plan, items);
    wallSeconds = std::chrono::duration<double>(AwakeClock::now() - phaseStart).count();
    requireLanesSucceeded(decoded);
    bool committedEveryLane = decoded.size() == width;
    for (uint32_t lane = 0; committedEveryLane && lane < width; ++lane) {
      const auto &lengths = impl_->states.metadata(stateLaneOrder[lane]).lengths;
      committedEveryLane = !decoded[lane].outputTokens.empty() &&
                           lengths.targetTokens > 1 &&
                           lengths.targetTokens ==
                               1 + decoded[lane].outputTokens.size() -
                                   decoded[lane].outputTokensWithoutKv &&
                           lengths.hasCompleteDraftWindow(kDraftCacheStride);
    }
    if (!committedEveryLane) {
      throw std::runtime_error(
          "decode warmup B" + std::to_string(width) +
          " mismatch [results=" + std::to_string(decoded.size()) + "]");
    }
    for (uint32_t lane = 0; lane < width; ++lane) {
      lanes.push_back({std::move(decoded[lane]),
                       impl_->request(firstId + lane).pendingToken,
                       impl_->states.metadata(stateLaneOrder[lane]).lengths.targetTokens});
      end(firstId + lane);
    }
  } catch (...) {
    for (uint32_t lane = 0; lane < width; ++lane)
      end(firstId + lane);
    throw;
  }
  return {"real B" + std::to_string(width) + " draft/verify/commit decode",
          wallSeconds, std::move(lanes)};
}

WarmupStepResult Runtime::warmupCompositeStateRestore() {
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 121;
  constexpr uint32_t prefixTokens = 2 * kv::kPageTokens;
  constexpr uint32_t suffixTokens = kDecodeRows;
  constexpr uint32_t promptTokens = prefixTokens + suffixTokens;
  std::vector<uint32_t> warmupPrompt(promptTokens, 2);
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 8;
  std::shared_ptr<const CompositeState> cachedState;
  double wallSeconds = 0.0;
  beginColdRequest(request, 0);
  try {
    // Deliberately non-contiguous physical ids exercise page-table lookup.
    const std::vector<uint32_t> pages{12, 10, 11};
    requireRunwayPages(impl_->kvPages, pages);
    BatchPlan plan{.kind = WorkKind::Prefill,
                   .items = {{id, prefixTokens}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item = warmupItem(id, 0, prefixTokens, pages);
    item.inputTokens =
        std::span<const uint32_t>(request.prompt).first(prefixTokens);
    static_cast<void>(prefill(plan, std::span<const ModelBatchItem>(&item, 1)));
    wallSeconds = impl_->counters.lastPrefillWallSeconds;
    cachedState = snapshot(id);
    if (!cachedState)
      throw metal::MetalAllocationError("prefix warmup state allocation failed");
    end(id);
    beginColdRequest(request, 1);
    if (beginRestore(id, prefixTokens, cachedState, true, {}))
      throw std::logic_error("a resident state restore returned a read");
    setDraftContextPlan(id, planDraftContext(prefixTokens, promptTokens, {}));
    const auto &restored = impl_->states.metadata(1).lengths;
    if (restored.targetTokens != prefixTokens ||
        !restored.hasCompleteDraftWindow(kDraftCacheStride)) {
      throw std::runtime_error("prefix restore length mismatch");
    }

    // Continue from committed KV history. This M8 command teacher-forces a
    // new chunk, then the real speculative cycle overwrites its speculative
    // page suffix and advances only the accepted commit length.
    BatchPlan suffixPlan{.kind = WorkKind::Prefill,
                         .items = {{id, suffixTokens}},
                         .decodeStage = DecodeStage::Regular};
    ModelBatchItem suffix = warmupItem(id, prefixTokens, suffixTokens, pages);
    suffix.inputTokens = std::span<const uint32_t>(request.prompt)
                             .subspan(prefixTokens, suffixTokens);
    requireLanesSucceeded(
        prefill(suffixPlan, std::span<const ModelBatchItem>(&suffix, 1)));
    prepareWarmupDecode(id, warmupPrompt.back());
    const double continuationWallSeconds =
        impl_->counters.lastPrefillWallSeconds;
    wallSeconds += continuationWallSeconds;
    BatchPlan decodePlan{.kind = WorkKind::Decode,
                         .items = {{id, 0}},
                         .decodeStage = DecodeStage::Regular};
    ModelBatchItem decodeItem = warmupItem(id, promptTokens, 0, pages);
    auto decoded =
        decode(decodePlan, std::span<const ModelBatchItem>(&decodeItem, 1));
    requireLanesSucceeded(decoded);
    const double historicalDecodeWallSeconds =
        impl_->counters.lastDecodeWallSeconds;
    wallSeconds += historicalDecodeWallSeconds;
    const auto &continued = impl_->states.metadata(1).lengths;
    if (decoded.size() != 1 || decoded[0].outputTokens.empty() ||
        !continued.hasCompleteDraftWindow(kDraftCacheStride) ||
        continued.targetTokens <= promptTokens ||
        continued.targetTokens !=
            promptTokens + decoded[0].outputTokens.size() -
                decoded[0].outputTokensWithoutKv) {
      throw std::runtime_error(
          "restored historical prefix did not continue exactly");
    }
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  return {"real paged-KV state restore, arbitrary page table, lane move, "
          "bounded restore continuation, and decode",
          wallSeconds, {}};
}

ModelMemoryActual Runtime::actualRuntimeMemory() const {
  return {impl_->states.actualAllocatedBytes(), impl_->prefillArena->bytes(),
          impl_->decodeArena->bytes(), impl_->states.stagingBytes(),
          impl_->draftModel.restrictedHeadAllocatedBytes()};
}

ModelTelemetry Runtime::telemetry() const noexcept {
  ModelTelemetry result = impl_->counters;
  result.stateAllocatedBytes = impl_->states.actualAllocatedBytes();
  result.idleGdnCells = impl_->states.idleCells();
  result.idleDraftRings = impl_->states.idleRings();
  result.visionArenaBytes = impl_->vision ? impl_->vision->arenaBytes() : 0;
  result.embeddingCacheBytes = impl_->embeddingCacheBytes;
  result.stateHeldImageBytes = impl_->heldRowsBytes(false);
  for (const auto &[_, held] : impl_->imageRows) {
    if (const std::shared_ptr<const Impl::ImageRows> rows = held.lock())
      result.imageRowsBytes += rows->pixels.sizeBytes() + rows->embeddings.sizeBytes();
  }
  return result;
}

std::vector<ops::SwiGluProjections> aneFfnLayers(const LoadedModel &model) {
  std::vector<ops::SwiGluProjections> layers;
  if (const auto *dense = std::get_if<Qwen3_8Weights>(&model.target))
    for (const Qwen3_8LayerWeights &layer : dense->layers)
      layers.push_back({&layer.gateProjection, &layer.upProjection, &layer.downProjection});
  return layers;
}

void withPrefillArena(
    MetalBackend &backend, const LoadedModel &model, const ops::ExecutionPlans &operators, kv::Format format,
    const std::function<void(const ops::PrefillFfnBuffers &, const std::array<MetalBuffer, 2> &)> &use) {
  const PrefillArena arena(backend, RuntimeGeometry::from(model, format), operators);
  const QwenTargetPrefillBuffers buffers = prefillBuffers(arena);
  use(buffers.ffn(), buffers.hidden);
}

ModelMemoryPlan plannedRuntimeMemory(const LoadedModel &model,
                                     const ops::ExecutionPlans &operators,
                                     kv::Format format) {
  requireCompatibleModel(model);
  const RuntimeGeometry geometry = RuntimeGeometry::from(model, format);
  return {model.stateLayout().laneBytes(),
          plannedPrefillBytes(geometry, operators),
          plannedDecodeBytes(geometry, operators)};
}

std::unique_ptr<RuntimeModel> createRuntime(RuntimeContext context) {
  return std::make_unique<Runtime>(std::move(context));
}

} // namespace splash::model
