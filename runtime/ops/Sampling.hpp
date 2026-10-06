// Modified by meowkernels.
#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Sampling.h"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <optional>
#include <span>

namespace splash::ops {

// The sampling penalties, which rewrite a lane's target logits before its
// policy selects from them: repetition scales the logit of every token the
// prompt or the output holds, presence and frequency lower that of every
// token the output holds. The defaults leave the logits unchanged.
struct SamplingPenalties final {
  float repetition = 1.0F;
  float presence = 0.0F;
  float frequency = 0.0F;

  [[nodiscard]] bool active() const noexcept {
    return repetition != 1.0F || presence != 0.0F || frequency != 0.0F;
  }
};

// One lane's target policy. The defaults are a request's: greedy, and a
// sampling lane keeps every token.
struct SamplingPolicy final {
  // A sampling lane keeps the tokens minP leaves it, then its topK most
  // likely of those, or every one for 0 or a topK past the vocabulary (top-k
  // disabled), then its top-p nucleus of those. Greedy lanes (temperature 0)
  // read none of the three.
  uint32_t topK = 0;
  float temperature = 0.0F;
  float topP = 1.0F;
  bool constrained = false;
  // The lane ignores end-of-sequence: the target never selects a stop token,
  // though the draft may still propose one.
  bool excludesStopTokens = false;
  // Greedy lanes take the argmax of the penalized logits.
  SamplingPenalties penalties{};
  // A sampling lane first drops the tokens less likely than minP times its
  // most likely one; 0 drops none.
  float minP = 0.0F;

  [[nodiscard]] bool samples() const noexcept { return temperature > 0.0F; }
};

// The penalty words of every state lane, one row of vocabulary words each,
// and the row each lane of a dispatch reads. Dispatch lanes follow the batch
// plan, not state lanes, so a lane never binds a row by its own index.
struct PenaltyTable final {
  metal::MetalBuffer words;
  std::span<const uint32_t> rows;
};

struct SamplingWorkspace final {
  uint64_t argmaxValuesBytes = 0;
  uint64_t argmaxIndicesBytes = 0;
  uint64_t partialMassesBytes = 0;
  uint64_t vocabularyRowsBytes = 0;
  uint64_t vocabularyRangesBytes = 0;
  uint64_t vocabularyArrivalsBytes = 0;
  // SPLASH_BLOCK_VERIFY: TargetCandidateRow per row.
  uint64_t candidateRowsBytes = 0;
};

struct SamplingBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  // Per shard of a sampled row, its share of the row's softmax denominator
  // (metal/abi/Sampling.h TargetShardMass).
  metal::MetalBuffer partialMasses;
  // Per sampled row, where its distribution ends and, for a drafted row,
  // its draft token's probability (TargetVocabularyRow), which acceptance
  // reads.
  metal::MetalBuffer vocabularyRows;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer constraintMasks;
  metal::MetalBuffer outputTokens;
  metal::MetalBuffer argmaxValues;
  metal::MetalBuffer argmaxIndices;
  // Verify input tokens [rows]: row 0 of a lane is its anchor, rows 1..7 its
  // draft tokens, which penalized and sampled verify rows read.
  metal::MetalBuffer inputTokens;
  // The draft's candidates and their probabilities at each proposal
  // position (AcceptanceBuffers), which a sampled verify row draws its
  // correction's residual from.
  metal::MetalBuffer draftCandidates;
  metal::MetalBuffer draftProbabilities;
  // Per sampled row, the ranges of its draw (TargetVocabularyRange), which
  // the groups that share the draw sum, and how many of those groups have
  // finished: a count every draw returns to zero, where it starts.
  metal::MetalBuffer vocabularyRanges;
  metal::MetalBuffer vocabularyArrivals;
  // SPLASH_BLOCK_VERIFY: per verify row of a batch whose lanes all sample,
  // its draft candidates' target probabilities and the rest's
  // (TargetCandidateRow), which block acceptance reads.
  metal::MetalBuffer candidateRows{};
};

// fastkernel wide prompt lookup: one request's verify rows, each row's
// proposal the next verify input row (sampling.metal decode_accept_lookup).
struct LookupAcceptanceBuffers final {
  metal::MetalBuffer inputTokens;
  metal::MetalBuffer targetVocabularyRows;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer outputTokens;
  metal::MetalBuffer retainedCount;
  metal::MetalBuffer acceptedCount;
  // The rows each 8-row tile retains.
  metal::MetalBuffer tileRetained;
};

struct AcceptanceBuffers final {
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer candidates;
  metal::MetalBuffer proposalProbabilities;
  metal::MetalBuffer targetVocabularyRows;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer outputTokens;
  metal::MetalBuffer retainedCounts;
  metal::MetalBuffer acceptedCounts;
  // SPLASH_BLOCK_VERIFY: the verify selection's buffers, whose candidate
  // rows block acceptance reads and whose rows its correction redraws from,
  // and each lane's BlockCorrectionRow.
  SamplingBuffers verify{};
  metal::MetalBuffer corrections{};
};

// Target token policy (penalties, min-p/top-k/top-p, constrained selection,
// stop-token exclusion, greedy argmax) and DFlash acceptance of the draft's
// proposals; the model only supplies policy, buffers, penalty words and its
// stop tokens. The draft's own selector is ops::DraftSelector.
// SPLASH_BLOCK_VERIFY (default on, read at construction): a verify batch
// whose lanes all sample is accepted by block verification (Sun et al. 2024)
// instead of the token rule. Both keep the target's distribution; the block
// rule accepts more draft tokens per cycle (+1.29% in fastkernel 1.0.0).
class Sampling final {
public:
  explicit Sampling(uint32_t vocabulary);

  // Exact scratch/output bytes for the fixed precompiled sampling ABI.
  // Counts may cover one lane or a batch of lanes; the operator owns sharding.
  [[nodiscard]] static SamplingWorkspace workspace(uint32_t rows);

  // A penalized request's penalty words (metal/abi/Sampling.h), rebuilt
  // when it takes a state lane from the history the lane's prefill consumes:
  // its prompt, then the generatedTokens outputs it emitted. The prompt bit
  // marks every prompt token when markPrompt, as only repetition reads it,
  // and the counts are of every token the target selected: the emitted
  // outputs and the pending anchor. Every token must be inside the
  // vocabulary, one word each, and the history must hold a prompt.
  static void rebuildPenaltyWords(std::span<uint32_t> words,
                                  std::span<const uint32_t> history,
                                  uint64_t generatedTokens,
                                  std::optional<uint32_t> pendingToken,
                                  bool markPrompt);
  // Counts the tokens one step selected into a request's penalty words.
  static void countPenaltyTokens(std::span<uint32_t> words,
                                 std::span<const uint32_t> selected);

  // A lane whose policy has active penalties has its logits rewritten in
  // place first, from its row of the penalty table; the rows must hold the
  // LM head's fresh output. Other lanes dispatch nothing new. A greedy lane
  // then takes each row's argmax and a sampled lane draws from each row's
  // distribution over the whole vocabulary, both among the tokens the lane
  // admits: the first token after a prompt from each lane's row at
  // rowOffset, and every verify row of a batch. Each selected row's token
  // goes to the output tokens in order, one per lane for the first token
  // and eight for verify; acceptance reads a sampled verify row's draft
  // token probability from the vocabulary rows. Both run the same kernels
  // (addSelection). A block-verified batch (blockVerifies) also writes each
  // verify row's candidate rows; its acceptance redraws a correction whose
  // accepted prefix is below 1 (decode_sample_block_correction), so the
  // acceptance buffers' verify buffers must be the ones addVerify selected.
  void addInitial(metal::CommandGraph &graph,
                  std::span<const SamplingPolicy> policies,
                  SamplingBuffers buffers, uint32_t rowOffset,
                  uint32_t stopToken0, uint32_t stopToken1,
                  const PenaltyTable &penalties) const;
  void addVerify(metal::CommandGraph &graph,
                 std::span<const SamplingPolicy> policies,
                 SamplingBuffers buffers, uint32_t stopToken0,
                 uint32_t stopToken1, const PenaltyTable &penalties) const;
  void addAcceptance(
      metal::CommandGraph &graph, AcceptanceBuffers buffers,
      std::span<const uint32_t> maximumRetained,
      std::span<const SamplingPolicy> policies, uint32_t stopToken0,
      uint32_t stopToken1) const;
  // fastkernel wide prompt lookup (SPLASH_WIDE_PROMPT_LOOKUP): one request's
  // `rows` verify rows (16 or 32) selected as one lane of that many rows, row r
  // reading mask row r + 1 and, below the last row, following its proposal,
  // verify input row r + 1, whose point-mass candidates the buffers' draft
  // candidates hold at position r. A penalized policy is refused: its words
  // would not count the earlier rows' proposals.
  void addLookupVerify(metal::CommandGraph &graph, const SamplingPolicy &policy,
                       SamplingBuffers buffers, uint32_t rows, uint32_t stopToken0,
                       uint32_t stopToken1) const;
  // Acceptance of those rows: greedy while each proposal is its row's token,
  // sampled with uniform SPLASH_SAMPLING_UNIFORMS + r for row r.
  void addLookupAcceptance(metal::CommandGraph &graph, LookupAcceptanceBuffers buffers,
                           uint32_t rows, uint32_t maximumRetained, const SamplingPolicy &policy,
                           uint32_t stopToken0, uint32_t stopToken1) const;

private:
  // The rows a selection takes from each lane (metal/abi/Sampling.h
  // TargetSamplingParams).
  struct TargetRows final {
    uint32_t rows = 0;
    uint32_t logitsRow = 0;
    uint32_t maskRow = 0;
    uint32_t uniform = 0;
    uint32_t draftedRows = 0;
  };
  // The verify selection's rows (addVerify), which the block correction
  // redraws from.
  [[nodiscard]] static TargetRows verifyRows() noexcept;

  // Penalizes the row at rowOffset of each penalized lane or, for verify,
  // all its rows, each also counting the draft tokens its context adds.
  // Policies come from validated requests (Model.hpp
  // SamplingParameters::validationError). It follows selection(), whose
  // check of every lane's logits covers those it rewrites.
  void addPenalties(metal::CommandGraph &graph,
                    std::span<const SamplingPolicy> policies,
                    const SamplingBuffers &buffers, const PenaltyTable &table,
                    uint32_t rowOffset, bool verify) const;
  // The parameters that select the rows `rows` names of every lane; throws
  // unless the buffers hold what the selection's kernels reach, before
  // addInitial or addVerify encodes anything.
  [[nodiscard]] TargetSamplingParams selection(std::span<const SamplingPolicy> policies,
                                               const SamplingBuffers &buffers, const TargetRows &rows,
                                               uint32_t stopToken0, uint32_t stopToken1) const;
  // Selects the rows `selection` names of its `lanes` lanes; a block
  // selection also writes the candidate rows.
  void addSelection(metal::CommandGraph &graph, const TargetSamplingParams &selection, uint32_t lanes,
                    const SamplingBuffers &buffers, bool block) const;
  // Whether a verify batch of these policies is block-verified: the switch
  // is on and every lane samples.
  [[nodiscard]] bool blockVerifies(std::span<const SamplingPolicy> policies) const noexcept;

  uint32_t vocabulary_ = 0;
  uint32_t maskWords_ = 0;
  bool blockVerify_ = false;
};

} // namespace splash::ops
