// Modified by meowkernels.
#include "ops/Sampling.hpp"

#include "metal/EnvSwitch.hpp"
#include "metal/abi/Sampling.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kTargetShards = SPLASH_TARGET_SAMPLING_SHARDS;
constexpr uint32_t kVocabularyThreads = SPLASH_TARGET_VOCABULARY_THREADS;
constexpr uint32_t kVocabularyGroups = SPLASH_TARGET_VOCABULARY_GROUPS;

// A sampled lane keeps its topK most likely tokens, and every token for 0 or
// a topK past the vocabulary (top-k disabled).
uint32_t effectiveTopK(const SamplingPolicy &policy,
                       uint32_t vocabulary) noexcept {
  return policy.topK && policy.topK < vocabulary ? policy.topK : vocabulary;
}
constexpr uint32_t kPenaltyThreads = 256;

// SPLASH_SAMPLER_TOPK32: every sampled lane of the selection keeps at most
// SPLASH_SAMPLER_TOP_TOKENS tokens by top-k and none by min-p.
bool keepsTopTokens(const TargetSamplingParams &params, uint32_t lanes) noexcept {
  for (uint32_t lane = 0; lane < lanes; ++lane)
    if ((params.sampling_mask & (uint32_t{1} << lane)) &&
        (params.top_k[lane] > SPLASH_SAMPLER_TOP_TOKENS || params.min_p[lane] > 0.0F))
      return false;
  return true;
}

void requireVocabulary(std::span<const uint32_t> tokens, size_t vocabulary) {
  if (std::any_of(tokens.begin(), tokens.end(),
                  [&](uint32_t token) { return token >= vocabulary; }))
    throw std::invalid_argument("penalty token is outside the vocabulary");
}

} // namespace

SamplingWorkspace Sampling::workspace(uint32_t rows) {
  if (!rows)
    throw std::invalid_argument("invalid sampling workspace row count");
  const uint64_t shards = uint64_t{rows} * kTargetShards;
  return {shards * sizeof(float),
          shards * sizeof(uint32_t),
          shards * sizeof(TargetShardMass),
          uint64_t{rows} * sizeof(TargetVocabularyRow),
          uint64_t{rows} * SPLASH_TARGET_VOCABULARY_RANGES *
              sizeof(TargetVocabularyRange),
          uint64_t{rows} * sizeof(uint32_t),
          uint64_t{rows} * sizeof(TargetCandidateRow)};
}

void Sampling::rebuildPenaltyWords(std::span<uint32_t> words,
                                   std::span<const uint32_t> history,
                                   uint64_t generatedTokens,
                                   std::optional<uint32_t> pendingToken,
                                   bool markPrompt) {
  if (history.size() <= generatedTokens)
    throw std::logic_error("request history holds no prompt");
  const std::span<const uint32_t> prompt =
      history.first(history.size() - generatedTokens);
  requireVocabulary(history, words.size());
  if (pendingToken)
    requireVocabulary({&*pendingToken, 1}, words.size());
  std::fill(words.begin(), words.end(), 0U);
  if (markPrompt) {
    for (const uint32_t token : prompt)
      words[token] |= SPLASH_PENALTY_PROMPT_BIT;
  }
  for (const uint32_t token : history.subspan(prompt.size()))
    ++words[token];
  if (pendingToken)
    ++words[*pendingToken];
}

// Counts stay far below the prompt bit: a request selects at most one token
// per position of its context.
void Sampling::countPenaltyTokens(std::span<uint32_t> words,
                                  std::span<const uint32_t> selected) {
  requireVocabulary(selected, words.size());
  for (const uint32_t token : selected)
    ++words[token];
}

// SPLASH_BLOCK_VERIFY (default on; =0 is upstream's token rule). +1.29%
// tokens per cycle in Pulsar 1.0.0, exact against a serial reference on
// 22,848 real draws there.
Sampling::Sampling(uint32_t vocabulary)
    : vocabulary_(vocabulary), maskWords_((vocabulary + 31) / 32),
      blockVerify_(metal::envSwitch("SPLASH_BLOCK_VERIFY")) {
  if (!vocabulary)
    throw std::invalid_argument("invalid sampling geometry");
}

void Sampling::addPenalties(metal::CommandGraph &graph,
                            std::span<const SamplingPolicy> policies,
                            const SamplingBuffers &buffers,
                            const PenaltyTable &table, uint32_t rowOffset,
                            bool verify) const {
  SamplingPenaltyParams params{};
  params.vocabulary = vocabulary_;
  params.rows = verify ? SPLASH_TARGET_VERIFY_ROWS : 1;
  params.row_offset = rowOffset;
  const uint64_t rowBytes = uint64_t{vocabulary_} * sizeof(uint32_t);
  for (uint32_t lane = 0; lane < policies.size(); ++lane) {
    const SamplingPenalties &penalties = policies[lane].penalties;
    if (!penalties.active())
      continue;
    // The kernel indexes the whole table by this row.
    if (lane >= table.rows.size())
      throw std::invalid_argument("penalized lane has no penalty table row");
    requireBytes(table.words, (uint64_t{table.rows[lane]} + 1) * rowBytes, "penalty table");
    const uint32_t entry = params.entries++;
    params.logits_lane[entry] = lane;
    params.table_row[entry] = table.rows[lane];
    params.repetition[entry] = penalties.repetition;
    // 1 / 2^-149 overflows; the saturated inverse keeps the product finite.
    params.repetition_inverse[entry] = std::min(
        1.0F / penalties.repetition, std::numeric_limits<float>::max());
    params.presence[entry] = penalties.presence;
    params.frequency[entry] = penalties.frequency;
  }
  if (!params.entries)
    return;
  // Entries follow the lanes, so the last one's lane is the highest.
  const uint64_t lanes = uint64_t{params.logits_lane[params.entries - 1]} + 1;
  if (verify)
    requireBytes(buffers.inputTokens, lanes * SPLASH_TARGET_VERIFY_ROWS * sizeof(uint32_t), "verify input token");
  const metal::DispatchSize groups{
      (vocabulary_ + kPenaltyThreads - 1) / kPenaltyThreads, params.entries, 1};
  if (!verify) {
    graph.add("decode_sample_penalize", {buffers.logits, table.words}, params,
              groups, {kPenaltyThreads, 1, 1});
    return;
  }
  graph.add("decode_sample_penalize_verify",
            {buffers.logits, table.words, buffers.inputTokens}, params, groups,
            {kPenaltyThreads, 1, 1});
}

void Sampling::addInitial(metal::CommandGraph &graph,
                          std::span<const SamplingPolicy> policies,
                          SamplingBuffers buffers, uint32_t rowOffset,
                          uint32_t stopToken0, uint32_t stopToken1,
                          const PenaltyTable &penalties) const {
  if (policies.empty() || policies.size() > kMaximumLanes)
    throw std::invalid_argument("invalid sampling batch width");
  if (rowOffset >= SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("invalid initial sampling row");
  const TargetSamplingParams params =
      selection(policies, buffers, {1, rowOffset, 0, SPLASH_UNIFORM_INITIAL, 0}, stopToken0, stopToken1);
  addPenalties(graph, policies, buffers, penalties, rowOffset, false);
  addSelection(graph, params, static_cast<uint32_t>(policies.size()), buffers, false);
}

void Sampling::addVerify(metal::CommandGraph &graph,
                         std::span<const SamplingPolicy> policies,
                         SamplingBuffers buffers, uint32_t stopToken0,
                         uint32_t stopToken1,
                         const PenaltyTable &penalties) const {
  if (policies.empty() || policies.size() > kMaximumLanes)
    throw std::invalid_argument("invalid sampling batch width");
  const TargetSamplingParams params = selection(policies, buffers, verifyRows(), stopToken0, stopToken1);
  const uint32_t lanes = static_cast<uint32_t>(policies.size());
  const bool block = blockVerifies(policies);
  // A block draw writes the candidate rows of every drafted row.
  if (block)
    requireBytes(buffers.candidateRows,
                 (uint64_t{lanes - 1} * SPLASH_TARGET_VERIFY_ROWS + SPLASH_DRAFT_PROPOSAL_TOKENS) *
                     sizeof(TargetCandidateRow),
                 "target candidate row");
  addPenalties(graph, policies, buffers, penalties, 0, true);
  addSelection(graph, params, lanes, buffers, block);
}

// Verify row r of a lane reads mask row r + 1 and, below the last row,
// follows draft token r; every row draws with the lane's correction
// uniform.
Sampling::TargetRows Sampling::verifyRows() noexcept {
  return {SPLASH_TARGET_VERIFY_ROWS, 0, 1, SPLASH_UNIFORM_CORRECTION, SPLASH_DRAFT_PROPOSAL_TOKENS};
}

bool Sampling::blockVerifies(std::span<const SamplingPolicy> policies) const noexcept {
  return blockVerify_ &&
         std::all_of(policies.begin(), policies.end(), [](const SamplingPolicy &policy) { return policy.samples(); });
}

void Sampling::addLookupVerify(metal::CommandGraph &graph, const SamplingPolicy &policy,
                               SamplingBuffers buffers, uint32_t rows, uint32_t stopToken0,
                               uint32_t stopToken1) const {
  if (rows != 2 * SPLASH_TARGET_VERIFY_ROWS && rows != 4 * SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("invalid lookup verify rows");
  if (policy.penalties.active())
    throw std::invalid_argument("lookup verify refuses penalized lanes");
  const std::array<SamplingPolicy, 1> policies{policy};
  const TargetSamplingParams params =
      selection(policies, buffers, {rows, 0, 1, SPLASH_UNIFORM_CORRECTION, rows - 1}, stopToken0, stopToken1);
  addSelection(graph, params, 1, buffers, false);  // lookup proposals are point masses: the token rule
}

void Sampling::addLookupAcceptance(metal::CommandGraph &graph, LookupAcceptanceBuffers buffers,
                                   uint32_t rows, uint32_t maximumRetained, const SamplingPolicy &policy,
                                   uint32_t stopToken0, uint32_t stopToken1) const {
  if ((rows != 2 * SPLASH_TARGET_VERIFY_ROWS && rows != 4 * SPLASH_TARGET_VERIFY_ROWS) || !maximumRetained ||
      maximumRetained > rows)
    throw std::invalid_argument("invalid lookup acceptance");
  AcceptBatchParams params{};
  params.remaining[0] = maximumRetained;
  params.stop_token_0 = stopToken0;
  params.stop_token_1 = stopToken1;
  params.sampling_mask = policy.samples() ? 1U : 0U;
  requireBytes(buffers.inputTokens, uint64_t{rows} * sizeof(uint32_t), "lookup input token");
  requireBytes(buffers.outputTokens, uint64_t{rows} * sizeof(uint32_t), "lookup target token");
  requireBytes(buffers.retainedCount, sizeof(uint32_t), "lookup retained count");
  requireBytes(buffers.acceptedCount, sizeof(uint32_t), "lookup accepted count");
  requireBytes(buffers.tileRetained, uint64_t{rows / SPLASH_TARGET_VERIFY_ROWS} * sizeof(uint32_t),
               "lookup tile retained count");
  if (policy.samples()) {
    requireBytes(buffers.targetVocabularyRows, uint64_t{rows - 1} * sizeof(TargetVocabularyRow),
                 "lookup target vocabulary row");
    requireBytes(buffers.uniforms, (uint64_t{SPLASH_SAMPLING_UNIFORMS} + rows - 1) * sizeof(float),
                 "lookup acceptance uniform");
  }
  graph.add(rows == 2 * SPLASH_TARGET_VERIFY_ROWS ? "decode_accept_lookup16" : "decode_accept_lookup32",
            {buffers.inputTokens, buffers.targetVocabularyRows, buffers.uniforms, buffers.outputTokens,
             buffers.retainedCount, buffers.acceptedCount, buffers.tileRetained},
            params, {1, 1, 1}, {1, 1, 1});
}

TargetSamplingParams Sampling::selection(std::span<const SamplingPolicy> policies,
                                         const SamplingBuffers &buffers, const TargetRows &rows,
                                         uint32_t stopToken0, uint32_t stopToken1) const {
  TargetSamplingParams params{};
  params.vocabulary = vocabulary_;
  params.mask_words = maskWords_;
  params.rows = rows.rows;
  params.logits_row = rows.logitsRow;
  params.mask_row = rows.maskRow;
  params.uniform = rows.uniform;
  params.drafted_rows = rows.draftedRows;
  params.stop_token_0 = stopToken0;
  params.stop_token_1 = stopToken1;
  for (uint32_t lane = 0; lane < policies.size(); ++lane) {
    const SamplingPolicy &policy = policies[lane];
    if (policy.samples()) {
      params.top_k[lane] = effectiveTopK(policy, vocabulary_);
      params.temperature[lane] = policy.temperature;
      params.top_p[lane] = policy.topP;
      params.min_p[lane] = policy.minP;
      params.sampling_mask |= uint32_t{1} << lane;
    }
    if (policy.constrained)
      params.constrained_mask |= uint32_t{1} << lane;
    if (policy.excludesStopTokens)
      params.exclude_stop_mask |= uint32_t{1} << lane;
  }
  // Selected row s is row s % rows of lane s / rows (metal/abi/Sampling.h),
  // and the kernels of a lane's policy reach that lane's rows: each buffer
  // holds the rows up to the last lane that reads it, a workspace the
  // workspace() of those rows.
  const uint64_t lanes = policies.size();
  const uint64_t greedyLanes = std::bit_width(((uint32_t{1} << lanes) - 1) & ~params.sampling_mask);
  const uint64_t sampledLanes = std::bit_width(params.sampling_mask);
  const uint64_t constrainedLanes = std::bit_width(params.constrained_mask);
  const auto rowsWorkspace = [&](uint64_t policyLanes) {
    return policyLanes ? workspace(static_cast<uint32_t>(policyLanes * rows.rows)) : SamplingWorkspace{};
  };
  const SamplingWorkspace greedy = rowsWorkspace(greedyLanes), sampled = rowsWorkspace(sampledLanes);
  requireBytes(buffers.logits,
               ((lanes - 1) * SPLASH_TARGET_VERIFY_ROWS + rows.logitsRow + rows.rows) * vocabulary_ * sizeof(float),
               "sampling logits");
  if (constrainedLanes)
    requireBytes(buffers.constraintMasks,
                 ((constrainedLanes - 1) * (SPLASH_TARGET_VERIFY_ROWS + 1) + rows.maskRow + rows.rows) * maskWords_ *
                     sizeof(uint32_t),
                 "constraint mask");
  requireBytes(buffers.outputTokens, lanes * rows.rows * sizeof(uint32_t), "sampled token");
  requireBytes(buffers.argmaxValues, greedy.argmaxValuesBytes, "argmax value");
  requireBytes(buffers.argmaxIndices, greedy.argmaxIndicesBytes, "argmax index");
  requireBytes(buffers.partialMasses, sampled.partialMassesBytes, "sampling mass");
  requireBytes(buffers.vocabularyRows, sampled.vocabularyRowsBytes, "sampling vocabulary row");
  requireBytes(buffers.vocabularyRanges, sampled.vocabularyRangesBytes, "sampling vocabulary range");
  requireBytes(buffers.vocabularyArrivals, sampled.vocabularyArrivalsBytes, "sampling arrival");
  if (sampledLanes) {
    const uint64_t last = sampledLanes - 1, drafted = std::min(rows.rows, rows.draftedRows);
    requireBytes(buffers.uniforms, (last * SPLASH_SAMPLING_UNIFORMS + rows.uniform + 1) * sizeof(float),
                 "sampling uniform");
    // A drafted row reads its draft token, the next verify input row, and
    // its proposal position's candidates.
    if (drafted) {
      requireBytes(buffers.inputTokens, (last * rows.rows + drafted + 1) * sizeof(uint32_t), "verify input token");
      const uint64_t candidates = (last * SPLASH_DRAFT_PROPOSAL_TOKENS + drafted) * SPLASH_DRAFT_CANDIDATES;
      requireBytes(buffers.draftCandidates, candidates * sizeof(uint32_t), "draft candidate");
      requireBytes(buffers.draftProbabilities, candidates * sizeof(float), "draft probability");
    }
  }
  return params;
}

void Sampling::addSelection(metal::CommandGraph &graph, const TargetSamplingParams &params, uint32_t lanes,
                            const SamplingBuffers &buffers, bool block) const {
  // Greedy and sampled lanes run their own kernels, each over the selected
  // rows of every lane; the groups of the other kind's lanes return at once.
  const uint64_t selected = uint64_t{lanes} * params.rows;
  if (params.sampling_mask != (uint32_t{1} << lanes) - 1) {
    graph.add("decode_sample_argmax_sharded",
              {buffers.logits, buffers.constraintMasks, buffers.argmaxValues,
               buffers.argmaxIndices},
              params, {selected * kTargetShards, 1, 1});
    graph.add("decode_sample_argmax_reduce",
              {buffers.argmaxValues, buffers.argmaxIndices,
               buffers.outputTokens},
              params, {selected, 1, 1}, {32, 1, 1});
  }
  if (!params.sampling_mask)
    return;
  // SPLASH_SAMPLER_TOPK32 (default on; =0 searches every row over the whole
  // vocabulary, upstream's path; same records either way), read per
  // selection so an in-process A/B can switch it between cycles. The shards'
  // top token ids wait in the draw ranges, which the draw writes only after
  // the search reads them.
  if (metal::envSwitch("SPLASH_SAMPLER_TOPK32") && keepsTopTokens(params, lanes)) {
    graph.add("decode_sample_mass_top32_sharded",
              {buffers.logits, buffers.constraintMasks, buffers.partialMasses, buffers.vocabularyRanges},
              params, {selected * kTargetShards, 1, 1});
    graph.add("decode_sample_vocabulary_search_top32",
              {buffers.logits, buffers.constraintMasks, buffers.partialMasses, buffers.vocabularyRanges,
               buffers.vocabularyRows},
              params, {selected, 1, 1}, {kVocabularyThreads, 1, 1});
  } else {
    graph.add("decode_sample_mass_sharded",
              {buffers.logits, buffers.constraintMasks, buffers.partialMasses},
              params, {selected * kTargetShards, 1, 1});
    graph.add("decode_sample_vocabulary_search",
              {buffers.logits, buffers.constraintMasks, buffers.partialMasses,
               buffers.vocabularyRows},
              params, {selected, 1, 1}, {kVocabularyThreads, 1, 1});
  }
  if (block) {
    graph.add("decode_sample_vocabulary_draw_block",
              {buffers.logits, buffers.constraintMasks, buffers.vocabularyRows, buffers.inputTokens,
               buffers.draftCandidates, buffers.draftProbabilities, buffers.uniforms, buffers.outputTokens,
               buffers.vocabularyRanges, buffers.vocabularyArrivals, buffers.candidateRows},
              params, {selected * kVocabularyGroups, 1, 1}, {kVocabularyThreads, 1, 1});
    return;
  }
  graph.add("decode_sample_vocabulary_draw",
            {buffers.logits, buffers.constraintMasks, buffers.vocabularyRows,
             buffers.inputTokens, buffers.draftCandidates,
             buffers.draftProbabilities, buffers.uniforms,
             buffers.outputTokens, buffers.vocabularyRanges,
             buffers.vocabularyArrivals},
            params, {selected * kVocabularyGroups, 1, 1},
            {kVocabularyThreads, 1, 1});
}

void Sampling::addAcceptance(
    metal::CommandGraph &graph, AcceptanceBuffers buffers,
    std::span<const uint32_t> maximumRetained,
    std::span<const SamplingPolicy> policies, uint32_t stopToken0,
    uint32_t stopToken1) const {
  if (maximumRetained.empty() || maximumRetained.size() != policies.size() ||
      maximumRetained.size() > kMaximumLanes)
    throw std::invalid_argument("invalid DFlash acceptance batch");
  const uint32_t lanes = static_cast<uint32_t>(maximumRetained.size());
  AcceptBatchParams params{};
  params.stop_token_0 = stopToken0;
  params.stop_token_1 = stopToken1;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (!maximumRetained[lane] ||
        maximumRetained[lane] > SPLASH_TARGET_VERIFY_ROWS)
      throw std::invalid_argument("invalid DFlash retention limit");
    params.remaining[lane] = maximumRetained[lane];
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  // A lane reads its draft's tokens and its verify rows' tokens, the latter
  // up to the longer of the draft and its retention limit; a sampling lane
  // also its draft's candidates and their probabilities, its rows' target
  // probabilities and its acceptance uniforms (metal/abi/Sampling.h).
  constexpr uint64_t kProposals = SPLASH_DRAFT_PROPOSAL_TOKENS;
  requireBytes(buffers.proposedTokens, lanes * kProposals * sizeof(uint32_t), "proposed token");
  requireBytes(buffers.outputTokens,
               (uint64_t{lanes - 1} * SPLASH_TARGET_VERIFY_ROWS + std::max<uint64_t>(kProposals, maximumRetained.back())) *
                   sizeof(uint32_t),
               "target token");
  requireBytes(buffers.retainedCounts, lanes * sizeof(uint32_t), "retained count");
  requireBytes(buffers.acceptedCounts, lanes * sizeof(uint32_t), "accepted count");
  if (const uint64_t sampledLanes = std::bit_width(params.sampling_mask)) {
    const uint64_t candidates = sampledLanes * kProposals * SPLASH_DRAFT_CANDIDATES;
    requireBytes(buffers.candidates, candidates * sizeof(uint32_t), "draft candidate");
    requireBytes(buffers.proposalProbabilities, candidates * sizeof(float), "draft probability");
    requireBytes(buffers.targetVocabularyRows,
                 ((sampledLanes - 1) * SPLASH_TARGET_VERIFY_ROWS + kProposals) * sizeof(TargetVocabularyRow),
                 "target vocabulary row");
    requireBytes(buffers.uniforms,
                 ((sampledLanes - 1) * SPLASH_SAMPLING_UNIFORMS + SPLASH_UNIFORM_ACCEPTANCE + kProposals) *
                     sizeof(float),
                 "acceptance uniform");
  }
  if (blockVerifies(policies)) {
    // Block acceptance also reads candidate rows 1..6 of each lane and
    // leaves its accepted prefix; the correction redraws row `accepted` (at
    // most row 6) over the verify selection with the correction uniform.
    const TargetSamplingParams verify = selection(policies, buffers.verify, verifyRows(), stopToken0, stopToken1);
    requireBytes(buffers.verify.candidateRows,
                 (uint64_t{lanes - 1} * SPLASH_TARGET_VERIFY_ROWS + kProposals) * sizeof(TargetCandidateRow),
                 "target candidate row");
    requireBytes(buffers.uniforms,
                 (uint64_t{lanes - 1} * SPLASH_SAMPLING_UNIFORMS + SPLASH_UNIFORM_CORRECTION + 1) * sizeof(float),
                 "correction uniform");
    requireBytes(buffers.corrections, lanes * sizeof(BlockCorrectionRow), "block correction");
    graph.add("decode_accept_dflash_block",
              {buffers.proposedTokens, buffers.candidates, buffers.proposalProbabilities,
               buffers.targetVocabularyRows, buffers.verify.candidateRows, buffers.uniforms, buffers.outputTokens,
               buffers.retainedCounts, buffers.acceptedCounts, buffers.corrections},
              params, {lanes, 1, 1}, {32, 1, 1});
    BlockCorrectionParams correction{verify, {}};
    std::copy_n(params.remaining, lanes, correction.remaining);
    graph.add("decode_sample_block_correction",
              {buffers.verify.logits, buffers.verify.constraintMasks, buffers.targetVocabularyRows,
               buffers.verify.inputTokens, buffers.candidates, buffers.uniforms, buffers.outputTokens,
               buffers.verify.vocabularyRanges, buffers.verify.vocabularyArrivals, buffers.corrections,
               buffers.acceptedCounts, buffers.retainedCounts},
              correction, {uint64_t{lanes} * kVocabularyGroups, 1, 1}, {kVocabularyThreads, 1, 1});
    return;
  }
  graph.add("decode_accept_dflash",
            {buffers.proposedTokens, buffers.candidates,
             buffers.proposalProbabilities, buffers.targetVocabularyRows,
             buffers.uniforms, buffers.outputTokens, buffers.retainedCounts,
             buffers.acceptedCounts},
            params, {lanes, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops
