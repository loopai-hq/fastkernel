// Modified by meowkernels.
#include "ops/DraftSelector.hpp"

#include "metal/abi/Sampling.h"
#include "ops/BufferExtent.hpp"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace splash::ops {
namespace {

constexpr uint32_t kShards = SPLASH_DRAFT_SAMPLING_SHARDS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group B1 dispatch
// against the 28 groups of B4 (wider groups speed up B1 and slow down B4).
constexpr uint32_t kEdgeThreads = 256;

float envFloat(const char *name, float fallback) {
  const char *value = std::getenv(name);
  if (!value)
    return fallback;
  char *end = nullptr;
  const float parsed = std::strtof(value, &end);
  if (end == value || *end != '\0')
    throw std::invalid_argument(std::string(name) + " must be a number");
  return parsed;
}

// SPLASH_DRAFT_TAU: 0.85 won fastkernel 1.0.0's exact block-rule replay
// (+0.80% acceptance, every category >= 0); its sampling passed chi^2 over
// 800 seeds. t = 1 multiplies by exactly 1: the drafter's own q.
float draftTau() {
  const float tau = envFloat("SPLASH_DRAFT_TAU", 0.85F);
  if (!std::isfinite(tau) || tau <= 0.0F)
    throw std::invalid_argument("SPLASH_DRAFT_TAU must be a positive number");
  return tau;
}

// SPLASH_DRAFT_TOP_P: 0.99 gave +0.26% (27B) / +0.29% (35B) in the block-rule
// replay, every category >= 0. p = 1 takes the kernel's unchanged path.
float draftTopP() {
  const float topP = envFloat("SPLASH_DRAFT_TOP_P", 0.99F);
  if (!(topP > 0.0F && topP <= 1.0F))
    throw std::invalid_argument("SPLASH_DRAFT_TOP_P must be in (0, 1]");
  return topP;
}

} // namespace

DraftSelector::DraftSelector(uint32_t vocabulary)
    : vocabulary_(vocabulary), tau_(draftTau()), topP_(draftTopP()) {
  if (!vocabulary)
    throw std::invalid_argument("invalid draft selector vocabulary");
}

DraftSelectorWorkspace DraftSelector::workspace(uint32_t positions) {
  if (!positions)
    throw std::invalid_argument("invalid draft selector workspace position count");
  const uint64_t candidates = uint64_t{positions} * SPLASH_DRAFT_CANDIDATES;
  // The partial values are followed by each position's 16 x 16 edge table.
  return {candidates * kShards * sizeof(uint32_t),
          candidates * (kShards + SPLASH_DRAFT_CANDIDATES) * sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(float),
          candidates * sizeof(float)};
}

SelectorBatchParams DraftSelector::params(std::span<const uint32_t> anchors,
                                          std::span<const SamplingPolicy> policies) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  SelectorBatchParams params{};
  params.lanes = static_cast<uint32_t>(anchors.size());
  params.vocabulary = vocabulary_;
  params.top_p = topP_;
  for (uint32_t lane = 0; lane < params.lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature * tau_;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  return params;
}

void DraftSelector::add(metal::CommandGraph &graph,
                        const DraftSelectorBuffers &buffers,
                        const DraftCodebooks &codebooks,
                        std::span<const uint32_t> anchors,
                        std::span<const SamplingPolicy> policies,
                        const DraftHeadMap &head) const {
  const SelectorBatchParams params = this->params(anchors, policies);
  const uint32_t lanes = params.lanes;
  // A lane's proposal positions take rows 1-7 of its eight query rows of the
  // logits and the selector hidden rows; a sampling lane draws its proposals
  // with its proposal uniforms and writes their candidates' probabilities.
  const DraftSelectorWorkspace workspace = DraftSelector::workspace(lanes * kPositions);
  const uint64_t rows = uint64_t{lanes} * SPLASH_DRAFT_QUERY_ROWS;
  const uint64_t codebook = uint64_t{vocabulary_} * SPLASH_DRAFT_SELECTOR_RANK * 2;
  // A restricted head's logits rows are headRows wide, and every head row
  // maps to a vocabulary id.
  if (head.rows > vocabulary_)
    throw std::invalid_argument("draft head has more rows than the vocabulary");
  const uint32_t headRows = head.rows ? head.rows : vocabulary_;
  if (head.rows)
    requireBytes(head.ids, uint64_t{headRows} * sizeof(uint32_t), "draft head id");
  requireBytes(buffers.logits, rows * headRows * sizeof(float), "draft logits");
  requireBytes(buffers.partialIds, workspace.partialIdsBytes, "draft selector partial id");
  requireBytes(buffers.partialValues, workspace.partialValuesBytes, "draft selector partial value");
  requireBytes(buffers.candidates, workspace.candidatesBytes, "draft candidate");
  requireBytes(buffers.unary, workspace.unaryBytes, "draft candidate score");
  requireBytes(buffers.selectorHidden, rows * SPLASH_DRAFT_SELECTOR_RANK * 2, "draft selector hidden");
  requireBytes(codebooks.predecessor, codebook, "draft predecessor codebook");
  requireBytes(codebooks.successor, codebook, "draft successor codebook");
  requireBytes(buffers.proposedTokens, uint64_t{lanes} * kPositions * sizeof(uint32_t), "proposed token");
  if (const uint64_t sampledLanes = std::bit_width(params.sampling_mask)) {
    requireBytes(buffers.uniforms,
                 ((sampledLanes - 1) * SPLASH_SAMPLING_UNIFORMS + SPLASH_UNIFORM_PROPOSALS + kPositions) * sizeof(float),
                 "proposal uniform");
    requireBytes(buffers.proposalProbabilities, sampledLanes * kPositions * SPLASH_DRAFT_CANDIDATES * sizeof(float),
                 "proposal probability");
  }
  graph.add("draft_select_top16_sharded",
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            headRows, {uint64_t{lanes} * kPositions * kShards, 1, 1});
  if (head.rows) {
    const uint32_t count = lanes * kPositions * kShards * SPLASH_DRAFT_CANDIDATES;
    graph.add("draft_map_head_ids", {buffers.partialIds, head.ids}, count,
              {(count + 255) / 256, 1, 1}, {256, 1, 1});
  }
  if (buffers.deviceParams) {
    // SPLASH_DRAFT_AHEAD: the same bytes, written on the GPU.
    requireBytes(buffers.deviceParams, sizeof(SelectorBatchParams), "draft selector params");
    graph.add("draft_select_edges",
              {buffers.partialIds, buffers.partialValues, buffers.candidates,
               buffers.unary, buffers.selectorHidden, codebooks.predecessor,
               codebooks.successor, buffers.deviceParams},
              {uint64_t{lanes} * kPositions, 1, 1}, {kEdgeThreads, 1, 1});
    graph.add("draft_select_dflash",
              {buffers.candidates, buffers.unary, buffers.partialValues,
               buffers.uniforms, buffers.proposedTokens,
               buffers.proposalProbabilities, buffers.deviceParams},
              {lanes, 1, 1}, {1, 1, 1});
    return;
  }
  graph.add("draft_select_edges",
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden, codebooks.predecessor,
             codebooks.successor},
            params, {uint64_t{lanes} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add("draft_select_dflash",
            {buffers.candidates, buffers.unary, buffers.partialValues,
             buffers.uniforms, buffers.proposedTokens,
             buffers.proposalProbabilities},
            params, {lanes, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops
