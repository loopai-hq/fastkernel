// Modified by meowkernels.
#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/Sampling.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

struct DraftSelectorWorkspace final {
  uint64_t partialIdsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t candidatesBytes = 0;
  uint64_t unaryBytes = 0;
  uint64_t proposalProbabilitiesBytes = 0;
};

struct DraftSelectorBuffers final {
  // fp32 [rows][vocabulary], or [rows][head rows] for a restricted head.
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
  // SPLASH_DRAFT_AHEAD: a GPU-written SelectorBatchParams bound in place of
  // the host bytes of params(). Empty: host bytes.
  metal::MetalBuffer deviceParams{};
};

// SPLASH_DRAFT_HEAD_IDS: a restricted draft head of `rows` rows (0 = the
// whole vocabulary), whose logits rows are head rows; `ids` holds the
// vocabulary id of each (u32 [rows]), so the candidates stay vocabulary ids.
struct DraftHeadMap final {
  metal::MetalBuffer ids;
  uint32_t rows = 0;
};

// The draft's selector codebooks, which score the edge from a proposal
// position's predecessor candidate to each of its candidates.
struct DraftCodebooks final {
  metal::MetalBuffer predecessor;
  metal::MetalBuffer successor;
};

// The DFlash draft's proposal policy (draft_select_* in
// metal/kernels/decode/sampling.metal): each lane keeps the
// SPLASH_DRAFT_CANDIDATES most likely draft tokens of every proposal
// position, scores each candidate with its edge from the previous position's
// choice, and walks the SPLASH_DRAFT_PROPOSAL_TOKENS positions greedily or,
// for a sampling lane, drawing at its temperature.
// Read at construction, both exact for any value (acceptance verifies the
// distribution a lane proposed from), neither touching a greedy lane:
// SPLASH_DRAFT_TAU=t (default 0.85; 1 = off): a sampling lane draws at its
// temperature times t. SPLASH_DRAFT_TOP_P=p (default 0.99; 1 = off): it then
// keeps its most probable candidates until their mass reaches p and draws
// from those, renormalized.
class DraftSelector final {
public:
  explicit DraftSelector(uint32_t vocabulary);

  // Exact scratch/output bytes for that many proposal positions.
  [[nodiscard]] static DraftSelectorWorkspace workspace(uint32_t positions);

  void add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
           const DraftCodebooks &codebooks, std::span<const uint32_t> anchors,
           std::span<const SamplingPolicy> policies,
           const DraftHeadMap &head = {}) const;
  // The parameters add() binds: each lane's anchor and temperature (times
  // SPLASH_DRAFT_TAU), the sampling lanes and SPLASH_DRAFT_TOP_P.
  [[nodiscard]] SelectorBatchParams params(std::span<const uint32_t> anchors,
                                           std::span<const SamplingPolicy> policies) const;

private:
  uint32_t vocabulary_ = 0;
  float tau_ = 1.0F;
  float topP_ = 1.0F;
};

} // namespace splash::ops
