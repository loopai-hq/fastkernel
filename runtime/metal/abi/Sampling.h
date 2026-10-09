// Modified by Pulsar.
#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The uniforms one lane draws a decode cycle with, in [0, 1): the first
// token after a prompt draws SPLASH_UNIFORM_INITIAL; the draft's sampled
// proposal at position p draws SPLASH_UNIFORM_PROPOSALS + p; acceptance tests
// draft token p against SPLASH_UNIFORM_ACCEPTANCE + p; and a sampled verify
// row draws its correction, or the bonus token after the whole draft, with
// SPLASH_UNIFORM_CORRECTION. Lane l's uniforms start at
// l * SPLASH_SAMPLING_UNIFORMS.
#define SPLASH_UNIFORM_INITIAL 0u
#define SPLASH_UNIFORM_PROPOSALS 1u
#define SPLASH_UNIFORM_ACCEPTANCE                                          \
  (SPLASH_UNIFORM_PROPOSALS + SPLASH_DRAFT_PROPOSAL_TOKENS)
#define SPLASH_UNIFORM_CORRECTION                                          \
  (SPLASH_UNIFORM_ACCEPTANCE + SPLASH_DRAFT_PROPOSAL_TOKENS)
#define SPLASH_SAMPLING_UNIFORMS (SPLASH_UNIFORM_CORRECTION + 1u)

// One target-policy dispatch over lanes of SPLASH_TARGET_VERIFY_ROWS logits
// rows, SPLASH_TARGET_VERIFY_ROWS + 1 constraint-mask rows and
// SPLASH_SAMPLING_UNIFORMS uniforms each. Selected row s is row s % rows of
// lane s / rows: its logits row is lane * SPLASH_TARGET_VERIFY_ROWS +
// logits_row + s % rows, its mask row lane * (SPLASH_TARGET_VERIFY_ROWS + 1) +
// mask_row + s % rows, and its draw takes uniform
// lane * SPLASH_SAMPLING_UNIFORMS + uniform; workspaces and output tokens are
// indexed by s. Rows below drafted_rows follow draft token s % rows (verify
// input row s % rows + 1).
struct TargetSamplingParams {
  uint32_t vocabulary;
  uint32_t mask_words;
  uint32_t rows;
  uint32_t logits_row;
  uint32_t mask_row;
  uint32_t uniform;
  uint32_t drafted_rows;
  uint32_t top_k[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  float top_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  float min_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  // Lanes that sample; the others take the argmax.
  uint32_t sampling_mask;
  uint32_t constrained_mask;
  // Lanes that ignore end-of-sequence: they never select a stop token.
  uint32_t exclude_stop_mask;
  uint32_t stop_token_0;
  uint32_t stop_token_1;
};

static_assert(sizeof(TargetSamplingParams) == 112,
              "Target sampling parameters are 112 bytes on both sides");

// One shard's share of a sampled row's softmax denominator: the largest
// logit it admits, the sum of exp((logit - maximum) / temperature) over its
// admitted tokens, and how many it admits.
struct TargetShardMass {
  float maximum;
  float sum;
  uint32_t admitted;
};

static_assert(sizeof(TargetShardMass) == 12,
              "Target shard masses are 12 bytes on both sides");

// A sampled row's selection over the whole vocabulary. The search records
// the row's largest admitted logit and where its min-p/top-k/top-p
// distribution ends in the order of the logits (the key and id of its last
// token, metal/kernels/decode/sampling.metal). The draw writes a drafted
// row's target probability of its draft token; the token it draws goes to
// the output tokens: for a verify row with a draft token, the correction
// acceptance takes if it rejects that token (a draw from the residual
// distribution); otherwise a draw from the row's distribution.
struct TargetVocabularyRow {
  float maximum;
  uint32_t end_key;
  uint32_t end_last;
  float draft_probability;
};

static_assert(sizeof(TargetVocabularyRow) == 16,
              "Target vocabulary rows are 16 bytes on both sides");

// One range of the vocabulary in such a row's draw, which one simdgroup of
// the row's groups sums: the kept weight of its tokens other than the
// draft's candidates, and one past the last of those with weight.
struct TargetVocabularyRange {
  float rest;
  uint32_t after;
};

static_assert(sizeof(TargetVocabularyRange) == 8,
              "Target vocabulary ranges are 8 bytes on both sides");

// SPLASH_SAMPLER_TOPK32 (fork): a sampled lane that keeps at most this many
// tokens by top-k, and none by min-p, is searched from each shard's first
// this many tokens in the order of the logits, whose ids a row's draw
// ranges hold until its draw (decode_sample_mass_top32_sharded).
#define SPLASH_SAMPLER_TOP_TOKENS 32u

static_assert(SPLASH_TARGET_SAMPLING_SHARDS * SPLASH_SAMPLER_TOP_TOKENS *
                      sizeof(uint32_t) <=
                  SPLASH_TARGET_VOCABULARY_RANGES * sizeof(TargetVocabularyRange),
              "A row's draw ranges hold its shards' top token ids");

// SPLASH_BLOCK_VERIFY (block verification, Sun et al. 2024): a drafted verify
// row's target probability of each of its draft candidates, in the draft's
// candidate order (0 for a candidate listed again after its first entry), and
// that of every other token, which block acceptance's residual masses read.
struct TargetCandidateRow {
  float probability[SPLASH_DRAFT_CANDIDATES];
  float rest;
};

static_assert(sizeof(TargetCandidateRow) == 68,
              "Target candidate rows are 68 bytes on both sides");

// What block acceptance leaves each lane: its accepted prefix probability
// and, when that is below 1, the correction row's draft probabilities
// divided by it. A verify draw against those follows the residual
// max(prefix p - q, 0) scaled by 1 / prefix, which the block correction
// redraws the lane's correction from; at 1 the verify draw's token stands.
struct BlockCorrectionRow {
  float proposal[SPLASH_DRAFT_CANDIDATES];
  float prefix;
};

static_assert(sizeof(BlockCorrectionRow) == 68,
              "Block correction rows are 68 bytes on both sides");

// A penalized request's word for each vocabulary token, in its state lane's
// row of the penalty table (ops::Sampling::rebuildPenaltyWords): the
// prompt bit marks a prompt token, and the count is how often the target
// selected it.
#define SPLASH_PENALTY_PROMPT_BIT 0x80000000u
#define SPLASH_PENALTY_COUNT_MASK 0x7fffffffu

// The penalized lanes of one penalty dispatch. Each entry names the lane of
// its logits and the penalty table row it reads; rows penalizes that many
// rows of the lane's SPLASH_TARGET_VERIFY_ROWS, from row_offset.
// repetition_inverse is 1 / repetition, saturated to the largest float.
struct SamplingPenaltyParams {
  uint32_t vocabulary;
  uint32_t rows;
  uint32_t row_offset;
  uint32_t entries;
  uint32_t logits_lane[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t table_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition_inverse[SPLASH_MAXIMUM_BATCH_WIDTH];
  float presence[SPLASH_MAXIMUM_BATCH_WIDTH];
  float frequency[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SamplingPenaltyParams) == 112,
              "Sampling penalty parameters are 112 bytes on both sides");

// The batched selector and acceptance kernels' grids cover exactly the
// dispatch's lanes: per-lane arrays hold those lanes, and entries past them
// are zero and unread.
struct SelectorBatchParams {
  uint32_t anchor[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  uint32_t sampling_mask;
  uint32_t vocabulary;
  // SPLASH_DRAFT_TOP_P; a value outside (0, 1) keeps all the candidates.
  float top_p;
};

static_assert(sizeof(SelectorBatchParams) == 48,
              "Draft selector parameters are 48 bytes on both sides");

struct AcceptBatchParams {
  uint32_t remaining[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  uint32_t sampling_mask;
};

static_assert(sizeof(AcceptBatchParams) == 28,
              "Batched acceptance parameters are 28 bytes on both sides");

// SPLASH_BLOCK_VERIFY: decode_sample_block_correction redraws over the verify
// selection's rows (whose stop tokens acceptance shares) and recounts what a
// lane retains under its acceptance limit.
struct BlockCorrectionParams {
  TargetSamplingParams selection;
  uint32_t remaining[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(BlockCorrectionParams) == 128,
              "Block correction parameters are 128 bytes on both sides");
