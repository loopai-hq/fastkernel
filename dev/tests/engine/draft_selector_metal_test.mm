// Modified by meowkernels.
// DFlash draft selector against a direct CPU reference: the sharded top-16
// scan and its reduce must return exactly the sixteen largest logits of every
// proposal row in (value desc, id asc) order, and the codebook walk must pick
// the same tokens as a double-precision evaluation of the same scores. The
// vocabularies cover the production size, an odd size that misaligns the
// 16-byte vectors and leaves shards with only a few tokens, and one wider
// than a single register chunk per thread. The logits are fp32, and their
// order is decided below the bf16 spacing.
// Fork switches, set per case: SPLASH_DRAFT_TAU scales a sampling lane's
// temperature and SPLASH_DRAFT_TOP_P keeps its most probable candidates up
// to that mass (the reference evaluates both in double); a restricted head
// (SPLASH_DRAFT_HEAD_IDS) selects over head rows and maps them back to
// vocabulary ids.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/DraftSelector.hpp"
#include "tuning/LinearNumerics.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kRank = SPLASH_DRAFT_SELECTOR_RANK;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;

using splash::test::rejects;
using splash::test::require;
using splash::test::requireExtent;

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F;
  }
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }

private:
  uint64_t state_;
};

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, "draft selector");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

MetalBuffer randomBfloat(MetalBackend &backend, uint64_t count, Random &random,
                         float scale) {
  MetalBuffer buffer = allocate(backend, count * sizeof(uint16_t));
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    values[index] = tuning::floatToBf16(random.unit() * scale);
  return buffer;
}

// Row patterns: peaked logits with a few spikes (the production shape),
// uniform noise, heavy ties from a seven-value alphabet with -inf runs, and
// a row with ten finite tokens so -inf tokens fill the tail by id.
enum class Pattern : uint8_t { Peaked, Uniform, Ties, Sparse };

void fillRow(float *row, uint32_t vocabulary, Pattern pattern,
             Random &random) {
  for (uint32_t token = 0; token < vocabulary; ++token) {
    float value = 0.0F;
    switch (pattern) {
    case Pattern::Peaked:
      value = -8.0F + 3.0F * random.unit() * random.unit();
      break;
    case Pattern::Uniform:
      value = 4.0F * random.unit();
      break;
    case Pattern::Ties:
      value = random.next() % 11 == 0
                  ? -INFINITY
                  : static_cast<float>(int(random.next() % 7)) - 3.0F;
      break;
    case Pattern::Sparse:
      value = -INFINITY;
      break;
    }
    row[token] = value;
  }
  if (pattern == Pattern::Peaked) {
    for (uint32_t spike = 0; spike < 40; ++spike)
      row[random.next() % vocabulary] = 4.0F + 6.0F * random.unit();
  }
  if (pattern == Pattern::Sparse) {
    for (uint32_t finite = 0; finite < 10; ++finite)
      row[random.next() % vocabulary] = random.unit();
  }
}

// The row's sixteen largest tokens: value descending, id ascending on ties.
std::vector<uint32_t> referenceTop16(const float *row, uint32_t vocabulary) {
  std::vector<uint32_t> order(vocabulary);
  std::iota(order.begin(), order.end(), 0U);
  const auto beats = [&](uint32_t a, uint32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
  };
  const size_t keep = std::min<size_t>(kCandidates, vocabulary);
  std::partial_sort(order.begin(), order.begin() + keep, order.end(), beats);
  order.resize(keep);
  return order;
}

struct Case final {
  uint32_t vocabulary;
  uint32_t lanes;
  bool sampling;
  float tau = 1.0F;
  float topP = 1.0F;
  // Rows of a restricted head with an ascending random id map; 0 = none.
  uint32_t headRows = 0;
};

// A sampling lane's proposal distribution at one position, as
// draft_select_dflash and draft_top_p_select define it: softmax at the
// temperature over the candidates' scores, and with top_p in (0, 1) only the
// most probable candidates (ties: the lower index) until their mass reaches
// top_p, renormalized. Ambiguous when that mass lies within float rounding
// of the bound at the last candidate kept.
struct Proposal final {
  std::array<double, kCandidates> probabilities{};
  uint32_t lastKept = kCandidates - 1;
  bool ambiguous = false;
};

Proposal referenceProposal(const std::array<double, kCandidates> &scores,
                           double temperature, float topP) {
  Proposal result;
  const double maximum = *std::max_element(scores.begin(), scores.end());
  std::array<double, kCandidates> weights{};
  double sum = 0.0;
  for (uint32_t rank = 0; rank < kCandidates; ++rank) {
    weights[rank] = std::exp((scores[rank] - maximum) / temperature);
    sum += weights[rank];
  }
  if (!(topP > 0.0F && topP < 1.0F)) {
    for (uint32_t rank = 0; rank < kCandidates; ++rank)
      result.probabilities[rank] = weights[rank] / sum;
    return result;
  }
  const double bound = double(topP) * sum;
  std::array<bool, kCandidates> kept{};
  double keptSum = 0.0;
  uint32_t count = 0;
  result.lastKept = 0;
  while (keptSum < bound && count < kCandidates) {
    uint32_t best = kCandidates;
    for (uint32_t rank = 0; rank < kCandidates; ++rank)
      if (!kept[rank] && (best == kCandidates || weights[rank] > weights[best]))
        best = rank;
    result.ambiguous |= std::fabs(keptSum - bound) <= 1e-5 * sum;
    kept[best] = true;
    keptSum += weights[best];
    ++count;
    result.lastKept = std::max(result.lastKept, best);
  }
  result.ambiguous |= std::fabs(keptSum - bound) <= 1e-5 * sum;
  for (uint32_t rank = 0; rank < kCandidates; ++rank)
    result.probabilities[rank] = kept[rank] ? weights[rank] / keptSum : 0.0;
  return result;
}

void runCase(MetalBackend &backend, const Case &c) {
  Random random(0x5e1ec7 + uint64_t{c.vocabulary} * 8 + c.lanes * 2 + c.sampling + c.headRows);
  const uint32_t rows = c.lanes * kRows;
  const uint32_t positions = c.lanes * kPositions;
  const auto workspace = DraftSelector::workspace(positions);
  if (setenv("SPLASH_DRAFT_TAU", std::to_string(c.tau).c_str(), 1) != 0 ||
      setenv("SPLASH_DRAFT_TOP_P", std::to_string(c.topP).c_str(), 1) != 0)
    throw std::runtime_error("cannot set the drafter's tau and top-p");
  const DraftSelector selector(c.vocabulary);
  // The logits rows are head rows, each standing for the vocabulary id the
  // map gives it.
  const uint32_t width = c.headRows ? c.headRows : c.vocabulary;
  std::vector<uint32_t> idMap(width);
  std::iota(idMap.begin(), idMap.end(), 0U);
  DraftHeadMap head{};
  if (c.headRows) {
    std::vector<uint32_t> all(c.vocabulary);
    std::iota(all.begin(), all.end(), 0U);
    for (uint32_t index = 0; index < width; ++index)
      std::swap(all[index], all[index + random.next() % (c.vocabulary - index)]);
    std::copy_n(all.begin(), width, idMap.begin());
    std::sort(idMap.begin(), idMap.end());
    head = {allocate(backend, uint64_t{width} * sizeof(uint32_t)), width};
    std::copy(idMap.begin(), idMap.end(), static_cast<uint32_t *>(head.ids.contents()));
  }

  MetalBuffer logits = allocate(backend, uint64_t{rows} * width * sizeof(float));
  auto *logitRows = static_cast<float *>(logits.contents());
  const std::array patterns{Pattern::Peaked, Pattern::Uniform, Pattern::Ties,
                            Pattern::Sparse};
  for (uint32_t row = 0; row < rows; ++row)
    fillRow(logitRows + uint64_t{row} * width, width,
            patterns[(row / kRows + row % kRows) % patterns.size()], random);
  const MetalBuffer selectorHidden =
      randomBfloat(backend, uint64_t{rows} * kRank, random, 0.1F);
  const DraftCodebooks codebooks{
      randomBfloat(backend, uint64_t{c.vocabulary} * kRank, random, 0.1F),
      randomBfloat(backend, uint64_t{c.vocabulary} * kRank, random, 0.1F)};
  DraftSelectorBuffers buffers{
      logits,
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      selectorHidden,
      allocate(backend,
               uint64_t{c.lanes} * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, uint64_t{positions} * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  auto *uniforms = static_cast<float *>(buffers.uniforms.contents());
  for (uint32_t index = 0; index < c.lanes * SPLASH_SAMPLING_UNIFORMS; ++index)
    uniforms[index] = (random.unit() + 1.0F) * 0.5F;
  std::vector<uint32_t> anchors(c.lanes);
  std::vector<SamplingPolicy> policies(c.lanes);
  for (uint32_t lane = 0; lane < c.lanes; ++lane) {
    anchors[lane] = random.next() % c.vocabulary;
    policies[lane] = SamplingPolicy{16, c.sampling ? 0.8F : 0.0F, 1.0F, false};
  }

  CommandGraph graph;
  selector.add(graph, buffers, codebooks, anchors, policies, head);
  require(graph.dispatches().size() == (c.headRows ? 4U : 3U),
          "draft selector dispatch count changed");
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  const auto *candidates =
      static_cast<const uint32_t *>(buffers.candidates.contents());
  const auto *unary = static_cast<const float *>(buffers.unary.contents());
  const auto *tokens =
      static_cast<const uint32_t *>(buffers.proposedTokens.contents());
  const auto *probabilities =
      static_cast<const float *>(buffers.proposalProbabilities.contents());
  const auto *hidden =
      static_cast<const uint16_t *>(selectorHidden.contents());
  const auto *predecessors =
      static_cast<const uint16_t *>(codebooks.predecessor.contents());
  const auto *successors =
      static_cast<const uint16_t *>(codebooks.successor.contents());

  for (uint32_t lane = 0; lane < c.lanes; ++lane) {
    uint32_t predecessor = anchors[lane];
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t global = lane * kPositions + position;
      const float *row =
          logitRows + (uint64_t{lane} * kRows + position + 1) * width;
      const auto expected = referenceTop16(row, width);
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t id = candidates[global * kCandidates + rank];
        const float value = unary[global * kCandidates + rank];
        if (rank < expected.size()) {
          require(id == idMap[expected[rank]] && value == row[expected[rank]],
                  "draft top-16 candidates differ from the exact sorted order");
        } else {
          require(id == 0xFFFFFFFFU && value == -INFINITY,
                  "draft top-16 padding lost the empty sentinel");
        }
      }

      // Scores as the kernel defines them, in double.
      std::array<double, kCandidates> scores{};
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t candidate =
            std::min(candidates[global * kCandidates + rank], c.vocabulary - 1);
        double edge = 0.0;
        for (uint32_t dim = 0; dim < kRank; ++dim) {
          edge += double(tuning::bf16ToFloat(predecessors[uint64_t{predecessor} * kRank + dim])) *
                  tuning::bf16ToFloat(hidden[(uint64_t{lane} * kRows + position + 1) * kRank + dim]) *
                  tuning::bf16ToFloat(successors[uint64_t{candidate} * kRank + dim]);
        }
        scores[rank] = double(unary[global * kCandidates + rank]) + edge;
      }
      const uint32_t token = tokens[global];
      uint32_t selected = kCandidates;
      for (uint32_t rank = 0; rank < kCandidates; ++rank)
        if (candidates[global * kCandidates + rank] == token)
          selected = selected == kCandidates ? rank : selected;
      require(selected < kCandidates,
              "draft selector proposed a token outside its candidates");
      if (c.sampling) {
        // The lane's temperature times tau, as the host multiplies it.
        const Proposal reference = referenceProposal(scores, double(0.8F * c.tau), c.topP);
        const float uniform = uniforms[lane * SPLASH_SAMPLING_UNIFORMS +
                                       SPLASH_UNIFORM_PROPOSALS + position];
        std::array<double, kCandidates> cumulative{};
        uint32_t expectedSelection = c.topP < 1.0F ? reference.lastKept : kCandidates - 1;
        bool found = false;
        for (uint32_t rank = 0; rank < kCandidates; ++rank) {
          const float probability = probabilities[global * kCandidates + rank];
          require(reference.ambiguous || std::fabs(probability - reference.probabilities[rank]) < 1e-4,
                  "draft selector probabilities diverged from the reference proposal");
          cumulative[rank] = (rank ? cumulative[rank - 1] : 0.0) + reference.probabilities[rank];
          if (!found && (c.topP == 1.0F || reference.probabilities[rank] > 0.0) &&
              cumulative[rank] > uniform) {
            expectedSelection = rank;
            found = true;
          }
        }
        require(c.topP == 1.0F || probabilities[global * kCandidates + selected] > 0.0F,
                "draft selector drew a candidate its nucleus does not keep");
        // A draw within fp32 rounding of the crossed boundary may go either way.
        const double boundary =
            std::fabs(cumulative[std::min(selected, expectedSelection)] - uniform);
        require(reference.ambiguous || selected == expectedSelection || boundary < 1e-5,
                "draft selector drew a different candidate than the reference");
      } else {
        uint32_t expectedSelection = 0;
        for (uint32_t rank = 1; rank < kCandidates; ++rank)
          if (scores[rank] > scores[expectedSelection])
            expectedSelection = rank;
        // fp32 accumulation over 256 products of magnitude 1e-3 stays far
        // below this margin; only an exact tie could legitimately differ.
        require(selected == expectedSelection ||
                    std::fabs(scores[selected] - scores[expectedSelection]) < 1e-5,
                "draft selector picked a different greedy candidate");
      }
      predecessor = token;
    }
  }
}

// Each buffer the selector reaches, at its extent and one element short, for
// three lanes of which the second samples: every lane's eight query rows of
// logits and selector hidden rows, the workspaces of its seven positions,
// each codebook's row of every token, and the sampled lanes' proposal
// uniforms and probabilities up to the second lane's.
void bufferExtents(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003, lanes = 3, sampledLanes = 2;
  const DraftSelector selector(vocabulary);
  const auto workspace = DraftSelector::workspace(lanes * kPositions);
  const DraftSelectorBuffers buffers{
      allocate(backend, uint64_t{lanes} * kRows * vocabulary * sizeof(float)),
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      allocate(backend, uint64_t{lanes} * kRows * kRank * 2),
      allocate(backend, uint64_t{lanes} * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, uint64_t{lanes} * kPositions * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  const DraftCodebooks codebooks{allocate(backend, uint64_t{vocabulary} * kRank * 2),
                                 allocate(backend, uint64_t{vocabulary} * kRank * 2)};
  const std::array<uint32_t, lanes> anchors{1, 2, 3};
  const std::array policies{SamplingPolicy{}, SamplingPolicy{16, 0.8F, 1.0F}, SamplingPolicy{}};
  const uint64_t candidates = uint64_t{lanes} * kPositions * kCandidates;
  for (const auto &[member, bytes, element, what] :
       std::initializer_list<std::tuple<MetalBuffer DraftSelectorBuffers::*, uint64_t, uint64_t, const char *>>{
           {&DraftSelectorBuffers::logits, uint64_t{lanes} * kRows * vocabulary * 4, 4, "draft logits"},
           {&DraftSelectorBuffers::partialIds, candidates * SPLASH_DRAFT_SAMPLING_SHARDS * 4, 4,
            "draft selector partial id"},
           {&DraftSelectorBuffers::partialValues, candidates * (SPLASH_DRAFT_SAMPLING_SHARDS + kCandidates) * 4, 4,
            "draft selector partial value"},
           {&DraftSelectorBuffers::candidates, candidates * 4, 4, "draft candidate"},
           {&DraftSelectorBuffers::unary, candidates * 4, 4, "draft candidate score"},
           {&DraftSelectorBuffers::selectorHidden, uint64_t{lanes} * kRows * kRank * 2, 2, "draft selector hidden"},
           {&DraftSelectorBuffers::proposedTokens, uint64_t{lanes} * kPositions * 4, 4, "proposed token"},
           {&DraftSelectorBuffers::uniforms,
            ((sampledLanes - 1) * SPLASH_SAMPLING_UNIFORMS + SPLASH_UNIFORM_PROPOSALS + kPositions) * 4, 4,
            "proposal uniform"},
           {&DraftSelectorBuffers::proposalProbabilities, uint64_t{sampledLanes} * kPositions * kCandidates * 4, 4,
            "proposal probability"}})
    requireExtent(backend, buffers.*member, bytes, element, what, [&](CommandGraph &graph, const MetalBuffer &buffer) {
      DraftSelectorBuffers changed = buffers;
      changed.*member = buffer;
      selector.add(graph, changed, codebooks, anchors, policies);
    });
  for (const auto &[member, what] :
       {std::pair{&DraftCodebooks::predecessor, "draft predecessor codebook"},
        std::pair{&DraftCodebooks::successor, "draft successor codebook"}})
    requireExtent(backend, codebooks.*member, uint64_t{vocabulary} * kRank * 2, 2, what,
                  [&](CommandGraph &graph, const MetalBuffer &buffer) {
                    DraftCodebooks changed = codebooks;
                    changed.*member = buffer;
                    selector.add(graph, buffers, changed, anchors, policies);
                  });
}

void invalidRequests(MetalBackend &backend) {
  rejects([] { DraftSelector(0); }, "invalid draft selector vocabulary",
          "a selector over no vocabulary was accepted");
  const DraftSelector selector(1024);
  const auto workspace = DraftSelector::workspace(kPositions);
  const DraftSelectorBuffers buffers{
      allocate(backend, uint64_t{kRows} * 1024 * sizeof(float)),
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      allocate(backend, uint64_t{kRows} * kRank * 2),
      allocate(backend, SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
      allocate(backend, kPositions * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  const DraftCodebooks codebooks{allocate(backend, uint64_t{1024} * kRank * 2),
                                 allocate(backend, uint64_t{1024} * kRank * 2)};
  const std::array<uint32_t, 2> anchors{1, 2};
  const std::array<SamplingPolicy, 1> policies{SamplingPolicy{}};
  CommandGraph graph;
  rejects([&] { selector.add(graph, buffers, codebooks, anchors, policies); },
          "invalid draft selector batch", "anchors without a policy each were accepted");
  require(graph.empty(), "invalid draft selector request encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: draft-selector METALLIB");
    MetalBackend backend(argv[1]);
    invalidRequests(backend);
    bufferExtents(backend);
    for (const uint32_t vocabulary : {248320U, 1003U, 270005U}) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        // Upstream's selector (tau and top-p 1), then the fork defaults
        // (0.85, 0.99), which greedy lanes ignore, and a narrow nucleus.
        runCase(backend, {vocabulary, lanes, false});
        runCase(backend, {vocabulary, lanes, true});
        runCase(backend, {vocabulary, lanes, false, 0.85F, 0.99F});
        runCase(backend, {vocabulary, lanes, true, 0.85F, 0.99F});
        runCase(backend, {vocabulary, lanes, true, 0.85F, 0.6F});
      }
    }
    // SPLASH_DRAFT_HEAD_IDS: a restricted head of 512 and of 98,304 rows.
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
      runCase(backend, {1003U, lanes, false, 0.85F, 0.99F, 512U});
      runCase(backend, {1003U, lanes, true, 0.85F, 0.99F, 512U});
      runCase(backend, {248320U, lanes, true, 0.85F, 0.99F, 98304U});
    }
    std::cout << "draft_selector_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_selector_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
