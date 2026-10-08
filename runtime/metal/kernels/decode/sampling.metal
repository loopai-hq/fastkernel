// Modified by meowkernels.
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/split_reduce.h"

// The rows of the target policy. A greedy lane takes each row's argmax. A
// sampled lane draws from each row's min-p/top-k/top-p distribution over the
// whole vocabulary: a sharded scan sums the row's softmax denominator, one
// group per row finds the last token the distribution keeps in the order of
// the logits (logit descending, id ascending) without sorting the
// vocabulary, and the draw takes one pass over the kept tokens in id order,
// shared by the row's kVocabularyGroups groups: each sums ranges of the
// vocabulary, and the one that finishes last (split_arrive_last) draws.
// Every reduction runs in a fixed order, so a row selects the same token on
// every run. The same kernels select the first token after a prompt and the
// verify rows (TargetSamplingParams): each dispatch covers the selected rows
// of every lane, and the groups of the other policy's lanes return at once.

// The row a lane selects from: its logits, the tokens it admits (its
// constraint mask row, less the stop tokens when the lane ignores
// end-of-sequence) and, for a sampled row whose maximum is known, their
// softmax weights.
struct TargetRow {
  device const float *logits;
  device const uint *mask;
  bool constrained;
  bool exclude_stop;
  uint stop_token_0;
  uint stop_token_1;
  uint vocabulary;
  float temperature;
  float maximum;

  bool admits(uint token) const {
    if (constrained && (mask[token / 32] & (1u << (token % 32))) == 0)
      return false;
    return !exclude_stop || (token != stop_token_0 && token != stop_token_1);
  }
  float weight(float value) const {
    return exp((value - maximum) / temperature);
  }
};

// The lane of selected row s, and whether it samples.
inline uint selected_lane(constant TargetSamplingParams &params, uint s) {
  return s / params.rows;
}
inline bool lane_samples(constant TargetSamplingParams &params, uint s) {
  return (params.sampling_mask & (1u << selected_lane(params, s))) != 0;
}

// Selected row s of a dispatch (TargetSamplingParams).
inline TargetRow selected_row(device const float *logits,
                              device const uint *token_mask,
                              constant TargetSamplingParams &params, uint s) {
  const uint lane = selected_lane(params, s);
  const uint index = s % params.rows;
  return {logits + (ulong(lane) * SPLASH_TARGET_VERIFY_ROWS +
                    params.logits_row + index) *
                       params.vocabulary,
          token_mask + (ulong(lane) * (SPLASH_TARGET_VERIFY_ROWS + 1) +
                        params.mask_row + index) *
                           params.mask_words,
          (params.constrained_mask & (1u << lane)) != 0,
          (params.exclude_stop_mask & (1u << lane)) != 0,
          params.stop_token_0,
          params.stop_token_1,
          params.vocabulary,
          params.temperature[lane],
          0.0f};
}

// Shares of a row's softmax denominator combined in order: their largest
// maximum, their sums rescaled to it and their admitted counts. A group
// combines its simdgroups' shares, and a row's search its shards'.
template <class Shares>
inline TargetShardMass merge_masses(Shares shares, uint count,
                                    float temperature) {
  TargetShardMass total{-FLT_MAX, 0.0f, 0};
  for (uint index = 0; index < count; ++index)
    total.maximum = max(total.maximum, shares[index].maximum);
  for (uint index = 0; index < count; ++index) {
    const TargetShardMass share = shares[index];
    if (share.sum > 0.0f)
      total.sum +=
          share.sum * exp((share.maximum - total.maximum) / temperature);
    total.admitted += share.admitted;
  }
  return total;
}

// One shard's share of a sampled row's softmax denominator: each thread
// keeps a running maximum and sum of its admitted tokens' weights (an online
// softmax), combined over the group in a fixed order. The running maximum
// starts below every finite logit, so a -inf logit weighs nothing.
inline void shard_mass(TargetRow row, uint shard,
                       device TargetShardMass &partial,
                       threadgroup TargetShardMass *group_masses,
                       uint thread_index, uint lane, uint simd_group) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  TargetShardMass mass{-FLT_MAX, 0.0f, 0};
  for (uint token = shard * 256 + thread_index; token < row.vocabulary;
       token += Shards * 256) {
    if (!row.admits(token))
      continue;
    const float value = row.logits[token];
    if (value > mass.maximum) {
      mass.sum =
          mass.sum * exp((mass.maximum - value) / row.temperature) + 1.0f;
      mass.maximum = value;
    } else {
      mass.sum += exp((value - mass.maximum) / row.temperature);
    }
    ++mass.admitted;
  }
  const float simd_maximum = simd_max(mass.maximum);
  const float scaled =
      mass.sum > 0.0f
          ? mass.sum * exp((mass.maximum - simd_maximum) / row.temperature)
          : 0.0f;
  const float simd_total = simd_sum(scaled);
  const uint simd_admitted = simd_sum(mass.admitted);
  if (lane == 0)
    group_masses[simd_group] = {simd_maximum, simd_total, simd_admitted};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0)
    partial = merge_masses(group_masses, 8, row.temperature);
}

// One shard of each selected row of the sampled lanes.
kernel void decode_sample_mass_sharded(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device TargetShardMass *partial_masses [[buffer(2)]],
    constant TargetSamplingParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup TargetShardMass group_masses[8];
  const uint s = group / SPLASH_TARGET_SAMPLING_SHARDS;
  if (!lane_samples(params, s))
    return;
  shard_mass(selected_row(logits, token_mask, params, s),
             group % SPLASH_TARGET_SAMPLING_SHARDS, partial_masses[group],
             group_masses, thread_index, lane, simd_group);
}

// The groups that search and draw a sampled row.
constant constexpr uint kVocabularyThreads = SPLASH_TARGET_VOCABULARY_THREADS;
constant constexpr uint kVocabularySimdgroups = kVocabularyThreads / 32;
constant constexpr uint kVocabularyGroups = SPLASH_TARGET_VOCABULARY_GROUPS;
// The draw's ranges of the vocabulary; the last group loads them one per
// thread.
constant constexpr uint kVocabularyRanges = SPLASH_TARGET_VOCABULARY_RANGES;
static_assert(kVocabularyRanges <= kVocabularyThreads,
              "a group holds one draw range per thread");
// Pivots per pass: each pass splits a bracket sixteen ways. In logit order
// kKeyPivots of them split its keys evenly and the others follow the logits'
// scale (place_pivots).
constant constexpr uint kPivots = 15;
constant constexpr uint kKeyPivots = 4;
constant constexpr uint kDraftCandidates = SPLASH_DRAFT_CANDIDATES;

// The order of the logits as an unsigned key: a larger logit has a larger
// key, and -0 and +0, which compare equal, share one. Every logit's key is
// above kNoKey.
inline uint logit_key(float value) {
  uint bits = as_type<uint>(value);
  bits = (bits << 1) ? bits : 0u;
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}
inline float key_logit(uint key) {
  return as_type<float>((key & 0x80000000u) ? key & 0x7fffffffu : ~key);
}
constant constexpr uint kNoKey = 0u;
// The key of -infinity, at or below that of every admitted logit.
constant constexpr uint kLowestKey = 0x007fffffu;

// A position in the order of the logits: the tokens at or before it have a
// larger key, or the same key and an id at most last. {kNoKey, 0} lies
// after every token.
struct OrderBoundary {
  uint key;
  uint last;
};

inline bool at_or_before(uint key, uint token, OrderBoundary boundary) {
  return key > boundary.key || (key == boundary.key && token <= boundary.last);
}

// The orders a search runs in. The logit order is the order of the logits.
// Among the tokens tied at one logit, the tie order puts a smaller id first;
// every other token has kNoKey.
struct LogitOrder {
  static constexpr constant bool by_logit = true;
  uint key(float value, uint) const { return logit_key(value); }
  float logit(uint key) const { return key_logit(key); }
};
struct TieOrder {
  uint tied_key;
  uint vocabulary;
  uint key(float value, uint token) const {
    return logit_key(value) == tied_key ? vocabulary - token : kNoKey;
  }
  static constexpr constant bool by_logit = false;
  float logit(uint) const { return key_logit(tied_key); }
};

// What a search keeps: the tokens in order until their count, or their mass
// (the sum of their weights), exceeds the target. A count search measures
// counts alone (top-k, and the tie order's first ids), a mass search counts
// and masses (top-p).
struct CountTarget {
  static constexpr constant bool by_mass = false;
  uint count;

  bool exceeded(uint measured_count, float) const {
    return measured_count > count;
  }
};
struct MassTarget {
  static constexpr constant bool by_mass = true;
  float mass;

  bool exceeded(uint, float measured_mass) const {
    return measured_mass > mass;
  }
};

// A key range [lo, hi) and the count and mass of the tokens with at least
// each end's key: lo's exceed the search target, hi's do not. A count
// search narrows the counts alone and leaves the masses as they came in.
struct Bracket {
  uint lo;
  uint hi;
  uint lo_count;
  uint hi_count;
  float lo_mass;
  float hi_mass;

  uint tokens() const { return lo_count - hi_count; }
};

// The last token a search keeps, and the count and mass of the tokens at or
// before it. A ranked selection leaves those tokens in the scratch keys and
// ids, in order (resolve).
struct Selection {
  OrderBoundary last;
  uint count;
  float mass;
  bool ranked;
};

struct VocabularyScratch {
  uint counts[kVocabularySimdgroups][kPivots];
  float masses[kVocabularySimdgroups][kPivots];
  uint total_counts[kPivots];
  float total_masses[kPivots];
  atomic_uint gathered;
  uint keys[kVocabularyThreads];
  uint ids[kVocabularyThreads];
  Selection selection;
};

struct DrawScratch {
  uint arrival;
  // Per range of the draw: the kept weight of its tokens other than the
  // draft's candidates, one past the last of those with weight, and its
  // weight in the draw.
  float range_rest[kVocabularyRanges];
  uint range_last[kVocabularyRanges];
  float range_weights[kVocabularyRanges];
  // The draft's candidates, each listed once: its draft probability, kept
  // weight, weight in the draw and range.
  uint draft_ids[kDraftCandidates];
  float draft_probabilities[kDraftCandidates];
  float draft_weights[kDraftCandidates];
  float draft_draw_weights[kDraftCandidates];
  uint draft_ranges[kDraftCandidates];
  float kept_mass;
  uint drawn_range;
  float drawn_target;
  uint drawn_token;
};

// Pivots inside [lo, hi) that split the bracket. The tie order splits its
// keys evenly. In logit order the last kKeyPivots do, so a pass keeps at
// most a fifth of the bracket's keys, rounded up, and a search ends within
// fourteen passes whatever the logits, temperature or penalties; the others
// follow the logits' scale, which ends most searches within a few: geometric
// distances below the bracket's top, in units of the temperature, while it
// is open at the bottom, and even steps in logit value once it is not.
template <class Order>
inline void place_pivots(Order order, Bracket bracket, float temperature,
                         thread uint (&pivots)[kPivots]) {
  const ulong width = bracket.hi - bracket.lo;
  if (!order.by_logit) {
    for (uint pivot = 0; pivot < kPivots; ++pivot)
      pivots[pivot] = bracket.lo + uint(width * (pivot + 1) / (kPivots + 1));
    return;
  }
  constexpr uint kScaled = kPivots - kKeyPivots;
  const float top = key_logit(bracket.hi - 1);
  const float bottom = key_logit(bracket.lo);
  const bool open = bracket.lo == kLowestKey;
  for (uint pivot = 0; pivot < kScaled; ++pivot) {
    const float value =
        open ? top - temperature * 0.5f * exp2(float(kScaled - 1 - pivot))
             : bottom + (top - bottom) * float(pivot + 1) / float(kScaled + 1);
    pivots[pivot] = clamp(logit_key(value), bracket.lo, bracket.hi - 1);
  }
  for (uint pivot = 0; pivot < kKeyPivots; ++pivot)
    pivots[kScaled + pivot] =
        bracket.lo + uint(width * (pivot + 1) / (kKeyPivots + 1));
}

// The count and, with Masses, the mass of the admitted tokens at or before
// the floor whose keys are at least each pivot's, summed per thread in token
// order and then over the group in a fixed order. Counts alone weigh no
// token.
template <bool Masses, uint Pivots, class Order>
inline void measure_pivots(TargetRow row, Order order, OrderBoundary floor,
                           thread const uint (&pivots)[Pivots],
                           threadgroup VocabularyScratch &scratch,
                           uint thread_index, uint lane, uint simd_group) {
  uint counts[Pivots];
  float masses[Pivots];
  uint lowest = pivots[0];
  for (uint pivot = 0; pivot < Pivots; ++pivot) {
    counts[pivot] = 0;
    if constexpr (Masses)
      masses[pivot] = 0.0f;
    lowest = min(lowest, pivots[pivot]);
  }
  for (uint token = thread_index; token < row.vocabulary;
       token += kVocabularyThreads) {
    const float value = row.logits[token];
    const uint key = order.key(value, token);
    // Tokens below every pivot add nothing.
    if (key < lowest || !row.admits(token) ||
        !at_or_before(logit_key(value), token, floor))
      continue;
    if constexpr (Masses) {
      const float weight = row.weight(value);
      for (uint pivot = 0; pivot < Pivots; ++pivot) {
        const bool above = key >= pivots[pivot];
        counts[pivot] += above ? 1u : 0u;
        masses[pivot] += above ? weight : 0.0f;
      }
    } else {
      for (uint pivot = 0; pivot < Pivots; ++pivot)
        counts[pivot] += key >= pivots[pivot] ? 1u : 0u;
    }
  }
  for (uint pivot = 0; pivot < Pivots; ++pivot) {
    const uint count = simd_sum(counts[pivot]);
    if (lane == 0)
      scratch.counts[simd_group][pivot] = count;
    if constexpr (Masses) {
      const float mass = simd_sum(masses[pivot]);
      if (lane == 0)
        scratch.masses[simd_group][pivot] = mass;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < Pivots) {
    uint count = 0;
    for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
      count += scratch.counts[simd][thread_index];
    scratch.total_counts[thread_index] = count;
    if constexpr (Masses) {
      float mass = 0.0f;
      for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
        mass += scratch.masses[simd][thread_index];
      scratch.total_masses[thread_index] = mass;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The count and mass of the admitted tokens at or before the floor whose
// logits' keys are at least key, summed as measure_pivots sums a pivot's.
struct Measure {
  uint count;
  float mass;
};
inline Measure measure_at_or_above(TargetRow row, OrderBoundary floor,
                                   uint key,
                                   threadgroup VocabularyScratch &scratch,
                                   uint thread_index, uint lane,
                                   uint simd_group) {
  const uint pivots[1] = {key};
  measure_pivots<true>(row, LogitOrder{}, floor, pivots, scratch,
                       thread_index, lane, simd_group);
  const Measure measure{scratch.total_counts[0], scratch.total_masses[0]};
  // Every thread reads the totals before the next pass rewrites them.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return measure;
}

// Narrows the bracket until it holds at most one token per thread or a
// single key.
template <class Order, class Target>
inline Bracket narrow(TargetRow row, Order order, OrderBoundary floor,
                      Bracket bracket, Target target,
                      threadgroup VocabularyScratch &scratch,
                      uint thread_index, uint lane, uint simd_group) {
  while (bracket.tokens() > kVocabularyThreads &&
         bracket.hi - bracket.lo > 1) {
    uint pivots[kPivots];
    place_pivots(order, bracket, row.temperature, pivots);
    measure_pivots<Target::by_mass>(row, order, floor, pivots, scratch,
                                    thread_index, lane, simd_group);
    // The highest pivot whose measure exceeds the target and the lowest one
    // whose measure does not bound the new bracket. A pivot at lo never
    // becomes hi: lo's measure may come from another sum (the shards' masses,
    // or the top-k selection's), and rounding must not empty the bracket.
    for (uint pivot = 0; pivot < kPivots; ++pivot) {
      const uint count = scratch.total_counts[pivot];
      const float mass = Target::by_mass ? scratch.total_masses[pivot] : 0.0f;
      if (target.exceeded(count, mass)) {
        if (pivots[pivot] >= bracket.lo) {
          bracket.lo = pivots[pivot];
          bracket.lo_count = count;
          if (Target::by_mass)
            bracket.lo_mass = mass;
        }
      } else if (pivots[pivot] > bracket.lo && pivots[pivot] < bracket.hi) {
        bracket.hi = pivots[pivot];
        bracket.hi_count = count;
        if (Target::by_mass)
          bracket.hi_mass = mass;
      }
    }
    // Every thread reads the totals before the next pass rewrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return bracket;
}

// The last token the target keeps in a bracket of at most one token per
// thread: its tokens are gathered, put in order by rank, and added to the
// measure at hi one at a time until it exceeds the target. A count search in
// logit order measures no masses while it narrows: its gather sums the mass
// at hi as measure_pivots would have, or, when every token from lo up fits
// the group, gathers them all and adds them from the first, which leaves the
// tokens it keeps ranked in the scratch (Selection::ranked).
template <class Order, class Target>
inline Selection resolve(TargetRow row, Order order, OrderBoundary floor,
                         Bracket bracket, Target target,
                         threadgroup VocabularyScratch &scratch,
                         uint thread_index, uint lane, uint simd_group) {
  constexpr bool unmeasured = Order::by_logit && !Target::by_mass;
  const bool ranked = unmeasured && bracket.lo_count <= kVocabularyThreads;
  const bool measures = unmeasured && !ranked;
  if (thread_index == 0)
    atomic_store_explicit(&scratch.gathered, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float above = 0.0f;
  for (uint token = thread_index; token < row.vocabulary;
       token += kVocabularyThreads) {
    const float value = row.logits[token];
    const uint key = order.key(value, token);
    if (key < bracket.lo || !row.admits(token) ||
        !at_or_before(logit_key(value), token, floor))
      continue;
    if (!ranked && key >= bracket.hi) {
      if (measures)
        above += row.weight(value);
      continue;
    }
    const uint slot = atomic_fetch_add_explicit(&scratch.gathered, 1u,
                                                memory_order_relaxed);
    if (slot < kVocabularyThreads) {
      scratch.keys[slot] = key;
      scratch.ids[slot] = token;
    }
  }
  if (measures) {
    above = simd_sum(above);
    if (lane == 0)
      scratch.masses[simd_group][0] = above;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint gathered = min(atomic_load_explicit(&scratch.gathered,
                                                 memory_order_relaxed),
                            kVocabularyThreads);
  uint key = 0;
  uint id = 0;
  uint rank = 0;
  if (thread_index < gathered) {
    key = scratch.keys[thread_index];
    id = scratch.ids[thread_index];
    for (uint other = 0; other < gathered; ++other) {
      const uint other_key = scratch.keys[other];
      rank += other_key > key || (other_key == key && scratch.ids[other] < id)
                  ? 1u
                  : 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < gathered) {
    scratch.keys[rank] = key;
    scratch.ids[rank] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    uint count = ranked ? 0 : bracket.hi_count;
    float mass = ranked ? 0.0f : bracket.hi_mass;
    if (measures) {
      mass = 0.0f;
      for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
        mass += scratch.masses[simd][0];
    }
    // Rounding may leave a mass target unreached: keep the whole bracket.
    Selection selection{{bracket.lo, 0xffffffffu},
                        bracket.lo_count,
                        Target::by_mass ? bracket.lo_mass : mass,
                        ranked};
    for (uint index = 0; index < gathered; ++index) {
      count += 1;
      mass += row.weight(order.logit(scratch.keys[index]));
      selection = {{scratch.keys[index], scratch.ids[index]}, count, mass,
                   ranked};
      if (target.exceeded(count, mass))
        break;
    }
    scratch.selection = selection;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch.selection;
}

// The last token the target keeps among the admitted tokens at or before
// the floor, starting from a bracket in logit order.
template <class Target>
inline Selection select_last(TargetRow row, OrderBoundary floor,
                             Bracket bracket, Target target,
                             threadgroup VocabularyScratch &scratch,
                             uint thread_index, uint lane, uint simd_group) {
  bracket = narrow(row, LogitOrder{}, floor, bracket, target, scratch,
                   thread_index, lane, simd_group);
  if (bracket.tokens() <= kVocabularyThreads)
    return resolve(row, LogitOrder{}, floor, bracket, target, scratch,
                   thread_index, lane, simd_group);
  // More tokens tie at one logit than a group gathers: the target keeps its
  // first ones by id, each of the same weight, after the mass above them,
  // which a count search measures in one more pass.
  const uint ties = bracket.tokens();
  const float weight = row.weight(key_logit(bracket.lo));
  float above;
  uint keep;
  if constexpr (Target::by_mass) {
    above = bracket.hi_mass;
    keep = 1u + uint(clamp((target.mass - above) / weight, 0.0f,
                           float(ties - 1)));
  } else {
    above = measure_at_or_above(row, floor, bracket.hi, scratch, thread_index,
                                lane, simd_group)
                .mass;
    keep = target.count + 1 - bracket.hi_count;
  }
  uint last = bracket.lo == floor.key ? floor.last : 0xffffffffu;
  if (keep < ties) {
    const TieOrder order{bracket.lo, row.vocabulary};
    const CountTarget first{keep - 1};
    Bracket tied{1, row.vocabulary + 1, ties, 0, 0.0f, 0.0f};
    tied = narrow(row, order, floor, tied, first, scratch, thread_index, lane,
                  simd_group);
    last = resolve(row, order, floor, tied, first, scratch, thread_index, lane,
                   simd_group)
               .last.last;
  }
  return {{bracket.lo, last}, bracket.hi_count + keep,
          above + float(keep) * weight, false};
}

// The last token of the row's distribution: the tokens that weigh at least
// min_p of the heaviest, then the top_k of those, then within those the top_p
// nucleus, whose mass is measured against the mass of what the two cuts
// before it keep. A top-k selection the scratch holds ranked is walked for
// the nucleus in that order, with the sums it was selected with; otherwise
// the nucleus is searched for. A row that keeps every admitted token ends at
// {kNoKey, 0}.
inline OrderBoundary distribution_end(TargetRow row, float min_p, uint top_k,
                                      float top_p, float mass, uint admitted,
                                      threadgroup VocabularyScratch &scratch,
                                      uint thread_index, uint lane,
                                      uint simd_group) {
  OrderBoundary end{kNoKey, 0};
  Bracket bracket{kLowestKey, logit_key(row.maximum) + 1, admitted, 0, mass,
                  0.0f};
  if (min_p > 0.0f) {
    // A token weighs min_p of the heaviest, which weighs 1, at the logit
    // -temperature * log(min_p) below the maximum. The tokens at or above
    // that logit stay, the heaviest always.
    const uint lowest = logit_key(
        min(row.maximum + row.temperature * log(min_p), row.maximum));
    const Measure kept = measure_at_or_above(row, end, lowest, scratch,
                                             thread_index, lane, simd_group);
    if (kept.count < admitted) {
      end = {lowest, 0xffffffffu};
      bracket.lo = lowest;
      bracket.lo_count = kept.count;
      bracket.lo_mass = kept.mass;
    }
  }
  if (top_k < bracket.lo_count) {
    // A min_p cut ends on a whole key, which the bracket bounds from below,
    // so this search needs no floor.
    const Selection top = select_last(row, {kNoKey, 0}, bracket,
                                      CountTarget{top_k - 1}, scratch,
                                      thread_index, lane, simd_group);
    end = top.last;
    if (top.ranked && top_p < 1.0f) {
      // Every thread reads the selection before the walk rewrites it.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (thread_index == 0) {
        // Rounding may leave the nucleus unreached: keep the whole top-k.
        const float nucleus = top_p * top.mass;
        float walked = 0.0f;
        for (uint index = 0; index < top.count; ++index) {
          walked += row.weight(key_logit(scratch.keys[index]));
          if (walked > nucleus) {
            end = {scratch.keys[index], scratch.ids[index]};
            break;
          }
        }
        scratch.selection.last = end;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      return scratch.selection.last;
    }
    bracket.lo = end.key;
    bracket.lo_count = top.count;
    bracket.lo_mass = top.mass;
  }
  if (top_p < 1.0f)
    end = select_last(row, end, bracket, MassTarget{top_p * bracket.lo_mass},
                      scratch, thread_index, lane, simd_group)
              .last;
  return end;
}

// A token's weight in the row's distribution: its softmax weight if the
// row admits it at or before the distribution's end, else 0.
inline float kept_weight(TargetRow row, OrderBoundary end, uint token) {
  if (token >= row.vocabulary || !row.admits(token))
    return 0.0f;
  const float value = row.logits[token];
  return at_or_before(logit_key(value), token, end) ? row.weight(value) : 0.0f;
}

// The draft's candidates in one range of the draw, one bit per staged entry.
inline uint range_candidates(uint range, threadgroup DrawScratch &scratch) {
  uint held = 0;
  for (uint index = 0; index < kDraftCandidates; ++index)
    held |= scratch.draft_ranges[index] == range ? 1u << index : 0u;
  return held;
}

// One range's weight in the draw: the kept weight of its tokens other than
// the draft's candidates, plus its candidates' draw weights.
inline float range_weight(uint range, threadgroup DrawScratch &scratch) {
  float weight = scratch.range_rest[range];
  for (uint held = range_candidates(range, scratch); held; held &= held - 1)
    weight += scratch.draft_draw_weights[ctz(held)];
  return weight;
}

// The last token of a range with weight in the draw, which the draw takes
// when rounding leaves its target unreached.
inline uint range_last(uint range, threadgroup DrawScratch &scratch) {
  uint after = scratch.range_last[range];
  for (uint held = range_candidates(range, scratch); held; held &= held - 1) {
    const uint index = ctz(held);
    if (scratch.draft_draw_weights[index] > 0.0f)
      after = max(after, scratch.draft_ids[index] + 1);
  }
  return after - 1;
}

// Draws the row's token, shared by its groups; returns true in the group that
// finishes last, with the drawn token and the draft token's probability. Each
// kept token weighs its softmax weight; with a draft, each of the draft's
// candidates weighs its weight less its draft probability times the kept
// mass, never below zero, so the draw follows the residual distribution
// acceptance corrects a rejected draft token from (the whole distribution
// when nothing remains). Each simdgroup of the row's groups sums one range of
// the vocabulary without the candidates; the last group computes their
// weights once, adds the ranges in order and walks the one that holds the
// draw with prefix sums. A range's weight and its walk add the same token
// weights, so the range the draw picks always has a token to draw.
inline bool vocabulary_draw(
    TargetRow row, OrderBoundary end, bool drafted, uint draft_token,
    device const uint *draft_ids, device const float *draft_probabilities,
    float uniform, device coherent(device) TargetVocabularyRange *ranges,
    device atomic_uint *arrivals, uint slice, threadgroup DrawScratch &scratch,
    uint thread_index, uint lane, uint simd_group, thread uint &drawn,
    thread float &draft_probability) {
  const uint range_tokens = (row.vocabulary + kVocabularyRanges * 32 - 1) /
                            (kVocabularyRanges * 32) * 32;
  if (thread_index < kDraftCandidates) {
    // A candidate the draft lists twice counts once, as lookups find its
    // first entry.
    uint id = drafted ? draft_ids[thread_index] : 0xffffffffu;
    for (uint earlier = 0; drafted && earlier < thread_index; ++earlier)
      id = draft_ids[earlier] == id ? 0xffffffffu : id;
    scratch.draft_ids[thread_index] = id;
    scratch.draft_ranges[thread_index] =
        id < row.vocabulary ? id / range_tokens : kVocabularyRanges;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint own = slice * kVocabularySimdgroups + simd_group;
  const uint begin = min(own * range_tokens, row.vocabulary);
  const uint finish = min(begin + range_tokens, row.vocabulary);
  const uint held = range_candidates(own, scratch);
  float rest = 0.0f;
  uint after = 0;
  for (uint first = begin; first < finish; first += 32) {
    const uint token = first + lane;
    float weight = token < finish ? kept_weight(row, end, token) : 0.0f;
    for (uint bits = held; bits; bits &= bits - 1)
      weight = scratch.draft_ids[ctz(bits)] == token ? 0.0f : weight;
    rest += weight;
    after = weight > 0.0f ? token + 1 : after;
  }
  rest = simd_sum(rest);
  after = simd_max(after);
  if (lane == 0)
    ranges[own] = {rest, after};
  if (!split_arrive_last(arrivals, kVocabularyGroups, thread_index,
                         &scratch.arrival))
    return false;
  if (thread_index < kVocabularyRanges) {
    const TargetVocabularyRange measured = ranges[thread_index];
    scratch.range_rest[thread_index] = measured.rest;
    scratch.range_last[thread_index] = measured.after;
  }
  if (thread_index < kDraftCandidates) {
    scratch.draft_probabilities[thread_index] =
        drafted ? draft_probabilities[thread_index] : 0.0f;
    scratch.draft_weights[thread_index] =
        kept_weight(row, end, scratch.draft_ids[thread_index]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float kept_mass = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range)
      kept_mass += scratch.range_rest[range];
    for (uint index = 0; index < kDraftCandidates; ++index)
      kept_mass += scratch.draft_weights[index];
    bool residual = false;
    for (uint range = 0; drafted && range < kVocabularyRanges; ++range)
      residual = residual || scratch.range_rest[range] > 0.0f;
    for (uint index = 0; drafted && index < kDraftCandidates; ++index) {
      const float left = max(scratch.draft_weights[index] -
                                 scratch.draft_probabilities[index] * kept_mass,
                             0.0f);
      scratch.draft_draw_weights[index] = left;
      residual = residual || left > 0.0f;
    }
    // Without a draft, or with nothing left of its residual, the draw
    // follows the distribution itself.
    for (uint index = 0; !residual && index < kDraftCandidates; ++index)
      scratch.draft_draw_weights[index] = scratch.draft_weights[index];
    scratch.kept_mass = kept_mass;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < kVocabularyRanges)
    scratch.range_weights[thread_index] = range_weight(thread_index, scratch);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range)
      total += scratch.range_weights[range];
    const float target = uniform * total;
    float cumulative = 0.0f;
    uint range_drawn = 0;
    float range_target = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range) {
      const float weight = scratch.range_weights[range];
      if (!(weight > 0.0f))
        continue;
      // Rounding may leave the target unreached: the last range with weight.
      range_drawn = range;
      range_target = target - cumulative;
      cumulative += weight;
      if (cumulative > target)
        break;
    }
    scratch.drawn_range = range_drawn;
    scratch.drawn_target = range_target;
    scratch.drawn_token = range_last(range_drawn, scratch);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const uint walked = scratch.drawn_range;
    const uint walk_begin = min(walked * range_tokens, row.vocabulary);
    const uint walk_finish = min(walk_begin + range_tokens, row.vocabulary);
    const uint walk_held = range_candidates(walked, scratch);
    const float target = scratch.drawn_target;
    float cumulative = 0.0f;
    for (uint first = walk_begin; first < walk_finish; first += 32) {
      const uint token = first + lane;
      float weight = token < walk_finish ? kept_weight(row, end, token) : 0.0f;
      for (uint bits = walk_held; bits; bits &= bits - 1) {
        const uint index = ctz(bits);
        if (scratch.draft_ids[index] == token)
          weight = scratch.draft_draw_weights[index];
      }
      const float prefix = simd_prefix_inclusive_sum(weight);
      const uint hit =
          simd_min(weight > 0.0f && cumulative + prefix > target ? lane : 32u);
      if (hit < 32) {
        if (lane == 0)
          scratch.drawn_token = first + hit;
        break;
      }
      cumulative += simd_broadcast(prefix, 31);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  drawn = scratch.drawn_token;
  draft_probability =
      drafted ? kept_weight(row, end, draft_token) / scratch.kept_mass : 0.0f;
  split_release(arrivals, thread_index);
  return true;
}

// Where a sampled row's distribution ends: its shards' masses, merged in
// shard order into the row's largest admitted logit, softmax denominator and
// admitted count, then the search. The record keeps the maximum and the
// end, which the draw reads.
inline void search_row(TargetRow row, device const TargetShardMass *masses,
                       float min_p, uint top_k, float top_p,
                       device TargetVocabularyRow &record,
                       threadgroup VocabularyScratch &scratch,
                       uint thread_index, uint lane, uint simd_group) {
  const TargetShardMass merged =
      merge_masses(masses, SPLASH_TARGET_SAMPLING_SHARDS, row.temperature);
  row.maximum = merged.maximum;
  const OrderBoundary end =
      distribution_end(row, min_p, top_k, top_p, merged.sum, merged.admitted,
                       scratch, thread_index, lane, simd_group);
  if (thread_index == 0)
    record = {merged.maximum, end.key, end.last, 0.0f};
}

// Where the distribution of each selected row of the sampled lanes ends, one
// group per row.
kernel void decode_sample_vocabulary_search(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const TargetShardMass *partial_masses [[buffer(2)]],
    device TargetVocabularyRow *vocabulary_rows [[buffer(3)]],
    constant TargetSamplingParams &params [[buffer(4)]],
    uint s [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup VocabularyScratch scratch;
  if (!lane_samples(params, s))
    return;
  const uint batch = selected_lane(params, s);
  search_row(selected_row(logits, token_mask, params, s),
             partial_masses + ulong(s) * SPLASH_TARGET_SAMPLING_SHARDS,
             params.min_p[batch], params.top_k[batch], params.top_p[batch],
             vocabulary_rows[s], scratch, thread_index, lane, simd_group);
}

// The draw of each selected row of the sampled lanes, kVocabularyGroups
// groups per row: the first token after a prompt, a verify row's draft
// token's probability and the correction a rejection takes or, for the last
// verify row, which follows the whole draft, its bonus token.
kernel void decode_sample_vocabulary_draw(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device TargetVocabularyRow *vocabulary_rows [[buffer(2)]],
    device const uint *input_tokens [[buffer(3)]],
    device const uint *draft_ids [[buffer(4)]],
    device const float *draft_probabilities [[buffer(5)]],
    device const float *uniforms [[buffer(6)]],
    device uint *tokens [[buffer(7)]],
    device coherent(device) TargetVocabularyRange *ranges [[buffer(8)]],
    device atomic_uint *arrivals [[buffer(9)]],
    constant TargetSamplingParams &params [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup DrawScratch scratch;
  const uint s = group / kVocabularyGroups;
  if (!lane_samples(params, s))
    return;
  const uint batch = selected_lane(params, s);
  const uint index = s % params.rows;
  device TargetVocabularyRow &record = vocabulary_rows[s];
  TargetRow row = selected_row(logits, token_mask, params, s);
  row.maximum = record.maximum;
  // A drafted row follows its draft token, the next verify input row.
  const bool drafted = index < params.drafted_rows;
  const ulong position =
      ulong(batch) * SPLASH_DRAFT_PROPOSAL_TOKENS + (drafted ? index : 0);
  uint token;
  float draft_probability;
  if (vocabulary_draw(
          row, {record.end_key, record.end_last}, drafted,
          drafted ? input_tokens[s + 1] : 0u,
          draft_ids + position * kDraftCandidates,
          draft_probabilities + position * kDraftCandidates,
          uniforms[ulong(batch) * SPLASH_SAMPLING_UNIFORMS + params.uniform],
          ranges + ulong(s) * kVocabularyRanges, arrivals + s,
          group % kVocabularyGroups, scratch, thread_index, lane, simd_group,
          token, draft_probability) &&
      thread_index == 0) {
    tokens[s] = token;
    if (drafted)
      record.draft_probability = draft_probability;
  }
}

// SPLASH_BLOCK_VERIFY: decode_sample_vocabulary_draw for a batch whose lanes
// all sample, which also exports each drafted row's target probabilities of
// its draft candidates and of every other token (TargetCandidateRow), from
// the kept weights and mass its draw computed once.
kernel void decode_sample_vocabulary_draw_block(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device TargetVocabularyRow *vocabulary_rows [[buffer(2)]],
    device const uint *input_tokens [[buffer(3)]],
    device const uint *draft_ids [[buffer(4)]],
    device const float *draft_probabilities [[buffer(5)]],
    device const float *uniforms [[buffer(6)]],
    device uint *tokens [[buffer(7)]],
    device coherent(device) TargetVocabularyRange *ranges [[buffer(8)]],
    device atomic_uint *arrivals [[buffer(9)]],
    device TargetCandidateRow *candidate_rows [[buffer(10)]],
    constant TargetSamplingParams &params [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup DrawScratch scratch;
  const uint s = group / kVocabularyGroups;
  if (!lane_samples(params, s))
    return;
  const uint batch = selected_lane(params, s);
  const uint index = s % params.rows;
  device TargetVocabularyRow &record = vocabulary_rows[s];
  TargetRow row = selected_row(logits, token_mask, params, s);
  row.maximum = record.maximum;
  const bool drafted = index < params.drafted_rows;
  const ulong position =
      ulong(batch) * SPLASH_DRAFT_PROPOSAL_TOKENS + (drafted ? index : 0);
  uint token;
  float draft_probability;
  if (!vocabulary_draw(
          row, {record.end_key, record.end_last}, drafted,
          drafted ? input_tokens[s + 1] : 0u,
          draft_ids + position * kDraftCandidates,
          draft_probabilities + position * kDraftCandidates,
          uniforms[ulong(batch) * SPLASH_SAMPLING_UNIFORMS + params.uniform],
          ranges + ulong(s) * kVocabularyRanges, arrivals + s,
          group % kVocabularyGroups, scratch, thread_index, lane, simd_group,
          token, draft_probability))
    return;
  if (thread_index == 0) {
    tokens[s] = token;
    if (drafted)
      record.draft_probability = draft_probability;
  }
  if (!drafted)
    return;
  // A candidate listed again was staged as no token, so it weighs 0.
  if (thread_index < kDraftCandidates)
    candidate_rows[s].probability[thread_index] =
        scratch.draft_weights[thread_index] / scratch.kept_mass;
  if (thread_index == 0) {
    float rest = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range)
      rest += scratch.range_rest[range];
    candidate_rows[s].rest = rest / scratch.kept_mass;
  }
}

// SPLASH_SAMPLER_TOPK32 (fork, default on): a sampled row whose lane keeps at
// most kTopTokens tokens by top-k and none by min-p ends among its first
// kTopTokens tokens in the order of the logits. In one pass over its shard,
// each group sums its share of the softmax denominator with shard_mass's
// arithmetic, token for token, and keeps its first kTopTokens tokens; the
// search merges the shards' lists and ends the row from them with
// distribution_end's own arithmetic, so its record, and every draw,
// acceptance and export that reads it, is decode_sample_vocabulary_search's.
// Where that is not certain (a kept logit that is not finite, or a top_p cut
// within 2^-16 of a cumulative mass that distribution_end may sum in another
// order), the row runs distribution_end itself. The lists sit in the row's
// draw ranges, which only its draw writes, later.
constant constexpr uint kTopTokens = SPLASH_SAMPLER_TOP_TOKENS;
static_assert(kTopTokens == 32, "a simdgroup holds one top token per lane");
static_assert(SPLASH_TARGET_SAMPLING_SHARDS * kTopTokens + kTopTokens <= kVocabularyThreads,
              "a search group holds every shard's list and the merged one");
constant constexpr uint kShardTopsPerRow =
    kVocabularyRanges * sizeof(TargetVocabularyRange) / sizeof(uint);
// The key of +infinity: only the NaNs' are above it.
constant constexpr uint kInfinityKey = 0xff800000u;

// A token before another in the order of the logits.
inline bool order_before(uint key, uint id, uint other_key, uint other_id) {
  return key > other_key || (key == other_key && id < other_id);
}

// Pulsar 1.0.0's lane-distributed sorted top-32, in the order of the
// logits: lane r holds rank r, an empty rank {kNoKey, ~0u}. Each lane offers
// one token, or none when !valid; the offers that pass rank 31 are inserted
// one at a time, so the list stays in registers. Every lane of the simdgroup
// must call it (an invalid one with valid = false): the shuffles read lanes.
inline void simd_top32_offer(thread uint &key, thread uint &id,
                             uint offered_key, uint offered_id, bool valid,
                             uint lane) {
  bool pending = valid;
  while (true) {
    // Every lane takes part in the shuffles, so rank 31's lane is always an
    // active source; behind `pending &&` a lane with nothing pending would
    // skip them.
    const uint last_key = simd_shuffle(key, ushort(31));
    const uint last_id = simd_shuffle(id, ushort(31));
    pending = pending && order_before(offered_key, offered_id, last_key, last_id);
    const uint first = simd_min(pending ? lane : 32u);
    if (first == 32u)
      return;
    const uint inserted_key = simd_shuffle(offered_key, ushort(first));
    const uint inserted_id = simd_shuffle(offered_id, ushort(first));
    const uint position = simd_sum(
        order_before(key, id, inserted_key, inserted_id) ? 1u : 0u);
    const uint up_key = simd_shuffle_up(key, ushort(1));
    const uint up_id = simd_shuffle_up(id, ushort(1));
    if (lane == position) {
      key = inserted_key;
      id = inserted_id;
    } else if (lane > position) {
      key = up_key;
      id = up_id;
    }
    if (lane == first)
      pending = false;
  }
}

// shard_mass's arithmetic for the one-pass kernel below, in two steps; a
// copy, so that upstream's kernels compile exactly as they did. One admitted
// token's step of a thread's online softmax:
inline void accumulate_mass(thread TargetShardMass &mass, float value,
                            float temperature) {
  if (value > mass.maximum) {
    mass.sum = mass.sum * exp((mass.maximum - value) / temperature) + 1.0f;
    mass.maximum = value;
  } else {
    mass.sum += exp((value - mass.maximum) / temperature);
  }
  ++mass.admitted;
}

// The shard's share from its threads' running masses.
inline void store_shard_mass(TargetShardMass mass, TargetRow row,
                             device TargetShardMass &partial,
                             threadgroup TargetShardMass *group_masses,
                             uint thread_index, uint lane, uint simd_group) {
  const float simd_maximum = simd_max(mass.maximum);
  const float scaled =
      mass.sum > 0.0f
          ? mass.sum * exp((mass.maximum - simd_maximum) / row.temperature)
          : 0.0f;
  const float simd_total = simd_sum(scaled);
  const uint simd_admitted = simd_sum(mass.admitted);
  if (lane == 0)
    group_masses[simd_group] = {simd_maximum, simd_total, simd_admitted};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0)
    partial = merge_masses(group_masses, 8, row.temperature);
}

// One shard of each selected row of the sampled lanes: its share of the
// row's softmax denominator, the bytes decode_sample_mass_sharded writes, and
// the ids of its first kTopTokens admitted tokens in the order of the logits
// (~0u past the last), in the row's draw ranges.
kernel void decode_sample_mass_top32_sharded(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device TargetShardMass *partial_masses [[buffer(2)]],
    device uint *shard_tops [[buffer(3)]],
    constant TargetSamplingParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  threadgroup TargetShardMass group_masses[8];
  threadgroup uint group_keys[8 * kTopTokens];
  threadgroup uint group_ids[8 * kTopTokens];
  const uint s = group / Shards;
  if (!lane_samples(params, s))
    return;
  const uint shard = group % Shards;
  const TargetRow row = selected_row(logits, token_mask, params, s);
  // One pass: shard_mass's running masses, token for token, and each
  // simdgroup's first tokens; then simdgroup 0 merges the eight lists.
  TargetShardMass mass{-FLT_MAX, 0.0f, 0};
  uint key = kNoKey;
  uint id = 0xffffffffu;
  for (uint base = shard * 256; base < row.vocabulary; base += Shards * 256) {
    const uint token = base + thread_index;
    const bool admitted = token < row.vocabulary && row.admits(token);
    const float value = admitted ? row.logits[token] : 0.0f;
    if (admitted)
      accumulate_mass(mass, value, row.temperature);
    simd_top32_offer(key, id, admitted ? logit_key(value) : kNoKey, token,
                     admitted, lane);
  }
  store_shard_mass(mass, row, partial_masses[group], group_masses,
                   thread_index, lane, simd_group);
  group_keys[simd_group * kTopTokens + lane] = key;
  group_ids[simd_group * kTopTokens + lane] = id;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group != 0)
    return;
  key = kNoKey;
  id = 0xffffffffu;
  for (uint list = 0; list < 8; ++list) {
    const uint offered = group_ids[list * kTopTokens + lane];
    simd_top32_offer(key, id, group_keys[list * kTopTokens + lane], offered,
                     offered != 0xffffffffu, lane);
  }
  shard_tops[ulong(s) * kShardTopsPerRow + shard * kTopTokens + lane] = id;
}

// distribution_end's end, found from the row's first tokens in the order of
// the logits (ranked keys and ids; every admitted token when there are fewer
// than kTopTokens), for a lane that keeps at most kTopTokens tokens by top-k
// and none by min-p; found is false where distribution_end may end
// elsewhere. Top-k alone ends on the top_k-th token, or keeps every admitted
// token. A nucleus walks the kept tokens' weights in order and ends on the
// first whose cumulative mass exceeds top_p times their mass, which
// distribution_end sums from the first token as here, or over the shards
// (merged.sum) when top_k keeps every token; it sums the top-k mass in
// another order when its search brackets more than a group ranks, which
// moves the target by under 2^-16 of it, so a cumulative mass that close to
// the target leaves the end unknown. So do non-finite kept logits, which the
// key order and distribution_end's sums place differently.
struct TopEnd {
  OrderBoundary end;
  bool found;
};

inline TopEnd top_tokens_end(TargetRow row, TargetShardMass merged,
                             uint top_k, float top_p,
                             threadgroup const uint *keys,
                             threadgroup const uint *ids) {
  const TopEnd unknown{{kNoKey, 0}, false};
  const bool cut = top_k < merged.admitted;
  const uint kept = cut ? top_k : merged.admitted;
  if (kept == 0 || keys[0] >= kInfinityKey || keys[kept - 1] <= kLowestKey)
    return unknown;
  if (!(top_p < 1.0f))
    return {cut ? OrderBoundary{keys[kept - 1], ids[kept - 1]}
                : OrderBoundary{kNoKey, 0},
            true};
  float mass = merged.sum;
  if (cut) {
    mass = 0.0f;
    for (uint index = 0; index < kept; ++index)
      mass += row.weight(key_logit(keys[index]));
  }
  const float target = top_p * mass;
  const float margin = target * (1.0f / 65536.0f);
  float walked = 0.0f;
  for (uint index = 0; index < kept; ++index) {
    walked += row.weight(key_logit(keys[index]));
    if (!(fabs(walked - target) > margin))
      return unknown;
    if (walked > target)
      return {{keys[index], ids[index]}, true};
  }
  // Rounding left the nucleus unreached, within the margin of its target.
  return unknown;
}

// decode_sample_vocabulary_search for selections whose sampled lanes keep at
// most kTopTokens tokens by top-k and none by min-p: simdgroup 0 merges the
// shards' lists into the row's first kTopTokens tokens, which end the row
// unless top_tokens_end leaves it to distribution_end.
kernel void decode_sample_vocabulary_search_top32(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const TargetShardMass *partial_masses [[buffer(2)]],
    device const uint *shard_tops [[buffer(3)]],
    device TargetVocabularyRow *vocabulary_rows [[buffer(4)]],
    constant TargetSamplingParams &params [[buffer(5)]],
    uint s [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  threadgroup VocabularyScratch scratch;
  threadgroup TopEnd top_end;
  if (!lane_samples(params, s))
    return;
  const uint batch = selected_lane(params, s);
  TargetRow row = selected_row(logits, token_mask, params, s);
  const TargetShardMass merged =
      merge_masses(partial_masses + ulong(s) * Shards, Shards, row.temperature);
  row.maximum = merged.maximum;
  const uint top_k = params.top_k[batch];
  const float top_p = params.top_p[batch];
  const float min_p = params.min_p[batch];
  // Simdgroup g loads shard g's list and its keys, all at once; simdgroup 0
  // then merges the lists into the row's first tokens, after them.
  constexpr uint Listed = Shards * kTopTokens;
  if (simd_group < Shards) {
    const uint index = simd_group * kTopTokens + lane;
    const uint offered = shard_tops[ulong(s) * kShardTopsPerRow + index];
    scratch.ids[index] = offered;
    scratch.keys[index] =
        offered != 0xffffffffu ? logit_key(row.logits[offered]) : kNoKey;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint key = kNoKey;
    uint id = 0xffffffffu;
    for (uint shard = 0; shard < Shards; ++shard) {
      const uint offered = scratch.ids[shard * kTopTokens + lane];
      simd_top32_offer(key, id, scratch.keys[shard * kTopTokens + lane],
                       offered, offered != 0xffffffffu, lane);
    }
    scratch.keys[Listed + lane] = key;
    scratch.ids[Listed + lane] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0)
    top_end = top_k <= kTopTokens && !(min_p > 0.0f)
                  ? top_tokens_end(row, merged, top_k, top_p,
                                   scratch.keys + Listed, scratch.ids + Listed)
                  : TopEnd{{kNoKey, 0}, false};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const TopEnd found = top_end;
  const OrderBoundary end =
      found.found ? found.end
                  : distribution_end(row, min_p, top_k, top_p, merged.sum,
                                     merged.admitted, scratch, thread_index,
                                     lane, simd_group);
  if (thread_index == 0)
    vocabulary_rows[s] = {merged.maximum, end.key, end.last, 0.0f};
}

// The sampling penalties of one logit: repetition divides a positive
// logit and multiplies a negative one when the prompt or the output holds
// the token, and presence and frequency lower it by the output's count of
// the token. The result saturates, so no rewritten logit is infinite.
inline float penalize_logit(float value, uint count, float repetition,
                            float repetition_inverse, float presence,
                            float frequency) {
  value *= value > 0.0f ? repetition_inverse : repetition;
  if (count)
    value -= frequency * float(count) + presence;
  return clamp(value, -FLT_MAX, FLT_MAX);
}

// Rewrites one token's logit in each penalized row of one entry, in place.
// Bit r of drafted is set when the lane's verify input row r holds the
// token: those are the draft tokens that verify row r's context adds to the
// output, so row r counts bits 1..r on top of the table's count.
inline void penalize_token(device float *logits, device const uint *words,
                           uint token, uint entry, uint drafted,
                           constant SamplingPenaltyParams &params) {
  const uint word =
      words[ulong(params.table_row[entry]) * params.vocabulary + token];
  if (!word && !drafted)
    return;
  device float *column =
      logits +
      (ulong(params.logits_lane[entry]) * SPLASH_TARGET_VERIFY_ROWS +
       params.row_offset) * params.vocabulary +
      token;
  for (uint row = 0; row < params.rows; ++row) {
    const uint count = (word & SPLASH_PENALTY_COUNT_MASK) +
                       popcount(drafted & ((2u << row) - 2u));
    if (!count && !(word & SPLASH_PENALTY_PROMPT_BIT))
      continue;
    device float &logit = column[ulong(row) * params.vocabulary];
    logit = penalize_logit(logit, count, params.repetition[entry],
                           params.repetition_inverse[entry],
                           params.presence[entry], params.frequency[entry]);
  }
}

// One thread per vocabulary token and penalized entry, over the rows the
// first token after a prompt is selected from.
kernel void decode_sample_penalize(device float *logits [[buffer(0)]],
                                   device const uint *words [[buffer(1)]],
                                   constant SamplingPenaltyParams &params
                                   [[buffer(2)]],
                                   uint2 position [[thread_position_in_grid]]) {
  if (position.x < params.vocabulary && position.y < params.entries)
    penalize_token(logits, words, position.x, position.y, 0, params);
}

// The verify rows of each penalized lane, whose contexts add the lane's
// draft tokens (verify input rows 1..7) one row at a time.
kernel void decode_sample_penalize_verify(
    device float *logits [[buffer(0)]], device const uint *words [[buffer(1)]],
    device const uint *input_tokens [[buffer(2)]],
    constant SamplingPenaltyParams &params [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint token = position.x;
  const uint entry = position.y;
  if (token >= params.vocabulary || entry >= params.entries)
    return;
  device const uint *inputs =
      input_tokens + ulong(params.logits_lane[entry]) * SPLASH_TARGET_VERIFY_ROWS;
  uint drafted = 0;
  for (uint row = 1; row < SPLASH_TARGET_VERIFY_ROWS; ++row)
    drafted |= uint(inputs[row] == token) << row;
  penalize_token(logits, words, token, entry, drafted, params);
}

inline bool top_beats(float value, uint token, float other, uint other_token) {
  return value > other || (value == other && token < other_token);
}

// Register-resident sorted top-16 insert for an entry the caller has already
// checked against the last slot; the unrolled shift keeps every index static.
inline void top16_insert(thread float (&values)[16], thread uint (&ids)[16],
                         float value, uint token) {
#pragma clang loop unroll(full)
  for (uint slot = 15; slot > 0; --slot) {
    bool here = top_beats(value, token, values[slot], ids[slot]);
    bool above = top_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  bool top = top_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

// Pops the head of a register-resident sorted list of Count entries.
template <uint Count>
inline void top_pop(thread float (&values)[Count], thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
  for (uint slot = 0; slot + 1 < Count; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[Count - 1] = -INFINITY;
  ids[Count - 1] = 0xffffffffu;
}

// The simdgroup's best list head: (value desc, id asc), so ties and the
// empty sentinel (-inf, ~0u) resolve the same way everywhere.
inline void simd_best_head(float value, uint token, thread float &best,
                           thread uint &best_token) {
  best = simd_max(value);
  best_token = simd_min(value == best ? token : 0xffffffffu);
}

// SPLASH_DRAFT_HEAD_IDS: the draft head's rows are a restricted set of
// vocabulary ids, so its shard top-16 lists hold head rows; this maps them
// back to vocabulary ids before the merge (the static ids are ascending, so
// their order and ties keep).
kernel void draft_map_head_ids(device uint *partial_ids [[buffer(0)]],
                               device const uint *id_map [[buffer(1)]],
                               constant uint &count [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  if (index < count && partial_ids[index] != 0xffffffffu)
    partial_ids[index] = id_map[partial_ids[index]];
}

// One shard of one proposal row. Threads stream their share of the shard in
// 16-byte vectors, keep the chunk in registers, and first find the 16th
// largest of the per-thread maxima: at least sixteen tokens are that large,
// so nothing below it can be in the row's top-16 and the exact sorted insert
// only runs for the few survivors. The group then pops its best sixteen in
// rank order. The (value desc, id asc) order is total, so the partial is the
// same set in the same order whatever the thread partition.
kernel void draft_select_top16_sharded(
    device const float *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    constant uint &vocabulary [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint K = SPLASH_DRAFT_CANDIDATES;
  constexpr uint VectorTokens = 4, ChunkVectors = 16;
  uint batch = group / (Positions * Shards);
  uint local = group % (Positions * Shards);
  uint position = local / Shards;
  uint shard = local % Shards;
  ulong row_start = (ulong(batch) * Rows + position + 1) * vocabulary;
  uint shard_tokens = (vocabulary + Shards - 1) / Shards;
  uint begin = min(shard * shard_tokens, vocabulary);
  uint end = min(begin + shard_tokens, vocabulary);
  // Vector loads require 16-byte alignment. Handle the shard's unaligned head
  // and tail with scalar inserts; the logits binding must also be aligned.
  uint head = uint((VectorTokens - (row_start + begin) % VectorTokens) %
                   VectorTokens);
  head = min(head, end - begin);
  uint vectors = (end - begin - head) / VectorTokens;
  uint vector_begin = begin + head;
  uint tail_begin = vector_begin + vectors * VectorTokens;
  device const float *row = logits + row_start;
  device const float4 *vector_row =
      reinterpret_cast<device const float4 *>(row + vector_begin);

  float values[K];
  uint ids[K];
  for (uint i = 0; i < K; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  if (thread_index < head) {
    uint token = begin + thread_index;
    float value = row[token];
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }
  if (thread_index < end - tail_begin) {
    uint token = tail_begin + thread_index;
    float value = row[token];
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }

  threadgroup float maxima[256];
  threadgroup float thresholds[8];
  for (uint chunk = 0; chunk < vectors; chunk += 256 * ChunkVectors) {
    float4 loaded[ChunkVectors];
    float best = -INFINITY;
    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      loaded[i] = index < vectors ? vector_row[index] : float4(0.0f);
      if (index < vectors) {
        for (uint j = 0; j < VectorTokens; ++j)
          if (loaded[i][j] > best)
            best = loaded[i][j];
      }
    }
    // The 16th largest thread maximum: the minimum over the maxima that
    // fewer than sixteen others exceed.
    maxima[thread_index] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint above = 0;
    for (uint other = 0; other < 256; ++other)
      above += maxima[other] > best ? 1u : 0u;
    float candidate = simd_min(above < K ? best : INFINITY);
    if (lane == 0)
      thresholds[simd_group] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threshold = thresholds[0];
    for (uint other = 1; other < 8; ++other)
      threshold = min(threshold, thresholds[other]);

    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      if (index >= vectors)
        continue;
      uint token = vector_begin + index * VectorTokens;
      for (uint j = 0; j < VectorTokens; ++j) {
        float value = loaded[i][j];
        if (value >= threshold &&
            top_beats(value, token + j, values[K - 1], ids[K - 1]))
          top16_insert(values, ids, value, token + j);
      }
    }
  }

  // Sixteen rounds pop the group-wide best head; the simdgroup bests
  // alternate between two slots so one barrier per round suffices.
  threadgroup float round_values[2][8];
  threadgroup uint round_ids[2][8];
  for (uint rank = 0; rank < K; ++rank) {
    float head_value = values[0];
    uint head_id = ids[0];
    float simd_value;
    uint simd_id;
    simd_best_head(head_value, head_id, simd_value, simd_id);
    uint slot = rank & 1;
    if (lane == 0) {
      round_values[slot][simd_group] = simd_value;
      round_ids[slot][simd_group] = simd_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = round_values[slot][0];
    uint best_id = round_ids[slot][0];
    for (uint other = 1; other < 8; ++other) {
      float value = round_values[slot][other];
      uint token = round_ids[slot][other];
      if (top_beats(value, token, best, best_id)) {
        best = value;
        best_id = token;
      }
    }
    if (thread_index == rank) {
      partial_ids[group * K + rank] = best_id;
      partial_values[group * K + rank] = best;
    }
    if (head_value == best && head_id == best_id)
      top_pop<K>(values, ids);
  }
}

// Merges the eight shard partials of one row into the simdgroup's lanes: each
// lane holds four consecutive entries of one shard's sorted list and sixteen
// rounds pop the simdgroup-wide best into rank order, lane `rank` keeping it.
inline void top16_merge_shards(device const uint *partial_ids,
                               device const float *partial_values,
                               uint row, uint lane, thread float &value,
                               thread uint &token) {
  constexpr uint K = SPLASH_DRAFT_CANDIDATES;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Entries = Shards * K / 32;
  uint origin = row * Shards * K + lane * Entries;
  float values[Entries];
  uint ids[Entries];
  for (uint i = 0; i < Entries; ++i) {
    values[i] = partial_values[origin + i];
    ids[i] = partial_ids[origin + i];
  }
  value = -INFINITY;
  token = 0xffffffffu;
  for (uint rank = 0; rank < K; ++rank) {
    float best;
    uint best_id;
    simd_best_head(values[0], ids[0], best, best_id);
    if (lane == rank) {
      value = best;
      token = best_id;
    }
    if (values[0] == best && ids[0] == best_id)
      top_pop<Entries>(values, ids);
  }
}

// One group per proposal row. Simdgroup 0 merges its candidates and unary
// scores; simdgroup 1 merges the preceding row (the anchor for row 0). The
// remaining simdgroups score the 16 x 16 predecessor/candidate edge table in
// fixed per-lane reduction order. The table follows the shard partials in
// partial-values scratch.
kernel void draft_select_edges(
    device const uint *partial_ids [[buffer(0)]],
    device float *partial_values [[buffer(1)]],
    device uint *candidates [[buffer(2)]],
    device float *unary [[buffer(3)]],
    device const bfloat *hidden [[buffer(4)]],
    device const bfloat *predecessor_codebook [[buffer(5)]],
    device const bfloat *successor_codebook [[buffer(6)]],
    constant SelectorBatchParams &params [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = SPLASH_DRAFT_CANDIDATES;
  constexpr uint Rank = SPLASH_DRAFT_SELECTOR_RANK;
  uint batch = row / Positions;
  uint position = row % Positions;
  threadgroup uint successors[Candidates];
  threadgroup uint predecessors[Candidates];
  if (simd_group == 0) {
    float value;
    uint token;
    top16_merge_shards(partial_ids, partial_values, row, lane, value, token);
    if (lane < Candidates) {
      candidates[row * Candidates + lane] = token;
      unary[row * Candidates + lane] = value;
      successors[lane] = token;
    }
  } else if (simd_group == 1) {
    uint token = params.anchor[batch];
    if (position > 0) {
      float value;
      top16_merge_shards(partial_ids, partial_values, row - 1, lane, value,
                         token);
    }
    if (lane < Candidates)
      predecessors[lane] = token;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One task is one predecessor and eight of the sixteen candidates, so a
  // simdgroup issues all its codebook loads at once instead of one cold row
  // per edge.
  constexpr uint TaskCandidates = 8, Dims = Rank / 32;
  uint tasks = (position > 0 ? Candidates : 1) * (Candidates / TaskCandidates);
  device float *table = partial_values +
                        ulong(params.lanes) * Positions * Shards * Candidates +
                        ulong(row) * Candidates * Candidates;
  device const bfloat *row_hidden =
      hidden + (ulong(batch) * Rows + position + 1) * Rank;
  for (uint task = simd_group; task < tasks; task += threads / 32) {
    uint predecessor_index = task / (Candidates / TaskCandidates);
    uint first_candidate =
        task % (Candidates / TaskCandidates) * TaskCandidates;
    // Ids are produced by the top-k selection and are always in range; the
    // clamp only keeps a corrupted id inside the codebooks.
    uint safe_predecessor =
        min(predecessors[predecessor_index], params.vocabulary - 1u);
    float context[Dims];
    float successor[TaskCandidates][Dims];
    for (uint i = 0; i < Dims; ++i) {
      uint dim = lane + i * 32;
      context[i] = float(predecessor_codebook[safe_predecessor * Rank + dim]) *
                   float(row_hidden[dim]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      uint safe_candidate =
          min(successors[first_candidate + j], params.vocabulary - 1u);
      for (uint i = 0; i < Dims; ++i)
        successor[j][i] =
            float(successor_codebook[safe_candidate * Rank + lane + i * 32]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      float score = 0.0f;
      for (uint i = 0; i < Dims; ++i)
        score += context[i] * successor[j][i];
      score = simd_sum(score);
      if (lane == 0)
        table[predecessor_index * Candidates + first_candidate + j] = score;
    }
  }
}

// SPLASH_DRAFT_TOP_P: a sampling lane's proposal distribution over one
// position's candidates keeps its most probable ones until their mass
// reaches top_p (ties: the lower index first; always the most probable), and
// q'' renormalized over those is what the lane draws from and writes, so
// acceptance verifies the distribution it proposed from. Registers only: the
// kept set is a bit mask and, with the loops over the candidates unrolled,
// every weight index is a compile-time constant (data-indexed arrays spill to
// thread-private memory, +50 us/cycle measured in Pulsar 1.0.0).
inline uint draft_top_p_select(thread const float (&scores)[16],
                               float maximum, float temperature, float top_p,
                               float uniform, device float *q_probs) {
  constexpr uint Candidates = SPLASH_DRAFT_CANDIDATES;
  float sum = 0.0f;
  float weight[Candidates];
#pragma clang loop unroll(full)
  for (uint i = 0; i < Candidates; ++i) {
    weight[i] = exp((scores[i] - maximum) / temperature);
    sum += weight[i];
  }
  const float bound = top_p * sum;
  uint kept = 0;
  float kept_sum = 0.0f;
  uint last_kept = 0;
  while (kept_sum < bound && kept != (1u << Candidates) - 1u) {
    float best_weight = -1.0f; // below every exp weight
    uint best = 0;
#pragma clang loop unroll(full)
    for (uint i = 0; i < Candidates; ++i) {
      const bool better = (kept & (1u << i)) == 0 && weight[i] > best_weight;
      best_weight = better ? weight[i] : best_weight;
      best = better ? i : best;
    }
    kept |= 1u << best;
    kept_sum += best_weight;
    last_kept = max(last_kept, best);
  }
  // Rounding may leave the uniform unreached: the last kept candidate.
  float cumulative = 0.0f;
  bool found = false;
  uint selected = last_kept;
#pragma clang loop unroll(full)
  for (uint i = 0; i < Candidates; ++i) {
    const float probability =
        (kept & (1u << i)) != 0 ? weight[i] / kept_sum : 0.0f;
    q_probs[i] = probability;
    cumulative += probability;
    if (!found && probability > 0.0f && cumulative > uniform) {
      selected = i;
      found = true;
    }
  }
  return selected;
}

// One thread per lane walks the seven positions: the score of a candidate is
// its unary score plus the edge from the previously chosen candidate, read
// from the table draft_select_edges left in the partial-values scratch.
kernel void draft_select_dflash(
    device const uint *candidates [[buffer(0)]],
    device const float *unary [[buffer(1)]],
    device const float *partial_values [[buffer(2)]],
    device const float *uniforms [[buffer(3)]],
    device uint *tokens [[buffer(4)]], device float *q_probs [[buffer(5)]],
    constant SelectorBatchParams &params [[buffer(6)]],
    uint batch [[thread_position_in_grid]]) {
  constexpr ulong Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr ulong Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr ulong Candidates = SPLASH_DRAFT_CANDIDATES;
  candidates += batch * Positions * Candidates;
  unary += batch * Positions * Candidates;
  device const float *tables = partial_values +
                               params.lanes * Positions * Shards * Candidates +
                               batch * Positions * Candidates * Candidates;
  uniforms += batch * SPLASH_SAMPLING_UNIFORMS;
  tokens += batch * Positions;
  q_probs += batch * Positions * Candidates;

  const bool sampling = (params.sampling_mask & (1u << batch)) != 0;
  uint predecessor_index = 0;
  for (uint position = 0; position < Positions; ++position) {
    device const float *edges =
        tables + (position * Candidates + predecessor_index) * Candidates;
    float scores[Candidates];
    for (uint i = 0; i < Candidates; ++i)
      scores[i] = unary[position * Candidates + i] + edges[i];
    uint selected = 0;
    // A top_p outside (0, 1), as a zero-initialized params holds, keeps every
    // candidate: the draw below, unchanged.
    if (sampling && params.top_p > 0.0f && params.top_p < 1.0f) {
      float maximum = scores[0];
      for (uint i = 1; i < Candidates; ++i)
        maximum = max(maximum, scores[i]);
      selected = draft_top_p_select(
          scores, maximum, params.temperature[batch], params.top_p,
          uniforms[SPLASH_UNIFORM_PROPOSALS + position],
          q_probs + position * Candidates);
    } else if (sampling) {
      float maximum = scores[0];
      for (uint i = 1; i < Candidates; ++i)
        maximum = max(maximum, scores[i]);
      float sum = 0.0f;
      for (uint i = 0; i < Candidates; ++i) {
        float probability =
            exp((scores[i] - maximum) / params.temperature[batch]);
        q_probs[position * Candidates + i] = probability;
        sum += probability;
      }
      float cumulative = 0.0f;
      selected = Candidates - 1;
      for (uint i = 0; i < Candidates; ++i) {
        float probability = q_probs[position * Candidates + i] / sum;
        q_probs[position * Candidates + i] = probability;
        cumulative += probability;
        if (selected == Candidates - 1 &&
            cumulative > uniforms[SPLASH_UNIFORM_PROPOSALS + position]) {
          selected = i;
        }
      }
    } else {
      for (uint i = 1; i < Candidates; ++i) {
        if (scores[i] > scores[selected])
          selected = i;
      }
    }
    predecessor_index = selected;
    tokens[position] = candidates[position * Candidates + selected];
  }
}

inline float sparse_lookup(device const uint *ids,
                           device const float *probabilities, uint count,
                           uint token) {
  for (uint i = 0; i < count; ++i) {
    if (ids[i] == token)
      return probabilities[i];
  }
  return 0.0f;
}

// One lane's view of AcceptBatchParams, built by decode_accept_dflash.
struct AcceptParams {
  uint remaining;
  uint stop_token_0;
  uint stop_token_1;
};

// Keeps at most params.remaining of the accepted tokens plus the correction,
// cut after the first stop token, and records the retained and accepted
// counts.
inline void finish_acceptance(device const uint *tokens, uint accepted,
                              AcceptParams params, device uint &retained,
                              device uint &accepted_count) {
  accepted_count = accepted;
  retained = min(accepted + 1, params.remaining);
  for (uint i = 0; i < retained; ++i) {
    if (tokens[i] == params.stop_token_0 || tokens[i] == params.stop_token_1) {
      retained = i + 1;
      break;
    }
  }
}

// A sampled lane's verify rows carry their draft tokens' target
// probabilities (TargetVocabularyRow), and its output tokens their draws:
// the correction a rejection takes or, for the last row, the bonus token.
// Acceptance overwrites the accepted rows with the draft tokens, so row
// accepted keeps its draw.
inline void accept_sampled_lane(device const uint *draft_tokens,
                                device const uint *draft_ids,
                                device const float *draft_probs,
                                device const TargetVocabularyRow *target_rows,
                                device const float *uniforms,
                                device uint *output_tokens,
                                device uint &retained,
                                device uint &accepted_count,
                                AcceptParams params) {
  uint accepted = 0;
  while (accepted < SPLASH_DRAFT_PROPOSAL_TOKENS) {
    uint token = draft_tokens[accepted];
    float q = sparse_lookup(draft_ids + accepted * kDraftCandidates,
                            draft_probs + accepted * kDraftCandidates,
                            kDraftCandidates, token);
    float p = target_rows[accepted].draft_probability;
    if (!(uniforms[SPLASH_UNIFORM_ACCEPTANCE + accepted] * q < p))
      break;
    output_tokens[accepted] = token;
    ++accepted;
  }
  finish_acceptance(output_tokens, accepted, params, retained,
                    accepted_count);
}

// One shard of a greedy row: its admitted token with the largest logit, the
// lowest id among ties, per thread in token order and then over the group.
inline void argmax_shard(TargetRow row, uint shard, device float &partial_value,
                         device uint &partial_index,
                         threadgroup float *group_values,
                         threadgroup uint *group_indices, uint thread_index) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  float best = -INFINITY;
  uint best_index = 0xffffffffu;
  for (uint token = shard * 256 + thread_index; token < row.vocabulary;
       token += Shards * 256) {
    if (!row.admits(token))
      continue;
    float value = row.logits[token];
    if (value > best || (value == best && token < best_index)) {
      best = value;
      best_index = token;
    }
  }
  uint lane = thread_index & 31;
  uint simd_group = thread_index >> 5;
  float simd_best = simd_max(best);
  uint simd_index = simd_min(best == simd_best ? best_index : 0xffffffffu);
  if (lane == 0) {
    group_values[simd_group] = simd_best;
    group_indices[simd_group] = simd_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float value = lane < 8 ? group_values[lane] : -INFINITY;
    float group_best = simd_max(value);
    uint index =
        lane < 8 && value == group_best ? group_indices[lane] : 0xffffffffu;
    index = simd_min(index);
    if (lane == 0) {
      partial_value = group_best;
      partial_index = index;
    }
  }
}

// One shard of each selected row of the greedy lanes.
kernel void decode_sample_argmax_sharded(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    device uint *partial_indices [[buffer(3)]],
    constant TargetSamplingParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float group_values[8];
  threadgroup uint group_indices[8];
  const uint s = group / SPLASH_TARGET_SAMPLING_SHARDS;
  if (lane_samples(params, s))
    return;
  argmax_shard(selected_row(logits, token_mask, params, s),
               group % SPLASH_TARGET_SAMPLING_SHARDS, partial_values[group],
               partial_indices[group], group_values, group_indices,
               thread_index);
}

// A greedy row's token from its shards' partials.
inline uint argmax_reduce(device const float *partial_values,
                          device const uint *partial_indices, uint row,
                          uint lane) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  float value = lane < Shards ? partial_values[row * Shards + lane] : -INFINITY;
  float best = simd_max(value);
  uint index = lane < Shards && value == best
                   ? partial_indices[row * Shards + lane]
                   : 0xffffffffu;
  return simd_min(index);
}

// The token of each selected row of the greedy lanes.
kernel void decode_sample_argmax_reduce(
    device const float *partial_values [[buffer(0)]],
    device const uint *partial_indices [[buffer(1)]],
    device uint *tokens [[buffer(2)]],
    constant TargetSamplingParams &params [[buffer(3)]],
    uint s [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (lane_samples(params, s))
    return;
  const uint token = argmax_reduce(partial_values, partial_indices, s, lane);
  if (lane == 0)
    tokens[s] = token;
}

inline void accept_greedy_lane(device const uint *draft_tokens,
                               device uint *target_tokens,
                               device uint &retained,
                               device uint &accepted_count,
                               AcceptParams params) {
  uint accepted = 0;
  while (accepted < SPLASH_DRAFT_PROPOSAL_TOKENS &&
         draft_tokens[accepted] == target_tokens[accepted]) {
    ++accepted;
  }
  finish_acceptance(target_tokens, accepted, params, retained,
                    accepted_count);
}

kernel void decode_accept_dflash(
    device const uint *draft_tokens [[buffer(0)]],
    device const uint *draft_ids [[buffer(1)]],
    device const float *draft_probs [[buffer(2)]],
    device const TargetVocabularyRow *target_rows [[buffer(3)]],
    device const float *uniforms [[buffer(4)]],
    device uint *target_tokens [[buffer(5)]],
    device uint *retained [[buffer(6)]],
    device uint *accepted_count [[buffer(7)]],
    constant AcceptBatchParams &params [[buffer(8)]],
    uint batch [[threadgroup_position_in_grid]]) {
  uint remaining = params.remaining[batch];
  AcceptParams lane_params{remaining, params.stop_token_0,
                           params.stop_token_1};
  device const uint *lane_draft =
      draft_tokens + batch * SPLASH_DRAFT_PROPOSAL_TOKENS;
  device uint *lane_target =
      target_tokens + batch * SPLASH_TARGET_VERIFY_ROWS;
  if (params.sampling_mask & (1u << batch)) {
    accept_sampled_lane(
        lane_draft,
        draft_ids + batch * SPLASH_DRAFT_PROPOSAL_TOKENS * kDraftCandidates,
        draft_probs + batch * SPLASH_DRAFT_PROPOSAL_TOKENS * kDraftCandidates,
        target_rows + batch * SPLASH_TARGET_VERIFY_ROWS,
        uniforms + batch * SPLASH_SAMPLING_UNIFORMS, lane_target,
        retained[batch], accepted_count[batch], lane_params);
  } else {
    accept_greedy_lane(lane_draft, lane_target, retained[batch],
                       accepted_count[batch], lane_params);
  }
}

// Pulsar wide prompt lookup (SPLASH_WIDE_PROMPT_LOOKUP): one request's
// Rows verify rows (16 or 32), row i following its proposal input_tokens[i + 1]
// with probability 1 (a point mass). A greedy request accepts while the
// proposal is the row's argmax. A sampled one accepts row i iff its uniform is
// below the proposal's target probability, which the row's draw recorded, and
// on a rejection keeps the row's draw: with the proposal as its only
// candidate, a sample from the target distribution without the proposal
// (vocabulary_draw). Row i's uniform is uniforms[SPLASH_SAMPLING_UNIFORMS + i],
// past the correction uniform the draws used. The rows each 8-row tile
// retains go to tile_retained for the GDN and draft commits. params holds
// the request's remaining count in remaining[0] and samples iff sampling_mask
// is 1.
template <uint Rows>
inline void accept_lookup(device const uint *input_tokens,
                          device const TargetVocabularyRow *target_rows,
                          device const float *uniforms, device uint *output_tokens,
                          device uint &retained, device uint &accepted_count,
                          device uint *tile_retained,
                          constant AcceptBatchParams &params) {
  constexpr uint Proposals = Rows - 1;
  uint accepted = 0;
  if (params.sampling_mask & 1u) {
    while (accepted < Proposals &&
           uniforms[SPLASH_SAMPLING_UNIFORMS + accepted] <
               target_rows[accepted].draft_probability) {
      output_tokens[accepted] = input_tokens[accepted + 1];
      ++accepted;
    }
  } else {
    while (accepted < Proposals &&
           input_tokens[accepted + 1] == output_tokens[accepted])
      ++accepted;
  }
  finish_acceptance(output_tokens, accepted,
                    {params.remaining[0], params.stop_token_0, params.stop_token_1},
                    retained, accepted_count);
  constexpr uint Tile = SPLASH_TARGET_VERIFY_ROWS;
  for (uint tile = 0; tile < Rows / Tile; ++tile)
    tile_retained[tile] = min(retained > tile * Tile ? retained - tile * Tile : 0u, Tile);
}

#define ACCEPT_LOOKUP_ENTRY(Name, Rows)                                        \
  kernel void Name(device const uint *input_tokens [[buffer(0)]],              \
                   device const TargetVocabularyRow *target_rows [[buffer(1)]],\
                   device const float *uniforms [[buffer(2)]],                 \
                   device uint *output_tokens [[buffer(3)]],                   \
                   device uint &retained [[buffer(4)]],                        \
                   device uint &accepted_count [[buffer(5)]],                  \
                   device uint *tile_retained [[buffer(6)]],                   \
                   constant AcceptBatchParams &params [[buffer(7)]]) {         \
    accept_lookup<Rows>(input_tokens, target_rows, uniforms, output_tokens,    \
                        retained, accepted_count, tile_retained, params);      \
  }
ACCEPT_LOOKUP_ENTRY(decode_accept_lookup16, 16)
ACCEPT_LOOKUP_ENTRY(decode_accept_lookup32, 32)
#undef ACCEPT_LOOKUP_ENTRY

// SPLASH_BLOCK_VERIFY: block verification (Sun et al. 2024), exact for any
// proposal q, for batches whose lanes all sample. Draft token i of a lane
// (i = 1..7) keeps the prefix probability prefix_i = min(prefix_{i-1} *
// p(x_i) / q(x_i), 1), prefix_0 = 1, and the chance h_i that its block
// stands: mass_i / (mass_i + 1 - prefix_i), 1 when prefix_i is 1, with
// mass_i = sum_t max(prefix_i p_i(t) - q_i(t), 0) over target row i, and
// prefix_7 for the last. The lane accepts the last i whose coin passes; an
// accepted block ends with a draw from max(prefix p - q, 0) of the row after
// it, or the bonus row after the whole draft. One simdgroup per lane: thread
// c < 16 owns draft candidate c, so a mass is prefix_i times the rest's
// probability plus one simd_sum over the candidates (TargetCandidateRow).
inline float block_residual_mass(device const TargetCandidateRow &row,
                                 device const float *draft_probs,
                                 float prefix, uint lane) {
  if (prefix == 0.0f)
    return 0.0f;
  const bool owns = lane < kDraftCandidates;
  const float p = owns ? row.probability[lane] : 0.0f;
  const float q = owns ? draft_probs[lane] : 0.0f;
  const float rest = row.rest;
  if (!simd_all(p >= 0.0f && isfinite(p) && q >= 0.0f && isfinite(q)) ||
      !(rest >= 0.0f) || !isfinite(rest))
    return -1.0f;
  const float total = simd_sum(max(prefix * p - q, 0.0f)) + prefix * rest;
  return isfinite(total) ? total : -1.0f;
}

// The verify draw already holds the correction when the accepted prefix is
// 1 (nothing accepted, a prefix that stayed 1, or the bonus row): its
// residual is max(p - q, 0). Otherwise the lane's BlockCorrectionRow gets the
// prefix and the correction row's draft probabilities divided by it, and
// decode_sample_block_correction redraws the correction. A non-finite target
// or draft row fails the lane with the sentinel token, as a draw from a
// non-finite row does (Runtime's invalidSelection).
kernel void decode_accept_dflash_block(
    device const uint *draft_tokens [[buffer(0)]],
    device const uint *draft_ids [[buffer(1)]],
    device const float *draft_probs [[buffer(2)]],
    device const TargetVocabularyRow *target_rows [[buffer(3)]],
    device const TargetCandidateRow *candidate_rows [[buffer(4)]],
    device const float *uniforms [[buffer(5)]],
    device uint *target_tokens [[buffer(6)]],
    device uint *retained [[buffer(7)]],
    device uint *accepted_count [[buffer(8)]],
    device BlockCorrectionRow *corrections [[buffer(9)]],
    constant AcceptBatchParams &params [[buffer(10)]],
    uint batch [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint Proposals = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  const AcceptParams lane_params{params.remaining[batch], params.stop_token_0,
                                 params.stop_token_1};
  device const uint *drafts = draft_tokens + batch * Proposals;
  device uint *output_tokens = target_tokens + batch * Rows;
  if (!(params.sampling_mask & (1u << batch))) {
    if (lane == 0) {
      corrections[batch].prefix = 1.0f;
      accept_greedy_lane(drafts, output_tokens, retained[batch],
                         accepted_count[batch], lane_params);
    }
    return;
  }
  draft_ids += batch * Proposals * kDraftCandidates;
  draft_probs += batch * Proposals * kDraftCandidates;
  target_rows += batch * Rows;
  candidate_rows += batch * Rows;
  uniforms += batch * SPLASH_SAMPLING_UNIFORMS;

  float prefix = 1.0f;
  float accepted_prefix = 1.0f;
  uint accepted = 0;
  bool valid = true;
  for (uint i = 1; i <= Proposals; ++i) {
    const uint previous = i - 1;
    float probability = 0.0f;
    if (valid && prefix > 0.0f) {
      const float q = sparse_lookup(draft_ids + previous * kDraftCandidates,
                                    draft_probs + previous * kDraftCandidates,
                                    kDraftCandidates, drafts[previous]);
      const float p = target_rows[previous].draft_probability;
      if (!(q > 0.0f) || !isfinite(q) || !(p >= 0.0f) || !isfinite(p)) {
        valid = false;
      } else {
        probability = min(prefix * p / q, 1.0f);
        valid &= probability >= 0.0f && isfinite(probability);
      }
    }
    prefix = valid ? probability : 0.0f;

    float acceptance = prefix;
    if (i < Proposals) {
      const float mass = block_residual_mass(
          candidate_rows[i], draft_probs + i * kDraftCandidates, prefix, lane);
      const bool valid_mass = mass >= 0.0f;
      valid &= valid_mass;
      const float denominator = mass + (1.0f - prefix);
      acceptance = valid_mass && prefix == 1.0f ? 1.0f
                   : valid_mass && denominator > 0.0f && isfinite(denominator)
                       ? mass / denominator
                       : 0.0f;
      valid &= acceptance >= 0.0f && acceptance <= 1.0f && isfinite(acceptance);
    }
    if (valid && uniforms[SPLASH_UNIFORM_ACCEPTANCE + previous] < acceptance) {
      accepted = i;
      accepted_prefix = prefix;
    }
  }
  if (!valid) {
    accepted = 0;
    accepted_prefix = 1.0f;
  }
  // Every thread holds the same outcome; candidate c's thread scales it.
  if (accepted < Proposals && accepted_prefix < 1.0f && lane < kDraftCandidates)
    corrections[batch].proposal[lane] =
        draft_probs[accepted * kDraftCandidates + lane] / accepted_prefix;
  if (lane != 0)
    return;
  if (!valid)
    output_tokens[0] = 0xffffffffu;
  for (uint i = 0; i < accepted; ++i)
    output_tokens[i] = drafts[i];
  corrections[batch].prefix = accepted < Proposals ? accepted_prefix : 1.0f;
  finish_acceptance(output_tokens, accepted, lane_params, retained[batch],
                    accepted_count[batch]);
}

// SPLASH_BLOCK_VERIFY: the correction of each lane block acceptance left
// with a prefix below 1, kVocabularyGroups groups per lane: the verify draw
// of row `accepted` (vocabulary_draw, with the lane's correction uniform,
// over the verify selection's rows, records and ranges) against the draft
// probabilities divided by the prefix, which follows max(prefix p - q, 0).
// The group that draws writes the token and recounts what the lane retains.
// Every other lane's groups return at once.
kernel void decode_sample_block_correction(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const TargetVocabularyRow *vocabulary_rows [[buffer(2)]],
    device const uint *input_tokens [[buffer(3)]],
    device const uint *draft_ids [[buffer(4)]],
    device const float *uniforms [[buffer(5)]],
    device uint *tokens [[buffer(6)]],
    device coherent(device) TargetVocabularyRange *ranges [[buffer(7)]],
    device atomic_uint *arrivals [[buffer(8)]],
    device const BlockCorrectionRow *corrections [[buffer(9)]],
    device uint *accepted_counts [[buffer(10)]],
    device uint *retained [[buffer(11)]],
    constant BlockCorrectionParams &correction [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup DrawScratch scratch;
  constant TargetSamplingParams &params = correction.selection;
  const uint batch = group / kVocabularyGroups;
  const uint accepted = accepted_counts[batch];
  device const BlockCorrectionRow &lane_correction = corrections[batch];
  if (accepted >= SPLASH_DRAFT_PROPOSAL_TOKENS ||
      !(lane_correction.prefix < 1.0f))
    return;
  const uint s = batch * params.rows + accepted;
  const TargetVocabularyRow record = vocabulary_rows[s];
  TargetRow row = selected_row(logits, token_mask, params, s);
  row.maximum = record.maximum;
  const ulong position =
      ulong(batch) * SPLASH_DRAFT_PROPOSAL_TOKENS + accepted;
  uint token;
  float draft_probability;
  if (vocabulary_draw(
          row, {record.end_key, record.end_last}, true, input_tokens[s + 1],
          draft_ids + position * kDraftCandidates, lane_correction.proposal,
          uniforms[ulong(batch) * SPLASH_SAMPLING_UNIFORMS + params.uniform],
          ranges + ulong(s) * kVocabularyRanges, arrivals + s,
          group % kVocabularyGroups, scratch, thread_index, lane, simd_group,
          token, draft_probability) &&
      thread_index == 0) {
    device uint *lane_tokens = tokens + batch * SPLASH_TARGET_VERIFY_ROWS;
    lane_tokens[accepted] = token;
    finish_acceptance(lane_tokens, accepted,
                      {correction.remaining[batch], params.stop_token_0,
                       params.stop_token_1},
                      retained[batch], accepted_counts[batch]);
  }
}
