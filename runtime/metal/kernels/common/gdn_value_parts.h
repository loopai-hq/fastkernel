// Modified by meowkernels.
#pragma once

// Pulsar's GDN value parts (SPLASH_GDN_VALUE_PARTS=4, ops/GDN.cpp), ported
// from Pulsar 1.0.0 onto 1.3.0's compiled-width variants; included by
// decode/gdn.metal only. Qualified M8/B1/VH48 port. Four threadgroups own
// disjoint 32-value slices of each head; part zero alone publishes commit
// artifacts. 1.3.0 keeps no recurrent rows and no completion counters: the
// scan writes its rows into the hidden rows, and the finalize gates them in
// place.

// Pulsar 1.0.0's gated RMSNorm of the recurrent rows (1.3.0's prefill and
// decode gates replaced it), one task per (token, value head) and one lane per
// dimension, tasks strided over `groups` threadgroups. The fork's GDN kernels
// keep it so their bytes stay Pulsar 1.0.0's. Each thread reads its row
// element before it writes it, so `recurrent` may be `hidden`.
template <uint ValueHeads, uint HeadDim, uint ConvDim, uint Simdgroups = 8>
inline void
gdn_rows_gate_phase(device const bfloat *recurrent, device const bfloat *packed,
                    device const bfloat *norm_weight, device bfloat *hidden,
                    uint tasks, uint groups, uint packed_width,
                    threadgroup float *scratch, uint group, uint thread_index,
                    uint lane, uint simd_group) {
  constexpr uint ZOffset = ConvDim;
  for (uint task = group; task < tasks; task += groups) {
    uint token = task / ValueHeads;
    uint head = task % ValueHeads;
    ulong base = ulong(task) * HeadDim;
    float value =
        thread_index < HeadDim ? float(recurrent[base + thread_index]) : 0.0f;
    float square_sum = simd_sum(value * value);
    if (lane == 0)
      scratch[simd_group] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      float total = 0.0f;
      for (uint i = 0; i < Simdgroups; ++i)
        total += scratch[i];
      scratch[0] = rsqrt(total / HeadDim + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < HeadDim) {
      bfloat normalized =
          bfloat(value * scratch[0] * float(norm_weight[thread_index]));
      float gate = float(packed[token * packed_width + ZOffset +
                                head * HeadDim + thread_index]);
      float silu = gate / (1.0f + fast::exp2(-1.44269504089f * gate));
      hidden[base + thread_index] = bfloat(float(normalized) * silu);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

inline void gdn_value_parts4_gates(
    device const bfloat *packed_row, device const bfloat *dt_bias,
    device const float *a_scale, uint value_head, thread bfloat &local_beta,
    thread float &local_decay) {
  constexpr uint ValueHeads = 48, ConvDim = 10240, ValueWidth = 48 * 128;
  constexpr uint BOffset = ConvDim + ValueWidth;
  constexpr uint AOffset = BOffset + ValueHeads;
  const float b = float(packed_row[BOffset + value_head]);
  local_beta = bfloat(1.0f / (1.0f + fast::exp2(-1.44269504089f * b)));
  const bfloat x = bfloat(float(packed_row[AOffset + value_head]) +
                          float(dt_bias[value_head]));
  const float xf = float(x);
  const bfloat softplus =
      bfloat(max(xf, 0.0f) +
             fast::log2(1.0f + fast::exp2(-1.44269504089f * abs(xf))) *
                 0.69314718056f);
  local_decay = fast::exp(a_scale[value_head] * float(softplus));
}

// StoreState = false (SPLASH_GDN_SCAN_NOSTORE): neither the convolution carry
// nor the recurrent state of all eight rows is stored; the commit replays
// every lane's retained rows, fully retained ones too.
template <bool StoreState = true>
inline void gdn_value_parts4_prologue(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_state_in, device bfloat *conv_state_out,
    device bfloat *mixed, device const float *a_scale,
    device const bfloat *dt_bias, device float *decay, device bfloat *beta,
    uint packed_width, uint value_head, uint part, uint lane, uint simd_group,
    threadgroup bfloat *queries, threadgroup bfloat *keys,
    threadgroup bfloat *values, threadgroup float *local_decay,
    threadgroup bfloat *local_beta) {
  constexpr uint KeyHeads = 16, ValueHeads = 48, HeadDim = 128;
  constexpr uint ConvDim = 10240, Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint KeyWidth = KeyHeads * HeadDim, Groups = HeadDim / 32;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads, PartWidth = HeadDim / 4;
  const uint key_head = value_head / HeadsPerKey;
  const bool publish_shared = part == 0 && value_head % HeadsPerKey == 0;
  const uint token = simd_group;
  const uint q_channel = key_head * HeadDim + lane;
  const uint k_channel = KeyWidth + q_channel;
  const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;
  const uint part_begin = part * PartWidth;
  const uint part_end = part_begin + PartWidth;
  float q[Groups], k[Groups];
  for (uint g = 0; g < Groups; ++g) {
    q[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                               packed_width, ConvDim, token,
                               q_channel + 32 * g));
    k[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                               packed_width, ConvDim, token,
                               k_channel + 32 * g));
  }
  float q_sum = 0.0f, k_sum = 0.0f;
  for (uint g = 0; g < Groups; ++g) {
    q_sum += simd_sum(q[g] * q[g]);
    k_sum += simd_sum(k[g] * k[g]);
  }
  const float q_scale = rsqrt(q_sum / HeadDim + 1e-6f);
  const float k_scale = rsqrt(k_sum / HeadDim + 1e-6f);
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    const bfloat query = bfloat(float(bfloat(q[g] * q_scale)) * 0.0078125f);
    const bfloat key = bfloat(float(bfloat(k[g] * k_scale)) * 0.08838834765f);
    queries[token * HeadDim + dim] = query;
    keys[token * HeadDim + dim] = key;
    if (publish_shared) {
      mixed[token * ConvDim + q_channel + 32 * g] = query;
      mixed[token * ConvDim + k_channel + 32 * g] = key;
    }
    if (part == 0 || (dim >= part_begin && dim < part_end)) {
      const bfloat value =
          gdn_conv_silu(packed, conv_state_in, conv_weights, packed_width,
                        ConvDim, token, v_channel + 32 * g);
      if (dim >= part_begin && dim < part_end)
        values[token * HeadDim + dim] = value;
      if (part == 0)
        mixed[token * ConvDim + v_channel + 32 * g] = value;
    }
  }
  if (lane == 0) {
    bfloat token_beta;
    float token_decay;
    gdn_value_parts4_gates(packed + token * packed_width, dt_bias, a_scale,
                           value_head, token_beta, token_decay);
    local_beta[token] = token_beta;
    local_decay[token] = token_decay;
    if (part == 0) {
      const uint gate_index = token * ValueHeads + value_head;
      beta[gate_index] = token_beta;
      decay[gate_index] = token_decay;
    }
  }
  if (StoreState && part == 0 && simd_group < 3) {
    const uint row = simd_group;
    for (uint g = 0; g < Groups; ++g) {
      conv_state_out[row * ConvDim + v_channel + 32 * g] =
          gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim, Rows,
                         row, v_channel + 32 * g);
      if (publish_shared) {
        conv_state_out[row * ConvDim + q_channel + 32 * g] =
            gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim, Rows,
                           row, q_channel + 32 * g);
        conv_state_out[row * ConvDim + k_channel + 32 * g] =
            gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim, Rows,
                           row, k_channel + 32 * g);
      }
    }
  }
}

// SPLASH_GDN_DEFER: the previous B1 cycle's retained rows, which a deferred
// scan replays first (verify_gdn_value_parts4_scan_defer).
struct GdnPendingRows {
  device const bfloat *mixed;  // its q/k/v rows, ConvDim apart
  device const float *decay;   // [row][value head]
  device const bfloat *beta;   // [row][value head]
  device float *committed;     // where the replayed state goes
  uint count;                  // 0: nothing pending
};

template <bool StoreState = true, bool Defer = false>
inline void gdn_value_parts4_scan(
    device const float *state_in, device float *state_out,
    device bfloat *recurrent, threadgroup const bfloat *queries,
    threadgroup const bfloat *keys, threadgroup const bfloat *values,
    threadgroup const float *decay, threadgroup const bfloat *beta,
    uint value_head, uint part, uint lane, uint simd_group,
    GdnPendingRows pending = {}) {
  constexpr uint ValueHeads = 48, HeadDim = 128, Rows = 8;
  constexpr uint KeyWidth = 16 * HeadDim, ConvDim = 10240, HeadsPerKey = 3;
  constexpr uint RowsInFlight = 2, BatchesPerPart = 4;
  const uint first_batch = part * BatchesPerPart;
  for (uint batch = first_batch; batch < first_batch + BatchesPerPart;
       batch += RowsInFlight) {
    float state[RowsInFlight][4];
    uint value_dim[RowsInFlight];
    ulong state_base[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r) {
      value_dim[r] = (batch + r) * kDecodeSimdgroups + simd_group;
      state_base[r] =
          (ulong(value_head) * HeadDim + value_dim[r]) * HeadDim + lane * 4;
      for (uint i = 0; i < 4; ++i)
        state[r][i] = state_in[state_base[r] + i];
    }
    // The pending rows replay with verify_gdn_commit's arithmetic (state
    // decay, memory, delta, update, per column), so the committed state is
    // the commit's bit for bit; it is stored, then this cycle scans from it.
    for (uint token = 0; Defer && token < pending.count; ++token) {
      const float d = pending.decay[token * ValueHeads + value_head];
      const float b = float(pending.beta[token * ValueHeads + value_head]);
      device const bfloat *key = pending.mixed + ulong(token) * ConvDim + KeyWidth +
                                 (value_head / HeadsPerKey) * HeadDim + lane * 4;
      float memory[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        memory[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] *= d;
          memory[r] += state[r][i] * float(key[i]);
        }
        memory[r] = simd_sum(memory[r]);
      }
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float delta =
            (float(pending.mixed[ulong(token) * ConvDim + 2 * KeyWidth +
                                 value_head * HeadDim + value_dim[r]]) -
             memory[r]) *
            b;
        for (uint i = 0; i < 4; ++i)
          state[r][i] += float(key[i]) * delta;
      }
    }
    if (Defer && pending.count)
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i)
          pending.committed[state_base[r] + i] = state[r][i];
    for (uint token = 0; token < Rows; ++token) {
      const float d = decay[token];
      const float b = float(beta[token]);
      threadgroup const bfloat *key = keys + token * HeadDim + lane * 4;
      threadgroup const bfloat *query = queries + token * HeadDim + lane * 4;
      float memory[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        memory[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] *= d;
          memory[r] += state[r][i] * float(key[i]);
        }
        memory[r] = simd_sum(memory[r]);
      }
      float result[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float delta =
            (float(values[token * HeadDim + value_dim[r]]) - memory[r]) * b;
        result[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] += float(key[i]) * delta;
          result[r] += state[r][i] * float(query[i]);
        }
        result[r] = simd_sum(result[r]);
      }
      if (lane == 0)
        for (uint r = 0; r < RowsInFlight; ++r)
          recurrent[(ulong(token) * ValueHeads + value_head) * HeadDim +
                    value_dim[r]] = bfloat(result[r]);
    }
    if (StoreState)
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i)
          state_out[state_base[r] + i] = state[r][i];
  }
}

// Grid 4 x 48: value head group / 4, part group % 4. Writes the recurrent
// rows into gdn_hidden.
#define GDN_VALUE_PARTS4_SCAN(Name, StoreState)                                             \
kernel void Name(                                                                           \
    device const bfloat *packed [[buffer(0)]],                                              \
    device const bfloat *conv_weights [[buffer(1)]],                                        \
    device const uchar *current0 [[buffer(2)]],                                             \
    device uchar *next0 [[buffer(3)]], device bfloat *mixed [[buffer(4)]],                  \
    device const float *a_scale [[buffer(5)]],                                              \
    device const bfloat *dt_bias [[buffer(6)]],                                             \
    device float *decay [[buffer(7)]], device bfloat *beta [[buffer(8)]],                   \
    device bfloat *gdn_hidden [[buffer(9)]],                                                \
    constant GDNDecodeBatchParams &params [[buffer(10)]],                                   \
    uint group [[threadgroup_position_in_grid]],                                            \
    uint lane [[thread_index_in_simdgroup]],                                                \
    uint simd_group [[simdgroup_index_in_threadgroup]]) {                                   \
  constexpr uint PackedWidth = 16640;                                                       \
  const uint value_head = group / 4;                                                        \
  const uint part = group % 4;                                                              \
  device const bfloat *conv_in = reinterpret_cast<device const bfloat *>(                   \
      current0 + ulong(params.layer) * params.conv_layer_bytes);                            \
  device bfloat *conv_out = reinterpret_cast<device bfloat *>(                              \
      next0 + ulong(params.layer) * params.conv_layer_bytes);                               \
  device const float *state_in = reinterpret_cast<device const float *>(                    \
      current0 + params.convolution_state_bytes +                                           \
      ulong(params.layer) * params.recurrent_layer_bytes);                                  \
  device float *state_out = reinterpret_cast<device float *>(                               \
      next0 + params.convolution_state_bytes +                                              \
      ulong(params.layer) * params.recurrent_layer_bytes);                                  \
  threadgroup bfloat queries[8 * 128], keys[8 * 128], values[8 * 128];                      \
  threadgroup float local_decay[8];                                                         \
  threadgroup bfloat local_beta[8];                                                         \
  gdn_value_parts4_prologue<StoreState>(                                                    \
      packed, conv_weights, conv_in, conv_out, mixed, a_scale, dt_bias, decay,              \
      beta, PackedWidth, value_head, part, lane, simd_group, queries,                       \
      keys, values, local_decay, local_beta);                                               \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
  gdn_value_parts4_scan<StoreState>(state_in, state_out, gdn_hidden, queries, keys, values, \
                        local_decay, local_beta, value_head, part, lane,                    \
                        simd_group);                                                        \
}

GDN_VALUE_PARTS4_SCAN(verify_gdn_value_parts4_scan, true)
// SPLASH_GDN_SCAN_NOSTORE: the B1 scan without its stores (ops/GDN.cpp).
GDN_VALUE_PARTS4_SCAN(verify_gdn_value_parts4_scan_nostore, false)
#undef GDN_VALUE_PARTS4_SCAN

// SPLASH_GDN_DEFER (ops/GDN.cpp): the value-parts scan of a deferred B1
// cycle. With params.count pending rows (the previous cycle's retained rows,
// at the pending_* lane slot) it first replays them from rec_in (the lane's
// next cell) into the lane's current cell, then scans its own eight rows from
// that state; with none it scans from rec_in (then the current cell). Its own
// rows store no state and no convolution carry, as under
// SPLASH_GDN_SCAN_NOSTORE: a verify_gdn_commit_conv dispatch commits the
// carry and the next deferred scan (or a flush) the recurrent rows.
kernel void verify_gdn_value_parts4_scan_defer(
    device const bfloat *packed [[buffer(0)]],
    device const bfloat *conv_weights [[buffer(1)]],
    device uchar *current0 [[buffer(2)]],
    device const uchar *rec_in [[buffer(3)]],
    device bfloat *mixed [[buffer(4)]],
    device const float *a_scale [[buffer(5)]],
    device const bfloat *dt_bias [[buffer(6)]],
    device float *decay [[buffer(7)]], device bfloat *beta [[buffer(8)]],
    device bfloat *gdn_hidden [[buffer(9)]],
    device const bfloat *pending_mixed [[buffer(10)]],
    device const float *pending_decay [[buffer(11)]],
    device const bfloat *pending_beta [[buffer(12)]],
    constant GDNDeferParams &params [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint PackedWidth = 16640;
  const uint value_head = group / 4;
  const uint part = group % 4;
  device bfloat *conv = reinterpret_cast<device bfloat *>(
      current0 + ulong(params.layer) * params.conv_layer_bytes);
  const ulong recurrent =
      params.convolution_state_bytes + ulong(params.layer) * params.recurrent_layer_bytes;
  device const float *state_in = reinterpret_cast<device const float *>(rec_in + recurrent);
  threadgroup bfloat queries[8 * 128], keys[8 * 128], values[8 * 128];
  threadgroup float local_decay[8];
  threadgroup bfloat local_beta[8];
  // No carry is stored (StoreState = false), so `conv` is only read.
  gdn_value_parts4_prologue<false>(
      packed, conv_weights, conv, conv, mixed, a_scale, dt_bias, decay, beta,
      PackedWidth, value_head, part, lane, simd_group, queries, keys, values,
      local_decay, local_beta);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  gdn_value_parts4_scan<false, true>(
      state_in, nullptr, gdn_hidden, queries, keys, values, local_decay,
      local_beta, value_head, part, lane, simd_group,
      {pending_mixed, pending_decay, pending_beta,
       reinterpret_cast<device float *>(current0 + recurrent), params.count});
}

// Grid 48: the scan's rows in gdn_hidden gated in place, then the split-K
// out-projection's group sums, [group][row].
kernel void verify_gdn_value_parts4_finalize(
    device const bfloat *packed [[buffer(0)]],
    device const bfloat *gdn_norm_weight [[buffer(1)]],
    device bfloat *gdn_hidden [[buffer(2)]],
    device float *split_sums [[buffer(3)]],
    uint value_head [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint PackedWidth = 16640;
  threadgroup float scratch[kDecodeSimdgroups];
  gdn_rows_gate_phase<48, 128, 10240, kDecodeSimdgroups>(
      gdn_hidden, packed, gdn_norm_weight, gdn_hidden, 8 * 48, 48,
      PackedWidth, scratch, value_head, thread_index, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_device);
  for (uint g = 0; g < 2; ++g) {
    const uint column = value_head * 128 + g * 64 + lane;
    const uint index = simd_group * (48 * 128) + column;
    const float sum =
        simd_sum(float(gdn_hidden[index]) + float(gdn_hidden[index + 32]));
    if (lane == 0)
      split_sums[(value_head * 2 + g) * 8 + simd_group] = sum;
  }
}
