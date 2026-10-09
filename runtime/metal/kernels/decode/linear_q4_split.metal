// Modified by Pulsar.
// Pulsar's split-K decode projections (restored from Pulsar 1.0.0;
// Splash deleted its Split32/Split64 tiles in 225fb96). The tile and the
// kernels below are 1.0.0's source, byte for byte, but for the parameter
// struct: the persistent grid's stride is Q4PersistentParams::groups, the
// 12-byte layout 1.0.0's Q4Params had.
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Split-K form of q4_mpp_tile_batched: SplitK partitions of Simdgroups
// simdgroups each, all in one threadgroup, stream equal quant-group ranges of
// the same Rows x TileN tile and leave fp32 partial sums in threadgroup memory,
// [partition][row][column]; the caller reduces them and applies the epilogue.
// For Rows=8, one tile thus occupies SplitK times the simdgroups of the
// sequential form. Larger row batches can instead spend the same thread count
// on narrower tiles and expose more independent output workgroups. Every
// partition runs the same loop count, so the input-sum barriers inside stay
// aligned across the threadgroup; input_size % (256 * SplitK) == 0 keeps each
// range a whole number of four-group input-sum blocks.
//
// Numerics: each partition accumulates its own quant-group range in the
// sequential kernel's group order with the same per-group terms, so the
// only difference from the sequential form is the association of the fp32
// sum: four range sums added at the end instead of one running sum. After
// the single bf16 rounding, cancellation can make the difference exceed one
// output ulp; qualification needs an operand-magnitude error bound. Instances
// with the same SplitK retain the same per-element accumulation order.
// LocalInputSync limits only the partition-private input-sum barriers to one
// simdgroup; partial publication remains threadgroup-wide.
template <ushort TileN, bool GateUp, ushort StorageN = TileN,
          bool Pipelined = true, ushort Simdgroups = 8, ushort SplitK = 4,
          bool LocalInputSync = false, bool PrecomputedInputSums = false,
          ushort Rows = 8, bool HoistMetadata = false>
inline void q4_mpp_tile_split(device bfloat *input, device uchar *weights_0,
                              device bfloat *scales_0, device bfloat *biases_0,
                              threadgroup float *partials,
                              device uchar *weights_1, device bfloat *scales_1,
                              device bfloat *biases_1, uint input_size,
                              threadgroup float *input_sums,
                              uint output_origin, uint simd_lane,
                              uint simd_group, uint partition,
                              device const float *global_sums = nullptr) {
  static_assert(Rows == 8 || PrecomputedInputSums,
                "larger split row tiles require precomputed input sums");
  static_assert(!LocalInputSync || Simdgroups == 1,
                "local input sync requires one simdgroup per partition");
  static_assert(!HoistMetadata ||
                    (((Rows == 8 && Simdgroups == 1) || (Rows == 16 && Simdgroups == 2)) &&
                     TileN == 32 && StorageN == 256 && SplitK == 4 && Pipelined &&
                     PrecomputedInputSums && !LocalInputSync && !GateUp),
                "metadata hoist requires the M8/M16 N32 precomputed Split4 specializations");
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.slice<64, Rows>(0, 0);
  uint total_quant_groups = input_size / 64;
  uint quant_groups = total_quant_groups / SplitK;
  uint first_group = partition * quant_groups;
  uint tile = output_origin / StorageN;
  uint tile_offset = output_origin % StorageN;
  device uchar *tile_weights_0 =
      weights_0 +
      (ulong(tile) * total_quant_groups + first_group) * StorageN * 64 / 2;
  device uchar *tile_weights_1 =
      weights_1 +
      (ulong(tile) * total_quant_groups + first_group) * StorageN * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b1(
      tile_weights_1 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto b10 = first_b1.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b00), float>();
  auto accumulated_1 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b10), float>();
  // Full Rows x TileN destination and uniform partition capacity, as above.
  const bool fullyOccupied =
      uint(accumulated_0.get_capacity()) * (uint(Simdgroups) * 32u) ==
      uint(Rows) * TileN;
  const auto traversal = fullyOccupied ? Q4Traversal::All
                                       : q4_traversal(accumulated_0);
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    accumulated_0[i] = 0.0f;
    if constexpr (GateUp)
      accumulated_1[i] = 0.0f;
  });

  if constexpr (!PrecomputedInputSums) {
    q4_store_input_sums<Rows, Simdgroups>(input, input_size, first_group * 64,
                                          input_sums, 0, simd_lane, simd_group);
    if constexpr (LocalInputSync)
      simdgroup_barrier(mem_flags::mem_threadgroup);
    else
      threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto run_group = [&](uint quant_group,
                       thread decltype(accumulated_0) &partial_0,
                       thread decltype(accumulated_1) &partial_1) {
    uint input_origin = (first_group + quant_group) * 64;
    auto a_slice = a.slice<64, Rows>(input_origin, 0);
    device uchar *group_weights_0 =
        tile_weights_0 + (ulong(quant_group) * StorageN + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b0(
        group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b0_slice = b0.slice<64, TileN>(0, 0);
    operation.run(a_slice, b0_slice, partial_0);
    device uchar *group_weights_1 =
        tile_weights_1 + (ulong(quant_group) * StorageN + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b1(
        group_weights_1, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b1_slice = b1.slice<64, TileN>(0, 0);
    if constexpr (GateUp)
      operation.run(a_slice, b1_slice, partial_1);
  };
  auto finish_group = [&](uint quant_group,
                          thread decltype(accumulated_0) &partial_0,
                          thread decltype(accumulated_1) &partial_1) {
    q4_visit(accumulated_0, traversal,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter =
          (ulong(tile) * total_quant_groups + first_group + quant_group) *
              StorageN + tile_offset + index[0];
      uint sum_offset =
          ((quant_group >> 2) & 1) * (4 * Rows) + (quant_group & 3) * Rows;
      float input_sum;
      if constexpr (PrecomputedInputSums)
        input_sum = global_sums[(first_group + quant_group) * Rows + row];
      else
        input_sum = input_sums[sum_offset + row];
      accumulated_0[i] +=
          partial_0[i] * float(scales_0[parameter]) +
          input_sum * float(biases_0[parameter]);
      if constexpr (GateUp) {
        accumulated_1[i] +=
            partial_1[i] * float(scales_1[parameter]) +
            input_sum * float(biases_1[parameter]);
      }
    });
    if constexpr (!PrecomputedInputSums) {
      if ((quant_group & 3) == 3 && quant_group + 1 < quant_groups) {
        uint next_group = (quant_group + 1) >> 2;
        q4_store_input_sums<Rows, Simdgroups>(
            input, input_size, (first_group + quant_group) * 64 + 64,
            input_sums, (next_group & 1) * (4 * Rows), simd_lane, simd_group);
        if constexpr (LocalInputSync)
          simdgroup_barrier(mem_flags::mem_threadgroup);
        else
          threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  };
  if constexpr (Pipelined && HoistMetadata && PrecomputedInputSums) {
    // finish_group's arithmetic, with each pair's scale, bias and input-sum
    // loads issued before the pair's matmuls instead of after them.
    // Destination slots per lane; the exact-output check catches a larger
    // capacity (8 x 32 over one SIMD-group, or 16 x 32 over two, is 8 valid slots).
    constexpr ushort kSlots = 64;
    auto load_group = [&](uint quant_group, thread float *scale_0,
                          thread float *bias_0, thread float *scale_1,
                          thread float *bias_1, thread float *sum) {
      q4_visit(accumulated_0, traversal,
               [&](ushort i) __attribute__((always_inline)) {
        auto index = accumulated_0.get_multidimensional_index(i);
        ulong parameter =
            (ulong(tile) * total_quant_groups + first_group + quant_group) *
                StorageN + tile_offset + index[0];
        scale_0[i] = float(scales_0[parameter]);
        bias_0[i] = float(biases_0[parameter]);
        if constexpr (GateUp) {
          scale_1[i] = float(scales_1[parameter]);
          bias_1[i] = float(biases_1[parameter]);
        }
        sum[i] = global_sums[(first_group + quant_group) * Rows + index[1]];
      });
    };
    auto finish_loaded = [&](thread decltype(accumulated_0) &partial_0,
                             thread decltype(accumulated_1) &partial_1,
                             thread const float *scale_0,
                             thread const float *bias_0,
                             thread const float *scale_1,
                             thread const float *bias_1,
                             thread const float *sum) {
      q4_visit(accumulated_0, traversal,
               [&](ushort i) __attribute__((always_inline)) {
        accumulated_0[i] = metal::fma(
            sum[i], bias_0[i], metal::fma(partial_0[i], scale_0[i], accumulated_0[i]));
        if constexpr (GateUp)
          accumulated_1[i] = metal::fma(
              sum[i], bias_1[i], metal::fma(partial_1[i], scale_1[i], accumulated_1[i]));
      });
    };
    uint quant_group = 0;
    for (; quant_group + 1 < quant_groups; quant_group += 2) {
      decltype(accumulated_0) first_0, second_0;
      decltype(accumulated_1) first_1, second_1;
      float first_scale_0[kSlots], first_bias_0[kSlots], first_scale_1[kSlots],
          first_bias_1[kSlots], first_sum[kSlots];
      float second_scale_0[kSlots], second_bias_0[kSlots],
          second_scale_1[kSlots], second_bias_1[kSlots], second_sum[kSlots];
      load_group(quant_group, first_scale_0, first_bias_0, first_scale_1,
                 first_bias_1, first_sum);
      load_group(quant_group + 1, second_scale_0, second_bias_0,
                 second_scale_1, second_bias_1, second_sum);
      run_group(quant_group, first_0, first_1);
      run_group(quant_group + 1, second_0, second_1);
      finish_loaded(first_0, first_1, first_scale_0, first_bias_0,
                    first_scale_1, first_bias_1, first_sum);
      finish_loaded(second_0, second_1, second_scale_0, second_bias_0,
                    second_scale_1, second_bias_1, second_sum);
    }
    if (quant_group < quant_groups) {
      decltype(accumulated_0) partial_0;
      decltype(accumulated_1) partial_1;
      run_group(quant_group, partial_0, partial_1);
      finish_group(quant_group, partial_0, partial_1);
    }
  } else if constexpr (Pipelined) {
    uint quant_group = 0;
    for (; quant_group + 1 < quant_groups; quant_group += 2) {
      decltype(accumulated_0) first_0, second_0;
      decltype(accumulated_1) first_1, second_1;
      run_group(quant_group, first_0, first_1);
      run_group(quant_group + 1, second_0, second_1);
      finish_group(quant_group, first_0, first_1);
      finish_group(quant_group + 1, second_0, second_1);
    }
    if (quant_group < quant_groups) {
      decltype(accumulated_0) partial_0;
      decltype(accumulated_1) partial_1;
      run_group(quant_group, partial_0, partial_1);
      finish_group(quant_group, partial_0, partial_1);
    }
  } else {
    for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
      decltype(accumulated_0) partial_0;
      decltype(accumulated_1) partial_1;
      run_group(quant_group, partial_0, partial_1);
      finish_group(quant_group, partial_0, partial_1);
    }
  }

  // fp32 partials, [partition][row][column]; the gate/up second stream
  // follows all SplitK first-stream partitions.
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    auto index = accumulated_0.get_multidimensional_index(i);
    uint slot = index[1] * TileN + index[0];
    partials[partition * Rows * TileN + slot] = accumulated_0[i];
    if constexpr (GateUp)
      partials[(SplitK + partition) * Rows * TileN + slot] = accumulated_1[i];
  });
  // The caller's reduction reads every partition's partials, and the next
  // tile's prologue rewrites input-sum region 0; every simdgroup finishes.
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Split-K decode projections for fixed row batches. A threadgroup holds
// Parts partitions of Simdgroups simdgroups each and launches with
// Parts * Simdgroups * 32 threads; the partitions stream equal K ranges of one
// Rows x TileN tile into threadgroup partials (q4_mpp_tile_split), then the whole
// threadgroup reduces the partials and applies the epilogue. A projection
// with fewer 256-wide tiles than the GPU has cores cannot fill the cores with
// the sequential kernels; splitting K multiplies the simdgroups per tile
// instead of narrowing the tile further. Requires input_size % 1024 == 0
// (four 256-input ranges); ops::Linear dispatches params.groups threadgroups,
// each striding over the N32 tiles.
template <ushort TileN, ushort Simdgroups, bool Residual, bool GateUp = false,
          bool LocalInputSync = false, bool PrecomputedInputSums = false,
          ushort Rows = 8, bool FooterGuard = false, bool HoistMetadata = false,
          class Out = bfloat>
inline void q4_split(device bfloat *input, device uchar *weights,
                     device bfloat *scales, device bfloat *biases,
                     device bfloat *residual, device Out *output,
                     device uchar *upWeights, device bfloat *upScales,
                     device bfloat *upBiases, constant Q4PersistentParams &p,
                     uint group,
                     uint lane, uint simd, threadgroup float *sums,
                     threadgroup float *partials,
                     device const float *globalSums = nullptr) {
  static_assert(Rows == 8 || PrecomputedInputSums,
                "larger split row tiles require precomputed input sums");
  constexpr uint Parts = 4;
  uint partition = simd / Simdgroups;
  for (uint tile = group; tile < p.output_size / TileN; tile += p.groups) {
    q4_mpp_tile_split<TileN, GateUp, 256, true, Simdgroups, Parts,
                      LocalInputSync, PrecomputedInputSums, Rows, HoistMetadata>(
        input, weights, scales, biases, partials, upWeights, upScales,
        upBiases, p.input_size, sums + partition * (8 * Rows), tile * TileN, lane,
        simd % Simdgroups, partition, globalSums);
    // Same epilogue as q4_mpp_tile: one bf16 rounding of the projection,
    // then the residual add or the SiLU gate, then the output rounding.
    for (uint i = simd * 32 + lane; i < Rows * TileN;
         i += Parts * Simdgroups * 32) {
      float value = 0;
      for (uint part = 0; part < Parts; ++part)
        value += partials[part * Rows * TileN + i];
      uint index = (i / TileN) * p.output_size + tile * TileN + i % TileN;
      if constexpr (!is_same_v<Out, float>) value = float(bfloat(value));
      if constexpr (GateUp) {
        float up = 0;
        for (uint part = 0; part < Parts; ++part)
          up += partials[(Parts + part) * Rows * TileN + i];
        value = value / (1.0f + fast::exp2(-1.44269504089f * value)) *
                float(bfloat(up));
      }
      if constexpr (Residual)
        value += float(residual[index]);
      output[index] = Out(value);
    }
    // The next tile's partials overwrite this reduction's inputs. FooterGuard (SPLASH_SPLIT4_FOOTER, exact): on a
    // group's last tile nothing writes its scratch again, so that one reuse barrier is skipped; the test is uniform
    // across the threadgroup, and every non-final tile keeps its barrier.
    if (!FooterGuard || tile + p.groups < p.output_size / TileN)
      threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void decode_linear_q4_split_sums(
    device bfloat *input [[buffer(0)]], device float *sums [[buffer(1)]],
    constant uint &inputSize [[buffer(2)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  q4_store_input_sums<8, 4>(input, inputSize, block * 256, sums, block * 32,
                            lane, simd);
}

kernel void decode_linear_q4_split_sums_m16(
    device bfloat *input [[buffer(0)]], device float *sums [[buffer(1)]],
    constant uint &inputSize [[buffer(2)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  q4_store_input_sums<16, 8>(input, inputSize, block * 256, sums, block * 64,
                             lane, simd);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float *sums [[buffer(5)]],
    constant Q4PersistentParams &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, false, false, false, true>(
      input, weights, scales, biases, output, output, weights, scales, biases,
      params, group, lane, simd, localSums, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_residual(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4PersistentParams &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, true, false, false, true>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, localSums, partials, sums);
}


// SPLASH_SPLIT4_FOOTER (exact scheduling): the two M8 entries above with the last-tile
// scratch-reuse barrier skipped (q4_split FooterGuard); the backend selects them per submission.
kernel void decode_linear_q4_n32_split4_precomputed_sums_ftr(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float *sums [[buffer(5)]],
    constant Q4PersistentParams &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, false, false, false, true, 8, true>(
      input, weights, scales, biases, output, output, weights, scales, biases,
      params, group, lane, simd, localSums, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_ftr(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4PersistentParams &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, true, false, false, true, 8, true>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, localSums, partials, sums);
}

// SPLASH_SPLIT4_HOIST: opt-in M8 metadata loads before each matmul pair,
// with explicit stock-order FMAs. Footer selection stays independent.
kernel void decode_linear_q4_n32_split4_precomputed_sums_hoist(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float *sums [[buffer(5)]],
    constant Q4PersistentParams &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, false, false, false, true, 8, false, true>(
      input, weights, scales, biases, output, output, weights, scales, biases,
      params, group, lane, simd, localSums, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_hoist_ftr(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float *sums [[buffer(5)]],
    constant Q4PersistentParams &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, false, false, false, true, 8, true, true>(
      input, weights, scales, biases, output, output, weights, scales, biases,
      params, group, lane, simd, localSums, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_hoist(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4PersistentParams &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, true, false, false, true, 8, false, true>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, localSums, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_hoist_ftr(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4PersistentParams &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float localSums[4 * 64], partials[4 * 8 * 32];
  q4_split<32, 1, true, false, false, true, 8, true, true>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, localSums, partials, sums);
}

// SPLASH_M16_INPUT_SUMS: 16-row (two-lane) input projections reading the
// input RMS's [group * 16 + row] sums; same partitions as the residual form.
kernel void decode_linear_q4_n32_split4_precomputed_sums_m16(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float *sums [[buffer(5)]],
    constant Q4PersistentParams &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[4 * 16 * 32];
  q4_split<32, 2, false, false, false, true, 16>(
      input, weights, scales, biases, output, output, weights, scales,
      biases, params, group, lane, simd, partials, partials, sums);
}

kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_m16(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4PersistentParams &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[4 * 16 * 32];
  q4_split<32, 2, true, false, false, true, 16>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, partials, partials, sums);
}


// SPLASH_SPLIT4_M16=1 (B2 only, chosen per encode by QwenTarget): the 16-row residual entry above with the M8
// footer (last-tile scratch-reuse barrier skipped) and metadata hoist (explicit stock-order FMAs).
kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_m16_hoist_ftr(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4PersistentParams &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[4 * 16 * 32];
  q4_split<32, 2, true, false, false, true, 16, true, true>(
      input, weights, scales, biases, residual, output, weights, scales,
      biases, params, group, lane, simd, partials, partials, sums);
}

// SPLASH_M24_NARROW_SPLIT: three- and four-lane (24/32-row) narrow projections
// as four K partitions of two simdgroups (256 threads) reading precomputed sums.
#define Q4_SPLIT_MULTI_ROW(Rows)                                                \
  kernel void decode_linear_q4_n32_split4_precomputed_sums_m##Rows(             \
      device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],  \
      device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]], \
      device bfloat *output [[buffer(4)]],                                      \
      device const float *sums [[buffer(5)]],                                   \
      constant Q4PersistentParams &params [[buffer(6)]],                        \
      uint group [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd [[simdgroup_index_in_threadgroup]]) {                           \
    threadgroup float partials[4 * Rows * 32];                                  \
    q4_split<32, 2, false, false, false, true, Rows>(                           \
        input, weights, scales, biases, output, output, weights, scales,        \
        biases, params, group, lane, simd, partials, partials, sums);           \
  }                                                                             \
  kernel void decode_linear_q4_n32_split4_precomputed_sums_residual_m##Rows(    \
      device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],  \
      device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]], \
      device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]], \
      device const float *sums [[buffer(6)]],                                   \
      constant Q4PersistentParams &params [[buffer(7)]],                        \
      uint group [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd [[simdgroup_index_in_threadgroup]]) {                           \
    threadgroup float partials[4 * Rows * 32];                                  \
    q4_split<32, 2, true, false, false, true, Rows>(                            \
        input, weights, scales, biases, residual, output, weights, scales,      \
        biases, params, group, lane, simd, partials, partials, sums);           \
  }                                                                             \
  kernel void decode_linear_q4_split_sums_m##Rows(                              \
      device bfloat *input [[buffer(0)]], device float *sums [[buffer(1)]],     \
      constant uint &inputSize [[buffer(2)]],                                   \
      uint block [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd [[simdgroup_index_in_threadgroup]]) {                           \
    q4_store_input_sums<Rows, 8>(input, inputSize, block * 256, sums,          \
                                 block * 4 * Rows, lane, simd);                 \
  }
Q4_SPLIT_MULTI_ROW(24)
Q4_SPLIT_MULTI_ROW(32)
#undef Q4_SPLIT_MULTI_ROW

// fp32 twins of the plain precomputed-sums entries (ops::Projection::destination):
// a plan with an fp32 destination keeps its bf16 tile, so every plain decode
// kernel the planner can select has one. The fp32 projection (the logits)
// takes none of these shapes on current GPUs: N/128 exceeds the core count.
#define Q4_SPLIT_PRECOMPUTED_F32(Name, Simdgroups, Rows)                        \
  kernel void Name##_f32(                                                       \
      device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],  \
      device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]], \
      device float *output [[buffer(4)]],                                       \
      device const float *sums [[buffer(5)]],                                   \
      constant Q4PersistentParams &params [[buffer(6)]],                        \
      uint group [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd [[simdgroup_index_in_threadgroup]]) {                           \
    threadgroup float partials[4 * Rows * 32];                                  \
    q4_split<32, Simdgroups, false, false, false, true, Rows, false, false,     \
             float>(input, weights, scales, biases, input, output, weights,     \
                    scales, biases, params, group, lane, simd, partials,        \
                    partials, sums);                                            \
  }
Q4_SPLIT_PRECOMPUTED_F32(decode_linear_q4_n32_split4_precomputed_sums, 1, 8)
Q4_SPLIT_PRECOMPUTED_F32(decode_linear_q4_n32_split4_precomputed_sums_m16, 2, 16)
Q4_SPLIT_PRECOMPUTED_F32(decode_linear_q4_n32_split4_precomputed_sums_m24, 2, 24)
Q4_SPLIT_PRECOMPUTED_F32(decode_linear_q4_n32_split4_precomputed_sums_m32, 2, 32)
#undef Q4_SPLIT_PRECOMPUTED_F32

// SPLASH_FFN_FUSED_SUMS (M8) / SPLASH_M16_FFN_SUMS (M16): the N256 gate/up
// tile as decode_linear_q4_n256_gate_up[_m16] runs it, then the tile's down
// projection input sums from its bf16 outputs, [group][row] as the
// decode_linear_q4_split_sums producers write them, so the down projection's
// split-K tile needs no sums dispatch (LinearInput::GroupSums).
#define Q4_GATE_UP_SUMS(Name, Rows)                                             \
  kernel void Name(                                                             \
      device bfloat *input [[buffer(0)]], device uchar *gate_weights [[buffer(1)]], \
      device bfloat *gate_scales [[buffer(2)]],                                 \
      device bfloat *gate_biases [[buffer(3)]], device bfloat *output [[buffer(4)]], \
      device uchar *up_weights [[buffer(5)]], device bfloat *up_scales [[buffer(6)]], \
      device bfloat *up_biases [[buffer(7)]],                                   \
      device float *output_sums [[buffer(8)]],                                  \
      constant Q4PersistentParams &params [[buffer(9)]],                        \
      uint group [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd [[simdgroup_index_in_threadgroup]]) {                           \
    threadgroup float input_sums[8 * Rows];                                     \
    for (uint tile = group; tile < params.output_size / 256;                    \
         tile += params.groups) {                                               \
      q4_mpp_tile<Rows, 256, true, false>(                                      \
          input, gate_weights, gate_scales, gate_biases, output, up_weights,    \
          up_scales, up_biases, output, params.output_size, params.input_size,  \
          input_sums, tile * 256, lane, simd);                                  \
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);  \
      q4_store_input_sums<Rows, 8>(output, params.output_size, tile * 256,      \
                                   output_sums, tile * 4 * Rows, lane, simd);   \
    }                                                                           \
  }
Q4_GATE_UP_SUMS(decode_linear_q4_n256_gate_up_sums, 8)
Q4_GATE_UP_SUMS(decode_linear_q4_n256_gate_up_m16_sums, 16)
#undef Q4_GATE_UP_SUMS
