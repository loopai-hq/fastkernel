// Modified by Pulsar.
#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/Linear.h"
#include "metal/EnvSwitch.hpp"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

constexpr uint32_t kAffinePrefillTileRows = 32;
constexpr uint32_t kQuantGroup = 64;
static_assert(SPLASH_TARGET_VERIFY_ROWS == 8,
              "Q4 register tiles require eight verify rows per lane");
// The kernels sum their input per block of four quant groups (256 inputs);
// Split128 partitions K in whole blocks.
constexpr uint32_t kInputSumBlock = 4 * kQuantGroup;
// Split32PrecomputedSums partitions K four ways in whole 256-input blocks.
constexpr uint32_t kSplitInputBlock = 4 * kInputSumBlock;
// Resident groups per core of the 128-thread M8 split-K body, measured on an
// M5 Max (40 cores: about 497 of 520 groups start in the first wave); other
// GPUs unmeasured.
constexpr uint32_t kSplit4InputWaveGroupsPerCore = 12;

bool oneLaneTile(LinearTile tile) noexcept {
  return tile == LinearTile::Paired128 || tile == LinearTile::Paired256;
}
// The decode tiles whose threadgroups stream several column tiles.
bool persistentTile(LinearTile tile) noexcept {
  return tile == LinearTile::N128 || tile == LinearTile::N256 || tile == LinearTile::Paired128 ||
         tile == LinearTile::Paired256 || tile == LinearTile::Split32PrecomputedSums;
}
// The tiles of block (GGUF) projections.
bool blockTile(LinearTile tile) noexcept {
  return tile == LinearTile::GgufStaged || tile == LinearTile::GgufPrefill || tile == LinearTile::GgufRegister;
}
// Simdgroups fixed by the kernel instance: the paired N256 tile runs four and
// Split128 the N128 tile's eight; Split32PrecomputedSums runs four
// single-simdgroup partitions at one lane and four of two at more.
std::optional<LinearSimdgroups> fixedSimdgroups(LinearTile tile, uint32_t rows) noexcept {
  switch (tile) {
  case LinearTile::Split32PrecomputedSums:
    return rows == SPLASH_TARGET_VERIFY_ROWS ? LinearSimdgroups::Four : LinearSimdgroups::Eight;
  case LinearTile::Q4Register:
  case LinearTile::Paired256: return LinearSimdgroups::Four;
  case LinearTile::Split128: return LinearSimdgroups::Eight;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::GgufStaged:
  case LinearTile::GgufPrefill:
  case LinearTile::GgufRegister: return std::nullopt;
  }
  return std::nullopt;
}

void validate(LinearWorkload w) {
  if (!w.matrix.outputSize || w.matrix.outputSize % 256 ||
      !w.matrix.inputSize || w.matrix.inputSize % kQuantGroup)
    throw std::invalid_argument("invalid linear matrix");
  if (w.phase == LinearPhase::Prefill) {
    if (!w.rows || w.rows > SPLASH_PREFILL_TOKEN_BUDGET ||
        w.epilogue == LinearEpilogue::GateUp)
      throw std::invalid_argument("invalid linear prefill workload");
  } else {
    if (w.matrix.inputSize % 256 || !w.rows || w.rows % SPLASH_TARGET_VERIFY_ROWS ||
        w.rows > SPLASH_TARGET_VERIFY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH ||
        w.epilogue == LinearEpilogue::UpWithGate)
      throw std::invalid_argument("invalid linear decode workload");
  }
}

LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid linear decode batch width");
  return {matrix, lanes * SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue};
}

// LinearScratch::rotated bytes of `rows` bf16 rows of `width` inputs.
constexpr uint64_t rotatedBytes(uint32_t width, uint64_t rows) noexcept { return uint64_t{width} * rows * 2; }

// The four-simdgroup kernels: every prefill N128 tile, the decode M24 N128
// plain and residual projections, the Q4 register tile, and the one-lane
// Paired256 (plain) tile.
bool supportsFourSimdgroups(LinearWorkload w, LinearTile tile) noexcept {
  if (tile == LinearTile::Q4Register) return w.phase == LinearPhase::Decode;
  if (tile == LinearTile::Split32PrecomputedSums)
    return w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS;
  // Only the affine paired N256 kernel is instantiated: this tile is used
  // for wide plain projections; residual and gate/up retain their own tiles.
  if (tile == LinearTile::Paired256)
    return w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS &&
        w.epilogue == LinearEpilogue::None;
  if (tile != LinearTile::N128) return false;
  return w.phase == LinearPhase::Prefill ||
      (w.rows == 24 && (w.epilogue == LinearEpilogue::None ||
                        w.epilogue == LinearEpilogue::Residual));
}

// Throws if `p` is a view of the leading inputs of wider weight rows
// (Projection::leadingInputs) that `plan` has no kernel instance for: only the
// prefill residual tiles of affine Q4 weights (N128, N256) and of quantized
// GGUF segments (GgufPrefill), unrotated, read one (leadingInputsInstance).
void requireLeadingInputs(const LinearPlan &plan, const Projection &p) {
  if (!p.planeInputs()) return;
  const LinearWorkload w = plan.workload();
  const LinearTile tile = plan.configuration().tile;
  if (w.phase != LinearPhase::Prefill || w.epilogue != LinearEpilogue::Residual ||
      (tile != LinearTile::N128 && tile != LinearTile::N256 && tile != LinearTile::GgufPrefill) || p.rotation)
    throw std::invalid_argument("a view of leading inputs runs only the quantized prefill residual tiles");
}

// Throws unless views of `p`'s planes can stand for it (takesPlaneViews).
void requirePlaneViews(const Projection &p) {
  if (p.takesPlaneViews()) return;
  throw std::invalid_argument(
      "views of a projection's planes take affine Q4 weights or one unrotated quantized GGUF tensor, not a view");
}

} // namespace

// Every plane of either layout holds its rows in tiles of QUANT_TILE_ROWS
// rows, each tile's units in order, so the leading rows' tiles lead it.
static_assert(SPLASH_AFFINE_TILE_ROWS == QUANT_TILE_ROWS, "affine Q4 and GGUF planes share their tiles");

bool Projection::takesPlaneViews() const noexcept {
  if (planeInputs_) return false;
  if (layout() == WeightLayout::Affine64) return true;
  const std::vector<QuantizedSegment> &segments = blocks().segments;
  return segments.size() == 1 && !segments.front().isFloat() && !rotation;
}

Projection Projection::leadingRows(const metal::MetalBackend &backend, uint32_t rows) const {
  requirePlaneViews(*this);
  if (!rows || rows % QUANT_TILE_ROWS || rows > outputSize)
    throw std::invalid_argument("a view of leading rows takes whole plane tiles of the projection's rows");
  // The leading rows of a plane of `rowBytes` bytes per row.
  const auto view = [&](const metal::MetalBuffer &plane, uint64_t rowBytes) {
    return rowBytes ? backend.view(plane, 0, rows * rowBytes) : metal::MetalBuffer{};
  };
  if (layout() == WeightLayout::Affine64) {
    const uint64_t groups = inputSize / kQuantGroup;
    const AffineWeights &planes = affine();
    return Projection(rows, inputSize,
                      AffineWeights{view(planes.weights, groups * kQuantGroup / 2), view(planes.scales, groups * 2),
                                    view(planes.biases, groups * 2)});
  }
  const QuantizedSegment &segment = blocks().segments.front();
  const QuantFormat &format = segment.format();
  const uint64_t groups = inputSize / 32;
  return Projection(rows, inputSize,
                    BlockWeights{{QuantizedSegment::planes(segment.formatId, rows, inputSize,
                                                           view(segment.plane0, groups * format.plane0_bytes),
                                                           view(segment.plane1, groups * format.plane1_bytes),
                                                           view(segment.meta,
                                                                groups / format.meta_groups * format.meta_bytes))}});
}

Projection Projection::leadingInputs(uint32_t inputs) const {
  requirePlaneViews(*this);
  const bool affineWeights = layout() == WeightLayout::Affine64;
  const uint32_t unit = affineWeights ? kQuantGroup : 32 * blocks().segments.front().format().meta_groups;
  if (!inputs || inputs > inputSize || inputs % unit)
    throw std::invalid_argument("a view of leading inputs takes whole quant groups and meta units of the projection's");
  const auto view = [&] {
    if (affineWeights) return Projection(outputSize, inputs, affine());
    const QuantizedSegment &segment = blocks().segments.front();
    return Projection(outputSize, inputs,
                      BlockWeights{{QuantizedSegment::planes(segment.formatId, outputSize, inputs, segment.plane0,
                                                             segment.plane1, segment.meta)}});
  };
  Projection result = view();
  result.planeInputs_ = inputSize;
  return result;
}

// Table16 holds its sums per eight-row tile (metal/abi/Gguf.h), a lane's rows.
uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept {
  return layout == LinearInput::Table16
             ? rows / SPLASH_TARGET_VERIFY_ROWS * table16_sums_per_tile(width) * sizeof(float)
       : layout == LinearInput::Table64 ? uint64_t{width} * rows / 16 : 0;
}

void requireTableScratch(const LinearScratch &scratch, LinearInput layout, uint32_t width, uint32_t rows) {
  if (layout == LinearInput::Plain || !rows || rows % SPLASH_TARGET_VERIFY_ROWS || width % 64)
    throw std::invalid_argument("invalid linear table geometry");
  requireBytes(scratch.input, tableBytes(width, rows), "linear table");
  requireBytes(scratch.sums, tableSumsBytes(layout, width, rows), "linear table sums");
}

const char *tableSuffix(LinearInput layout) noexcept {
  return layout == LinearInput::Table16 ? "_table16" : layout == LinearInput::Table64 ? "_table64" : "";
}

void requireAffineProjection(const Projection &p, LinearMatrix matrix) {
  if (p.layout() != WeightLayout::Affine64 || p.outputSize != matrix.outputSize ||
      p.inputSize != matrix.inputSize)
    throw std::invalid_argument("affine projection does not match plan");
  // Each plane holds a scale, a bias or 64 weights per row and group, in
  // tiles of SPLASH_AFFINE_TILE_ROWS rows, each tile's groups in order. It
  // ends at the last group the projection reads of its last tile: a view of
  // leading inputs leaves the tile's later groups unread.
  const uint64_t groups = matrix.inputSize / kQuantGroup, rowGroups = p.planeInputSize() / kQuantGroup;
  const uint64_t parameters =
      uint64_t{matrix.outputSize} * rowGroups - SPLASH_AFFINE_TILE_ROWS * (rowGroups - groups);
  requireBytes(p.affine().weights, parameters * kQuantGroup / 2, "projection weight");
  requireBytes(p.affine().scales, parameters * 2, "projection scale");
  requireBytes(p.affine().biases, parameters * 2, "projection bias");
}

uint32_t LinearPlan::storageRows() const noexcept {
  if (workload_.weightLayout == WeightLayout::Block32) return blockStorageRows();
  if (workload_.phase != LinearPhase::Prefill) return workload_.rows;
  return ((workload_.rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows) * kAffinePrefillTileRows;
}
uint32_t LinearPlan::tileColumns() const noexcept {
  switch (config_.tile) {
  case LinearTile::Q4Register: return workload_.epilogue == LinearEpilogue::GateUp ? 32 : 64;
  case LinearTile::GgufStaged:
  case LinearTile::GgufPrefill:
  case LinearTile::GgufRegister: return GGUF_TILE_COLUMNS;
  case LinearTile::Split32PrecomputedSums: return 32;
  case LinearTile::N256:
  case LinearTile::Paired256: return 256;
  case LinearTile::N128:
  case LinearTile::Paired128:
  case LinearTile::Split128: return 128;
  }
  return 0;
}
uint32_t LinearPlan::groups() const noexcept {
  return config_.groups ? config_.groups : workload_.matrix.outputSize / tileColumns();
}
uint32_t LinearPlan::threadsPerThreadgroup() const noexcept {
  switch (config_.tile) {
  case LinearTile::GgufStaged: return GGUF_STAGED_THREADS;
  case LinearTile::GgufPrefill: return GGUF_PREFILL_THREADS;
  case LinearTile::GgufRegister: return GGUF_REGISTER_THREADS;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::Split128:
  case LinearTile::Paired256:
  case LinearTile::Q4Register:
  case LinearTile::Split32PrecomputedSums: return static_cast<uint32_t>(config_.simdgroups) * 32;
  }
  return 0;
}
bool LinearPlan::usesQ4Register() const noexcept { return config_.tile == LinearTile::Q4Register; }
LinearInput LinearPlan::input() const noexcept {
  if (rotated_) return LinearInput::Plain;
  if (config_.tile == LinearTile::GgufRegister) return LinearInput::Table16;
  return usesQ4Register() ? LinearInput::Table64 : LinearInput::Plain;
}
LinearScratchSize LinearPlan::scratchSize() const noexcept {
  if (workload_.weightLayout == WeightLayout::Block32) return blockScratchSize();
  const auto [n, k] = workload_.matrix;
  // Split128: [split][row][column] fp32 partials over every row of the step
  // and one counter per column tile.
  if (config_.tile == LinearTile::Split128)
    return {0, 0, uint64_t{config_.splits} * workload_.rows * n * sizeof(float),
            uint64_t{n / tileColumns()} * sizeof(uint32_t)};
  // Split32PrecomputedSums: each row's fp32 sum per quant group.
  if (config_.tile == LinearTile::Split32PrecomputedSums)
    return {0, uint64_t{workload_.rows} * (k / kQuantGroup) * sizeof(float), 0, 0};
  if (!usesQ4Register()) return {};
  const uint64_t rows = workload_.rows;
  const uint64_t lanes = rows / SPLASH_TARGET_VERIFY_ROWS;
  // Each row tile owns two fp32 fragment streams per K partition and one
  // completion counter per column tile; a single partition needs neither.
  return {tableBytes(k, workload_.rows), tableSumsBytes(LinearInput::Table64, k, workload_.rows),
          config_.splits > 1 ? config_.splits * 2 * rows * n * sizeof(float) : 0,
          config_.splits > 1 ? lanes * (n / tileColumns()) * sizeof(uint32_t) : 0};
}

uint64_t LinearPlan::sumsBytes() const noexcept {
  return workload_.phase == LinearPhase::Prefill && workload_.weightLayout == WeightLayout::Affine64
      ? uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) * 4 : 0;
}
uint64_t LinearPlan::gateScratchBytes() const noexcept {
  // GGUF tiles run gate/up as a gate pass and an up-with-gate pass.
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
      (workload_.epilogue == LinearEpilogue::GateUp &&
       (!secondPipeline_.empty() || workload_.weightLayout == WeightLayout::Block32));
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}
uint64_t LinearPlan::downSumsBytes() const noexcept {
  return workload_.epilogue == LinearEpilogue::UpWithGate && workload_.weightLayout == WeightLayout::Affine64
      ? uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) * 4 : 0;
}

LinearPlan::LinearPlan(LinearWorkload w, LinearConfig config, FloatOutput destination)
    : workload_(w), config_(config), destination_(destination) {
  validate(w);
  if (destination == FloatOutput::Float32 &&
      (w.phase != LinearPhase::Decode || w.epilogue != LinearEpilogue::None))
    throw std::invalid_argument("an fp32 destination takes a plain decode projection");
  const bool ggufTile = blockTile(config.tile);
  if (ggufTile != (w.weightLayout == WeightLayout::Block32))
    throw std::invalid_argument("block projections run the GGUF tiles, affine ones the Q4 tiles");
  if (w.phase == LinearPhase::Decode && persistentTile(config.tile)
          ? !config.groups || config.groups > w.matrix.outputSize / tileColumns()
          : config.groups != 0)
    throw std::invalid_argument("a persistent decode tile takes 1 to its column tiles in groups, every other plan 0");
  if (ggufTile) {
    // Kernel names follow the segment formats (LinearGguf.cpp).
    requireBlockConfiguration();
    return;
  }
  const bool splitsK = config.tile == LinearTile::Q4Register || config.tile == LinearTile::Split128;
  if (!splitsK && config.splits != 1)
    throw std::invalid_argument("K splits require the register or Split128 Q4 tile");
  if (config.simdgroups == LinearSimdgroups::Four && !supportsFourSimdgroups(w, config.tile))
    throw std::invalid_argument("invalid Q4 cooperative execution scope");
  if (const auto fixed = fixedSimdgroups(config.tile, w.rows); fixed && config.simdgroups != *fixed)
    throw std::invalid_argument("Q4 tile requires its kernel's simdgroup count");
  const bool residual = w.epilogue == LinearEpilogue::Residual;
  const bool four = config.simdgroups == LinearSimdgroups::Four;
  if (w.phase == LinearPhase::Prefill) {
    if (oneLaneTile(config.tile) || splitsK || config.tile == LinearTile::Split32PrecomputedSums)
      throw std::invalid_argument("invalid Q4 prefill configuration");
    if (four) {
      pipeline_ = w.epilogue == LinearEpilogue::UpWithGate
          ? "prefill_linear_q4_n128_up_silu_sums_sg4"
          : residual ? "prefill_linear_q4_n128_residual_sg4" : "prefill_linear_q4_n128_sg4";
    } else if (w.epilogue == LinearEpilogue::UpWithGate) {
      if (config.tile != LinearTile::N256)
        throw std::invalid_argument(
            "Q4 fused prefill up requires N256 or four simdgroups");
      pipeline_ = "prefill_linear_q4_n256_up_silu_sums";
    } else if (residual) {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128_residual" : "prefill_linear_q4_n256_residual";
    } else {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128" : "prefill_linear_q4_n256";
    }
    return;
  }
  const uint32_t lane = w.rows / SPLASH_TARGET_VERIFY_ROWS - 1;
  if (oneLaneTile(config.tile) && lane != 0)
    throw std::invalid_argument("paired Q4 tile requires one lane");
  if (usesQ4Register()) {
    const uint32_t groups = w.matrix.inputSize / kQuantGroup;
    if (!config.validSplits() || groups % config.splits)
      throw std::invalid_argument("the Q4 register tile requires whole power-of-two K partitions");
    pipeline_ = w.epilogue == LinearEpilogue::GateUp ? "decode_linear_q4_sg_gate_up" :
        residual ? "decode_linear_q4_sg_residual" : "decode_linear_q4_sg";
    return;
  }
  if (config.tile == LinearTile::Split32PrecomputedSums) {
    if (w.epilogue != LinearEpilogue::None && w.epilogue != LinearEpilogue::Residual)
      throw std::invalid_argument("Split32PrecomputedSums runs plain and residual projections");
    if (w.matrix.inputSize % kSplitInputBlock || config.groups != w.matrix.outputSize / tileColumns())
      throw std::invalid_argument("Split32PrecomputedSums requires K % 1024 == 0 and one group per tile");
    constexpr std::array plainNames{"decode_linear_q4_n32_split4_precomputed_sums",
        "decode_linear_q4_n32_split4_precomputed_sums_m16", "decode_linear_q4_n32_split4_precomputed_sums_m24",
        "decode_linear_q4_n32_split4_precomputed_sums_m32"};
    constexpr std::array residualNames{"decode_linear_q4_n32_split4_precomputed_sums_residual",
        "decode_linear_q4_n32_split4_precomputed_sums_residual_m16",
        "decode_linear_q4_n32_split4_precomputed_sums_residual_m24",
        "decode_linear_q4_n32_split4_precomputed_sums_residual_m32"};
    pipeline_ = residual ? residualNames[lane] : plainNames[lane];
    return;
  }
  if (config.tile == LinearTile::Split128) {
    // Every partition holds at least one of the kernel's 256-input blocks.
    if (!config.validSplits() || config.splits < 2 ||
        w.matrix.inputSize / kInputSumBlock < config.splits)
      throw std::invalid_argument("Split128 requires 2, 4 or 8 K partitions of 256-input blocks");
    constexpr std::array plainNames{"decode_linear_q4_n128_split", "decode_linear_q4_n128_split_m16",
        "decode_linear_q4_n128_split_m24", "decode_linear_q4_n128_split_m32"};
    constexpr std::array residualNames{"decode_linear_q4_n128_split_residual",
        "decode_linear_q4_n128_split_residual_m16", "decode_linear_q4_n128_split_residual_m24",
        "decode_linear_q4_n128_split_residual_m32"};
    constexpr std::array upSiluNames{"decode_linear_q4_n128_split_up_silu",
        "decode_linear_q4_n128_split_up_silu_m16", "decode_linear_q4_n128_split_up_silu_m24",
        "decode_linear_q4_n128_split_up_silu_m32"};
    // Gate/up at every lane count: a plain gate pass into the gate scratch,
    // then the up pass whose epilogue applies the SiLU gate.
    pipeline_ = residual ? residualNames[lane] : plainNames[lane];
    if (w.epilogue == LinearEpilogue::GateUp) secondPipeline_ = upSiluNames[lane];
    return;
  }
  if (config.tile == LinearTile::Paired256) {
    pipeline_ = "decode_linear_q4_n256_paired_sg4";
    return;
  }
  if (four) {
    pipeline_ = residual ? "decode_linear_q4_n128_residual_m24_sg4"
                         : "decode_linear_q4_n128_m24_sg4";
    return;
  }
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (config.tile != LinearTile::N256)
      throw std::invalid_argument("Q4 gate/up requires N256");
    constexpr std::array names{"decode_linear_q4_n256_gate_up", "decode_linear_q4_n256_gate_up_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
    if (lane >= 2)
      secondPipeline_ = lane == 2 ? "decode_linear_q4_n256_up_silu_m24"
                                  : "decode_linear_q4_n256_up_silu_m32";
  } else if (residual) {
    if (config.tile == LinearTile::N256)
      throw std::invalid_argument("Q4 decode residual requires N128");
    constexpr std::array names{"decode_linear_q4_n128_residual", "decode_linear_q4_n128_residual_m16",
        "decode_linear_q4_n128_residual_m24", "decode_linear_q4_n128_residual_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
        ? "decode_linear_q4_n128_residual_paired" : names[lane];
  } else if (config.tile == LinearTile::N256) {
    constexpr std::array names{"decode_linear_q4_n256", "decode_linear_q4_n256_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
  } else {
    constexpr std::array names{"decode_linear_q4_n128", "decode_linear_q4_n128_m16",
        "decode_linear_q4_n128_m24", "decode_linear_q4_n128_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
                    ? "decode_linear_q4_n128_paired" : names[lane];
  }
}

namespace {

// Decode groups stream output tiles. Under round-robin group placement, the
// most loaded core sets dispatch latency. Use the full grid for small workloads,
// balanced two-tile groups at intermediate sizes, and one resident wave for
// longer chains; sufficiently large grids balance themselves.
struct DecodeGroupPolicy final {
  // The one-tile grid wins up to this many groups per core.
  uint32_t fullGridGroupsPerCore;
  // Resident groups per core: one wave for this kernel's register footprint.
  uint32_t waveGroupsPerCore;
  // From this many tiles per core the many-wave grid wins again.
  uint32_t manyWaveTilesPerCore;
};
// Resident-wave and full-grid thresholds measured on 16/20-core Apple10 GPUs.
// Gate/up takes the lower of the two devices' limits. Two thresholds are
// derived rather than measured: gate/up's many-wave threshold is N256's, and
// the four-simdgroup thresholds are N128's doubled.
constexpr DecodeGroupPolicy kN128Groups{4, 4, 12}, kN128M16Groups{5, 4, 12},
    kN256Groups{3, 3, 8}, kGateUpGroups{3, 3, 8},
    kFourSimdgroupGroups{8, 8, 24};

// Tiles on the most loaded core when `groups` threadgroups are placed
// round-robin on `cores` and group g streams tiles g, g + groups, ...
uint32_t maxCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) noexcept {
  uint32_t worst = 0;
  for (uint32_t core = 0; core < cores; ++core) {
    uint32_t load = 0;
    for (uint32_t group = core; group < groups; group += cores)
      load += (tiles - group + groups - 1) / groups;
    worst = std::max(worst, load);
  }
  return worst;
}

uint32_t decodeGroups(uint32_t tiles, uint32_t cores,
                      DecodeGroupPolicy policy) noexcept {
  const uint32_t wave = policy.waveGroupsPerCore * cores;
  if (tiles <= policy.fullGridGroupsPerCore * cores ||
      tiles >= policy.manyWaveTilesPerCore * cores)
    return tiles;
  const uint32_t twoTile = (tiles + 1) / 2;
  // Here wave < twoTile <= tiles, so the wave is a valid count (LinearPlan
  // rejects more groups than tiles) whatever the per-core constants are.
  if (twoTile > wave) return wave;
  // The smallest balanced two-tile count keeping three quarters of the
  // full-grid limit resident. A multiple of the core count is always
  // balanced, so the search ends within `cores` steps and below `tiles`.
  const uint32_t balanced = (tiles + cores - 1) / cores;
  uint32_t groups =
      std::max(twoTile, policy.fullGridGroupsPerCore * cores * 3 / 4);
  while (maxCoreTiles(tiles, groups, cores) != balanced) ++groups;
  return groups;
}
// A multi-row N256 decode tile halves the input re-reads of N128 but also
// halves the grid; it pays only while the N256 grid keeps two tiles per core.
constexpr uint32_t kWideDecodeTilesPerCore = 2;
// Apple9 GPUs of up to 32 cores prefill with the Apple10 rule, the
// four-simdgroup N128 tile, which on a 32-core M4 Max is never slower than
// N256 on any prefill shape or row count and 6-10% faster wherever the
// difference is measurable (tune-kernels, benchmark-prefill).
// Larger Apple9 GPUs (40-core class) run N256 for UpWithGate and wherever its
// grid reaches eight threadgroups per core, which amortizes the larger tile;
// N128's four-simdgroup tile is not measured against it on them.
constexpr uint32_t kApple9MeasuredPrefillCores = 32;
constexpr double kApple9WidePrefillGroupsPerCore = 8.0;

// Apple10 and later decode split K across the threadgroups of the Split128
// tile (256 threads) by one rule at every batch width: the largest power of
// two up to LinearConfig::kMaximumSplits whose split grid still fits four
// threadgroups per core (1024 threads, twice the 512-thread occupancy knee),
// with at least one 256-input block per partition. A grid of more than two
// tiles per core keeps one split, the sequential tiles. A split grid that
// only reaches the knee leaves time, and one past four threadgroups per core
// loses to its second wave.
constexpr uint32_t kSplitGroupsPerCore = 4;

uint32_t apple10Splits(LinearMatrix matrix, uint32_t cores) noexcept {
  const uint64_t grid = matrix.outputSize / 128;
  uint32_t splits = 1;
  while (splits < LinearConfig::kMaximumSplits && grid * 2 * splits <= uint64_t{kSplitGroupsPerCore} * cores &&
         2 * splits <= matrix.inputSize / kInputSumBlock)
    splits *= 2;
  return splits;
}

// Apple10 one-lane plain projections reduce input re-reads with paired N256
// tiles at one resident wave: one tile per group while the grid fits the
// wave, several per group from kPaired256TilesPerCore. Apple9's register
// tile policy is independent.
// From eight tiles per core each group streams enough tiles (16/20-core GPUs).
constexpr uint32_t kPaired256TilesPerCore = 8;
// One resident wave; a grid past it needs a second wave, 11-23% slower.
constexpr uint32_t kPaired256WaveGroupsPerCore = 4;
// From two tiles per core one wave takes 0.78-1.04 times as long as the paired
// N128 tile on 12-, 20- and 40-core GPUs; below two it loses on 40 cores.
constexpr uint32_t kPaired256OneWaveTilesPerCore = 2;

// Pulsar's one-lane split-K rule (SPLASH_ONE_LANE_SPLIT, default on): at
// most one N128 tile per core over a long K leaves the GPU idle; tune-kernels
// on M5 Max 40 measured Split32 at 4 groups/core +34% (K 17408/25600) and
// +38..42% (K 4096/6144) GPU per projection. SPLASH_NARROW_SPLIT (default on)
// lowers its bound from N >= 4096 to N >= 256, the drafter's narrow outputs
// (serving ms/step -1.24%). Reassociates K sums against the sequential tile;
// Pulsar 1.0.0 checked its outputs against the model's quality gates.
std::optional<LinearConfig> oneLaneSplitConfig(LinearWorkload w, uint32_t cores) {
  static const bool oneLaneSplit = metal::envSwitch("SPLASH_ONE_LANE_SPLIT");
  static const bool narrowSplit = metal::envSwitch("SPLASH_NARROW_SPLIT");
  const uint32_t n = w.matrix.outputSize;
  if (oneLaneSplit && (w.epilogue == LinearEpilogue::None || w.epilogue == LinearEpilogue::Residual) &&
      n / 128 <= cores && n >= (narrowSplit ? 256u : 4096u) && w.matrix.inputSize >= 4096 &&
      w.matrix.inputSize % kSplitInputBlock == 0)
    return LinearConfig{LinearTile::Split32PrecomputedSums, n / 32, LinearSimdgroups::Four};
  return std::nullopt;
}

Split128Policy split128Policy() noexcept {
  const char *value = std::getenv("SPLASH_SPLIT128");
  if (!value) return Split128Policy::Auto;
  return std::string_view(value) == "1" ? Split128Policy::First : Split128Policy::Off;
}

std::optional<LinearConfig> apple10OneLaneConfig(LinearWorkload w, uint32_t cores) {
  // validate() requires outputSize % 256 == 0, so every tile width divides it.
  const uint32_t n = w.matrix.outputSize;
  const uint32_t tiles256 = n / 256;
  const bool oneWave = tiles256 >= kPaired256OneWaveTilesPerCore * cores &&
                       tiles256 <= kPaired256WaveGroupsPerCore * cores;
  if (w.epilogue == LinearEpilogue::None && (oneWave || tiles256 >= kPaired256TilesPerCore * cores))
    return LinearConfig{LinearTile::Paired256,
                        std::min(tiles256, kPaired256WaveGroupsPerCore * cores),
                        LinearSimdgroups::Four};
  return std::nullopt;
}

} // namespace

Linear::Linear(const DeviceCapabilities &device) noexcept
    : family_(gpuFamilyClass(device.appleGpuFamily)),
      gpuCores_(plannedGpuCores(device)), split128_(split128Policy()),
      // Read per planner, not per process, so tests can build both policies.
      narrowSplit16_(metal::envSwitch("SPLASH_M16_NARROW_SPLIT")),
      narrowSplit24_(metal::envSwitch("SPLASH_M24_NARROW_SPLIT")),
      // Wide prompt lookup verifies 16 rows of one request; each must equal
      // the 8-row verify byte for byte, and the one-lane rule splits the same
      // residual shapes, so wide lookup keeps the 16-row residual split on.
      wideResidualSplit16_(metal::envSwitch("SPLASH_WIDE_PROMPT_LOOKUP")) {}

// SPLASH_M16_NARROW_SPLIT (16 rows) / SPLASH_M24_NARROW_SPLIT (24, 32 rows),
// both default on: multi-lane projections with at most one N128 tile per core
// over a long K take the one-lane split-K rule too (four K partitions of two
// simdgroups, precomputed sums, 256 threads). Two-request cycles -2.9%; 3-4
// request cycles -20% (Pulsar 1.0.0). Reassociates K sums, as the one-lane
// rule does. Affine projections only: block (GGUF) projections keep upstream's
// tiles.
bool Linear::narrowSplitShape(LinearWorkload w) const noexcept {
  const uint32_t n = w.matrix.outputSize;
  const bool enabled = w.rows == 16
      ? narrowSplit16_ || (wideResidualSplit16_ && w.epilogue == LinearEpilogue::Residual)
      : (w.rows == 24 || w.rows == 32) && narrowSplit24_;
  return enabled && family_ == GpuFamilyClass::Apple10 && w.weightLayout == WeightLayout::Affine64 &&
      w.phase == LinearPhase::Decode &&
      (w.epilogue == LinearEpilogue::None || w.epilogue == LinearEpilogue::Residual) &&
      n / 128 <= gpuCores_ && n >= 256 && n % 256 == 0 && w.matrix.inputSize >= 4096 &&
      w.matrix.inputSize % kSplitInputBlock == 0;
}

uint32_t Linear::decodeStorageRows(uint32_t rows, ProjectionShape shape) const {
  return plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Decode, LinearEpilogue::None, shape.layout})
      .storageRows();
}

// The GPU family class selects variants; core count and workload tile counts
// determine parallelism.
LinearConfig Linear::baseline(LinearWorkload w, std::span<const Projection *const> projections) const {
  validate(w);
  if (w.weightLayout == WeightLayout::Block32) return ggufBaseline(w, projections);
  // The fork's split-K rules run ahead of Split128 unless SPLASH_SPLIT128=1.
  const bool forkSplit = split128_ != Split128Policy::First;
  if (forkSplit && narrowSplitShape(w))
    return {LinearTile::Split32PrecomputedSums, w.matrix.outputSize / 32, LinearSimdgroups::Eight};
  const uint32_t tiles128 = w.matrix.outputSize / 128;
  const uint32_t tiles256 = w.matrix.outputSize / 256;
  if (w.phase == LinearPhase::Prefill) {
    if (family_ == GpuFamilyClass::Apple10 || gpuCores_ <= kApple9MeasuredPrefillCores)
      return {LinearTile::N128, 0, LinearSimdgroups::Four};
    const uint32_t rowTiles = (w.rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows;
    const bool wide = double(rowTiles) * tiles256 >=
        kApple9WidePrefillGroupsPerCore * gpuCores_;
    return {w.epilogue == LinearEpilogue::UpWithGate || wide ? LinearTile::N256
                                                              : LinearTile::N128, 0};
  }
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  // Wide plain projections of three or four lanes take the broad-column tiles
  // below on every family, since independent row tiles would repeat the weight
  // stream. Wide is the two-N256-tiles-per-core boundary
  // (kWideDecodeTilesPerCore), not a model dimension.
  const bool widePlain = lanes >= 3 && w.epilogue == LinearEpilogue::None &&
      tiles256 >= kWideDecodeTilesPerCore * gpuCores_;
  if (family_ == GpuFamilyClass::Apple9 && !widePlain) {
    const uint32_t columns = w.epilogue == LinearEpilogue::GateUp ? 32 : 64;
    const uint32_t grid = w.matrix.outputSize / columns, groups = w.matrix.inputSize / 64;
    uint32_t splits = 1;
    // Aim for sixteen independent column/K groups per core, retaining at
    // least twelve quant groups per partition to amortize the reduction.
    while (splits < LinearConfig::kMaximumSplits && uint64_t(grid) * splits < 16ULL * gpuCores_ &&
           groups % (2 * splits) == 0 && groups / (2 * splits) >= 12)
      splits *= 2;
    return {LinearTile::Q4Register, 0, LinearSimdgroups::Four, splits};
  }
  if (family_ == GpuFamilyClass::Apple10) {
    if (forkSplit && lanes == 1)
      if (const auto config = oneLaneSplitConfig(w, gpuCores_)) return *config;
    if (const uint32_t splits = apple10Splits(w.matrix, gpuCores_);
        splits > 1 && split128_ != Split128Policy::Off)
      return {LinearTile::Split128, 0, LinearSimdgroups::Eight, splits};
    if (lanes == 1)
      if (const auto config = apple10OneLaneConfig(w, gpuCores_)) return *config;
  }
  // Apple9 reaches here only for wide plain projections of three or four
  // lanes, which keep their one-tile grids: the round-robin policy above was
  // measured on Apple10.
  const auto groups = [&](uint32_t tiles, DecodeGroupPolicy policy) {
    return family_ == GpuFamilyClass::Apple10 ? decodeGroups(tiles, gpuCores_, policy)
                                              : tiles;
  };
  if (w.epilogue == LinearEpilogue::GateUp)
    return {LinearTile::N256, groups(tiles256, kGateUpGroups)};
  // Pipelined N128 hides the latency of a single lane's weight stream.
  if (lanes == 1) return {LinearTile::Paired128, groups(tiles128, kN128Groups)};
  // Every M24 projection that gets here runs four SIMD groups.
  if (lanes == 3)
    return {LinearTile::N128, groups(tiles128, kFourSimdgroupGroups),
            LinearSimdgroups::Four};
  if (widePlain) return {LinearTile::N256, groups(tiles256, kN256Groups)};
  return {LinearTile::N128,
          groups(tiles128, lanes == 2 ? kN128M16Groups : kN128Groups)};
}

LinearPlan Linear::plan(LinearWorkload workload) const {
  return LinearPlan(workload, baseline(workload));
}
LinearPlan Linear::plan(LinearWorkload workload, LinearConfig config, FloatOutput destination) {
  return LinearPlan(workload, config, destination);
}
LinearPlan Linear::plan(LinearWorkload w, const Projection &p, const Projection *gate) const {
  w.weightLayout = p.layout();
  const std::array<const Projection *, 2> projections{&p, gate};
  LinearPlan plan(w, baseline(w, projections), p.destination);
  plan.rotated_ = static_cast<bool>(p.rotation);
  return plan;
}

LinearInput Linear::normInput(const LinearPlan &plan) const {
  // SPLASH_INPUT_FUSED_SUMS (default on): the input RMS also writes the
  // projection's group sums, and the projection runs as the split-K kernel
  // that reads them (addPreparedSums; reassociates K). Serving ms/step -2.1%
  // on Pulsar 1.0.0, which checked its outputs against the model's
  // quality gates. SPLASH_M16_INPUT_SUMS and
  // SPLASH_M24_INPUT_SUMS (default on) extend it to 16- and 24/32-row
  // verifies whose N128 (or M24/M32 N256) projections become the split-K
  // consumer, so that a row's bytes don't depend on how many rows share the
  // pass (16- and 32-row row-invariance gates). Wider verifies take the sums
  // only when the 8-row verify does: its plan is Paired128.
  static const bool fused = metal::envSwitch("SPLASH_INPUT_FUSED_SUMS");
  static const bool m16 = metal::envSwitch("SPLASH_M16_INPUT_SUMS");
  static const bool m24 = metal::envSwitch("SPLASH_M24_INPUT_SUMS");
  const LinearWorkload w = plan.workload();
  if (!fused || w.phase != LinearPhase::Decode || w.epilogue != LinearEpilogue::None ||
      w.weightLayout != WeightLayout::Affine64 || plan.destination() != FloatOutput::BFloat16 ||
      w.matrix.inputSize > SPLASH_STAGED_NORM_WIDE_WIDTH || w.matrix.inputSize % kSplitInputBlock ||
      this->plan(decode(w.matrix, 1, LinearEpilogue::None)).configuration().tile != LinearTile::Paired128)
    return plan.input();
  const LinearTile tile = plan.configuration().tile;
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  const bool takes = lanes == 1 ||
      (lanes == 2 ? m16 && tile == LinearTile::N128 : m24 && (tile == LinearTile::N128 || tile == LinearTile::N256));
  return takes ? LinearInput::GroupSums : plan.input();
}

bool Linear::gateUpWritesDownSums(const LinearPlan &gateUp, const LinearPlan &down) const {
  // SPLASH_FFN_FUSED_SUMS (default on): the one-lane gate/up kernel writes
  // the down projection's sums; byte-exact outputs and sums, Pulsar 1.0.0
  // oracle PASS. SPLASH_M16_FFN_SUMS (default on; read at every call so one
  // binary serves both arms of an A/B): the same at two lanes, B2 lockstep
  // -0.117 ms/cycle.
  static const bool fused = metal::envSwitch("SPLASH_FFN_FUSED_SUMS");
  const uint32_t lanes = gateUp.workload().rows / SPLASH_TARGET_VERIFY_ROWS;
  return down.configuration().tile == LinearTile::Split32PrecomputedSums &&
      down.workload().rows == gateUp.workload().rows &&
      down.workload().matrix.inputSize == gateUp.workload().matrix.outputSize &&
      (lanes == 1 ? fused && gateUp.pipeline() == "decode_linear_q4_n256_gate_up"
                  : lanes == 2 && metal::envSwitch("SPLASH_M16_FFN_SUMS") &&
                        gateUp.pipeline() == "decode_linear_q4_n256_gate_up_m16");
}

bool Linear::sameRowReduction(const LinearPlan &narrow, const LinearPlan &wide, bool mixerInput) const {
  const auto reduction = [&](const LinearPlan &plan) {
    if (mixerInput && normInput(plan) == LinearInput::GroupSums)
      return std::pair{LinearTile::Split32PrecomputedSums, 1U};
    const LinearConfig config = plan.configuration();
    switch (config.tile) {
    case LinearTile::N128:
    case LinearTile::N256:
    case LinearTile::Paired128:
    case LinearTile::Paired256: return std::pair{LinearTile::N128, 1U};
    default: return std::pair{config.tile, config.splits};
    }
  };
  return reduction(narrow) == reduction(wide);
}

LinearPlan Linear::decodePlan(const Projection &p, uint32_t lanes, LinearEpilogue epilogue,
                              const Projection *gate) const {
  return plan(decode({p.outputSize, p.inputSize}, lanes, epilogue), p, gate);
}
LinearPlan Linear::prefillPlan(const Projection &p, uint32_t rows, LinearEpilogue epilogue) const {
  return plan({{p.outputSize, p.inputSize}, rows, LinearPhase::Prefill, epilogue}, p);
}

LinearScratchSize Linear::decodeScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
      LinearWorkload w = decode({shape.outputSize, shape.inputSize}, lanes, epilogue);
      w.weightLayout = shape.layout;
      bound.include(shape.layout == WeightLayout::Block32 ? ggufDecodeScratchSize(w) : plan(w).scratchSize());
    }
  // Decode plans store at most every lane's rows.
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, kMaximumDecodeTileRows);
  return bound;
}

LinearScratchSize Linear::prefillScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  // Prefill plans take split scratch only in chunks of up to a decode batch,
  // which a GGUF projection runs on the staged tile (LinearGguf.cpp).
  for (uint32_t rows = 1; rows <= kMaximumDecodeTileRows; ++rows)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
      bound.include(plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Prefill, epilogue, shape.layout})
                        .scratchSize());
  // Every prefill plan stores at most the token budget.
  static_assert(SPLASH_PREFILL_TOKEN_BUDGET % GGUF_PREFILL_ROWS == 0, "the prefill tiles cover the budget exactly");
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, SPLASH_PREFILL_TOKEN_BUDGET);
  return bound;
}


std::array<metal::MetalBuffer, 2> Linear::splitScratch(const LinearBuffers &buffers, uint32_t splits) {
  if (splits > 1) return {buffers.scratch.partials, buffers.scratch.counters};
  return {buffers.output, buffers.output};
}

PreparedInput Linear::add(metal::CommandGraph &graph, LinearBuffers b,
    const Projection &p, const LinearPlan &selected, const Projection *gate) const {
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  if (p.layout() != w.weightLayout || (gate && gate->layout() != w.weightLayout))
    throw std::invalid_argument("projection layout does not match execution plan");
  if ((w.epilogue == LinearEpilogue::GateUp) != (gate != nullptr))
    throw std::invalid_argument("a gate/up plan takes a gate projection and no other plan does");
  requireLeadingInputs(selected, p);
  if (gate) requireLeadingInputs(selected, *gate);
  const uint64_t rows = selected.storageRows();
  requireBytes(b.input, rows * k * 2, "projection input");
  requireBytes(b.output, rows * n * elementBytes(selected.destination()), "projection output");
  if (w.epilogue == LinearEpilogue::Residual) requireBytes(b.residual, rows * n * 2, "projection residual");
  requireBytes(b.sums, selected.sumsBytes(), "projection sums");
  requireBytes(b.gateScratch, selected.gateScratchBytes(), "projection gate scratch");
  requireBytes(b.downSums, selected.downSumsBytes(), "projection down sums");
  const LinearScratchSize scratch = selected.scratchSize();
  requireBytes(b.scratch.input, scratch.input, "projection scratch table");
  requireBytes(b.scratch.sums, scratch.sums, "projection scratch sums");
  requireBytes(b.scratch.partials, scratch.partials, "projection partials");
  requireBytes(b.scratch.counters, scratch.counters, "projection counters");
  if (p.layout() == WeightLayout::Block32) {
    if (p.rotation) requireBytes(b.scratch.rotated, rotatedBytes(k, rows), "projection rotated input");
    addGguf(graph, b, p, selected, gate);
    // A rotated projection's plan prepares its table, if any, from the
    // rotated rows, which no other plan reads.
    if (p.rotation) return {};
    // Only quantized segments run the plan's tile: float segments alone
    // leave the scratch table as it was.
    const std::vector<QuantizedSegment> &segments = p.blocks().segments;
    const bool tiled =
        std::any_of(segments.begin(), segments.end(), [](const QuantizedSegment &s) { return !s.isFloat(); });
    return tiled && selected.input() != LinearInput::Plain ? PreparedInput{b.input, selected.input()} : b.prepared;
  }
  requireAffineProjection(p, w.matrix);
  if (gate) requireAffineProjection(*gate, w.matrix);
  const AffineWeights &weights = p.affine();
  if (selected.usesQ4Register()) {
    if (b.prepared.layout != LinearInput::Table64 || !b.prepared.source.sameView(b.input))
      graph.add("decode_linear_q4_prepare", {b.input, b.scratch.input, b.scratch.sums},
                k, {k / 32, w.rows / SPLASH_TARGET_VERIFY_ROWS, 1}, {128, 1, 1});
    const AffineWeights &first = gate ? gate->affine() : weights;
    const auto [partials, counters] = splitScratch(b, selected.configuration().splits);
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, first.weights, first.scales, first.biases,
                                             b.output, b.scratch.sums, partials, counters};
    if (gate) bindings.insert(bindings.end(), {weights.weights, weights.scales, weights.biases});
    else if (w.epilogue == LinearEpilogue::Residual) bindings.push_back(b.residual);
    graph.add(kernelInstance(selected.pipeline(), selected.destination()), std::move(bindings),
        Q4Params{n, k},
        {selected.groups(), selected.configuration().splits, w.rows / SPLASH_TARGET_VERIFY_ROWS},
        {128, 1, 1});
    return {b.input, LinearInput::Table64};
  }
  const bool groupSums = selected.configuration().tile == LinearTile::Split32PrecomputedSums;
  if (groupSums && (b.prepared.layout != LinearInput::GroupSums || !b.prepared.source.sameView(b.input))) {
    // The input's sums, [group][row], unless its producer wrote them.
    constexpr std::array names{"decode_linear_q4_split_sums", "decode_linear_q4_split_sums_m16",
                               "decode_linear_q4_split_sums_m24", "decode_linear_q4_split_sums_m32"};
    graph.add(names[w.rows / SPLASH_TARGET_VERIFY_ROWS - 1], {b.input, b.scratch.sums}, k, {k / 256, 1, 1},
              {selected.threadsPerThreadgroup(), 1, 1});
    // They replaced whatever table the scratch held.
    b.prepared = {};
  }
  const auto dispatch = [&](std::string_view name,
      std::initializer_list<metal::MetalBuffer> bindings) {
    if (w.phase == LinearPhase::Prefill) {
      const metal::DispatchSize groups{selected.storageRows() / kAffinePrefillTileRows,
                                       n / selected.tileColumns(), 1};
      const metal::DispatchSize threads{selected.threadsPerThreadgroup(), 1, 1};
      if (p.planeInputs())
        graph.add(leadingInputsInstance(name), bindings, Q4PrefillLeadingParams{{n, k}, p.planeInputs()}, groups,
                  threads);
      else
        graph.add(std::string(name), bindings, Q4Params{n, k}, groups, threads);
    } else {
      // Split128 binds its partials and counters after the sequential
      // kernel's buffers, Split32PrecomputedSums its input sums; every other
      // decode tile runs one K split.
      const LinearConfig config = selected.configuration();
      std::vector<metal::MetalBuffer> buffers(bindings);
      if (config.tile == LinearTile::Split128)
        buffers.insert(buffers.end(), {b.scratch.partials, b.scratch.counters});
      if (config.tile == LinearTile::Split32PrecomputedSums) buffers.push_back(b.scratch.sums);
      const std::string kernel = kernelInstance(name, selected.destination());
      const metal::DispatchSize groups{selected.groups(), config.splits, 1};
      const metal::DispatchSize threads{selected.threadsPerThreadgroup(), 1, 1};
      // The persistent tiles stride over the column tiles by their groups.
      if (persistentTile(config.tile))
        graph.add(kernel, std::move(buffers), Q4PersistentParams{n, k, selected.groups()}, groups, threads);
      else
        graph.add(kernel, std::move(buffers), Q4Params{n, k}, groups, threads);
    }
  };
  const bool prefill = w.phase == LinearPhase::Prefill;
  if (w.epilogue == LinearEpilogue::GateUp) {
    const AffineWeights &g = gate->affine();
    if (!prefill && b.downSums) {
      // Pulsar's fused FFN sums (gateUpWritesDownSums): the N256 gate/up
      // tile also writes the down projection's input sums into the scratch.
      const bool m16 = selected.pipeline() == "decode_linear_q4_n256_gate_up_m16";
      if ((selected.pipeline() != "decode_linear_q4_n256_gate_up" && !m16) || !b.downSums.sameView(b.scratch.sums))
        throw std::invalid_argument("down sums take the M8/M16 N256 gate/up kernel and the linear scratch's sums");
      requireBytes(b.downSums, uint64_t{n} / kQuantGroup * w.rows * sizeof(float), "projection down sums");
      dispatch(m16 ? "decode_linear_q4_n256_gate_up_m16_sums" : "decode_linear_q4_n256_gate_up_sums",
               {b.input, g.weights, g.scales, g.biases, b.output, weights.weights, weights.scales,
                weights.biases, b.downSums});
      return {b.output, LinearInput::GroupSums};
    }
    if (selected.secondPipeline().empty())
      dispatch(selected.pipeline(), {b.input, g.weights, g.scales, g.biases,
                                     b.output, weights.weights, weights.scales, weights.biases});
    else {
      dispatch(selected.pipeline(), {b.input, g.weights, g.scales, g.biases, b.gateScratch});
      dispatch(selected.secondPipeline(),
               {b.input, weights.weights, weights.scales, weights.biases, b.gateScratch, b.output});
    }
  } else if (w.epilogue == LinearEpilogue::UpWithGate)
    dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                   b.gateScratch, b.output, b.sums, b.downSums});
  else if (w.epilogue == LinearEpilogue::Residual) {
    if (prefill)
      dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                     b.residual, b.output, b.sums});
    else
      dispatch(b.m16HoistFooter &&
                       selected.pipeline() == "decode_linear_q4_n32_split4_precomputed_sums_residual_m16"
                   ? "decode_linear_q4_n32_split4_precomputed_sums_residual_m16_hoist_ftr"
                   : selected.pipeline(),
               {b.input, weights.weights, weights.scales, weights.biases, b.residual, b.output});
  } else if (prefill)
    dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases, b.output, b.sums});
  else if (!groupSums && b.prepared.layout == LinearInput::GroupSums && b.prepared.source.sameView(b.input))
    addPreparedSums(graph, b, weights, selected);
  else dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases, b.output});
  return b.prepared;
}

// Pulsar's fused input sums (normInput): the input's producer wrote its
// group sums, so a plain projection planned as the full-K Paired128 (8 rows),
// N128 (16) or N128/N256 (24, 32 rows) tile runs the split-K tile over its N/32
// tiles instead, reading them. A full-K paired consumer without its own sum
// refills stalls on every core reading the same input lines in lockstep.
// Reassociates K sums.
void Linear::addPreparedSums(metal::CommandGraph &graph, const LinearBuffers &b, const AffineWeights &weights,
                             const LinearPlan &selected) const {
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  const LinearTile tile = selected.configuration().tile;
  const bool consumer = w.rows == SPLASH_TARGET_VERIFY_ROWS ? tile == LinearTile::Paired128
      : w.rows == 16                                       ? tile == LinearTile::N128
                                    : tile == LinearTile::N128 || tile == LinearTile::N256;
  if (!consumer || selected.destination() != FloatOutput::BFloat16 || k > SPLASH_STAGED_NORM_WIDE_WIDTH ||
      k % kSplitInputBlock)
    throw std::invalid_argument("prepared group sums take a bf16 Paired128/N128/N256 plan of K % 1024 == 0 up to "
                                "the staged norm width");
  requireBytes(b.scratch.sums, uint64_t{w.rows} * (k / kQuantGroup) * sizeof(float), "prepared sums");
  const bool multi = w.rows > SPLASH_TARGET_VERIFY_ROWS;
  const uint32_t tiles = n / 32;
  uint32_t groups = tiles;
  // SPLASH_SPLIT4_INPUT_DIV (default 1 = divisor 4; =N uses divisor N; 0 =
  // off; read at every encode): an M8 input whose N/32 one-tile groups
  // overflow a resident wave (more than kSplit4InputWaveGroupsPerCore per
  // core) runs as N/32/d groups of d tiles each. Exact: the same tile body,
  // which strides over its tiles by the group count. B1 lockstep -0.203
  // ms/step (Pulsar 1.0.0, M5 Max 40); below one wave it changes nothing.
  if (!multi) {
    const char *value = std::getenv("SPLASH_SPLIT4_INPUT_DIV");
    const uint32_t d = !value || std::string_view(value) == "1"
        ? 4u : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
    if (d > 1 && tiles % d == 0 && tiles > kSplit4InputWaveGroupsPerCore * gpuCores_) groups = tiles / d;
  }
  constexpr std::array names{"decode_linear_q4_n32_split4_precomputed_sums",
                             "decode_linear_q4_n32_split4_precomputed_sums_m16",
                             "decode_linear_q4_n32_split4_precomputed_sums_m24",
                             "decode_linear_q4_n32_split4_precomputed_sums_m32"};
  graph.add(names[w.rows / SPLASH_TARGET_VERIFY_ROWS - 1],
            {b.input, weights.weights, weights.scales, weights.biases, b.output, b.scratch.sums},
            Q4PersistentParams{n, k, groups}, {groups, 1, 1}, {multi ? 256u : 128u, 1, 1});
}

void Linear::addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer sums,
                            const Projection &consumer, uint32_t rows) const {
  validate({{consumer.outputSize, consumer.inputSize}, rows, LinearPhase::Prefill});
  const uint32_t tiles = (rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows;
  const uint64_t storageRows = uint64_t{tiles} * kAffinePrefillTileRows;
  requireBytes(input, storageRows * consumer.inputSize * 2, "projection input");
  requireBytes(sums, storageRows * (consumer.inputSize / kQuantGroup) * 4, "projection sums");
  graph.add("prefill_linear_q4_sums32", {input, sums}, consumer.inputSize, {tiles, 1, 1});
}
void Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                        metal::MetalBuffer output, metal::MetalBuffer sums, uint32_t rows,
                        LinearScratch scratch) const {
  add(graph, {.input = input, .output = output, .sums = sums, .scratch = scratch}, p,
      prefillPlan(p, rows, LinearEpilogue::None));
}
void Linear::addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                metal::MetalBuffer residual, metal::MetalBuffer output, metal::MetalBuffer sums,
                                uint32_t rows, LinearScratch scratch) const {
  add(graph, {.input = input, .output = output, .sums = sums, .residual = residual, .scratch = scratch}, p,
      prefillPlan(p, rows, LinearEpilogue::Residual));
}
void Linear::addPrefillSwiGlu(metal::CommandGraph &graph, const SwiGluProjections &ffn,
                              const PrefillFfnBuffers &b, metal::MetalBuffer residual, metal::MetalBuffer output,
                              uint32_t rows) const {
  addPrefill(graph, b.normalized, *ffn.gate, b.gateScratch, b.sums, rows, b.scratch);
  addPrefillUpWithGate(graph, b.normalized, *ffn.up, b.gateScratch, b.intermediate, b.sums, b.downSums, rows,
                       b.scratch);
  addPrefillResidual(graph, b.intermediate, *ffn.down, residual, output, b.downSums, rows, b.scratch);
}
void Linear::addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &up,
                                  metal::MetalBuffer gateScratch, metal::MetalBuffer output,
                                  metal::MetalBuffer sums, metal::MetalBuffer downSums, uint32_t rows,
                                  LinearScratch scratch) const {
  add(graph,
      {.input = input, .output = output, .sums = sums, .gateScratch = gateScratch, .downSums = downSums,
       .scratch = scratch},
      up, prefillPlan(up, rows, LinearEpilogue::UpWithGate));
}

} // namespace splash::ops
