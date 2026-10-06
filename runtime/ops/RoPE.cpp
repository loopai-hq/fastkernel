// Modified by meowkernels.
#include "RoPE.hpp"

#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace splash::ops {

void RoPE::addTables(
    metal::CommandGraph &graph, metal::MetalBuffer targetPositions,
    metal::MetalBuffer draftPositions,
    metal::MetalBuffer targetInverseFrequencies,
    metal::MetalBuffer draftInverseFrequencies,
    metal::MetalBuffer targetCosine, metal::MetalBuffer targetSine,
    metal::MetalBuffer draftCosine, metal::MetalBuffer draftSine,
    RopeTableParams rows, uint32_t maximumRows) {
  // Draft-only tables (target_rows 0) serve SPLASH_DRAFT_AHEAD blocks.
  if ((!rows.target_rows && !rows.draft_rows) || rows.target_rows > maximumRows ||
      rows.draft_rows > maximumRows) {
    throw std::invalid_argument("invalid RoPE table row count");
  }
  const uint64_t targetTable = uint64_t{rows.target_rows} * SPLASH_TARGET_ROPE_PAIRS * sizeof(float);
  const uint64_t draftTable = uint64_t{rows.draft_rows} * SPLASH_DRAFT_ROPE_PAIRS * sizeof(float);
  // Target rows carry three positions (t, h, w): Qwen3.5's M-RoPE.
  requireBytes(targetPositions, uint64_t{rows.target_rows} * 3 * sizeof(uint32_t), "target RoPE position");
  requireBytes(draftPositions, uint64_t{rows.draft_rows} * sizeof(uint32_t), "draft RoPE position");
  requireBytes(targetInverseFrequencies, SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "target inverse frequency");
  requireBytes(draftInverseFrequencies, rows.draft_rows ? SPLASH_DRAFT_ROPE_PAIRS * sizeof(float) : 0,
               "draft inverse frequency");
  requireBytes(targetCosine, targetTable, "target RoPE cosine");
  requireBytes(targetSine, targetTable, "target RoPE sine");
  requireBytes(draftCosine, draftTable, "draft RoPE cosine");
  requireBytes(draftSine, draftTable, "draft RoPE sine");
  const uint64_t elements =
      std::max<uint64_t>(uint64_t{rows.target_rows} * SPLASH_TARGET_ROPE_PAIRS,
                         uint64_t{rows.draft_rows} * SPLASH_DRAFT_ROPE_PAIRS);
  graph.add("rope_build_tables",
            {std::move(targetPositions), std::move(draftPositions),
             std::move(targetInverseFrequencies),
             std::move(draftInverseFrequencies), std::move(targetCosine),
             std::move(targetSine), std::move(draftCosine),
             std::move(draftSine)},
            rows, {(elements + 255) / 256, 1, 1}, {256, 1, 1});
}

} // namespace splash::ops
