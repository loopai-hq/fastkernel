// Modified by meowkernels.
// Per-kernel GPU time attribution for the production executor.
//
//   decode-profile METALLIB MODEL_ROOT [--prompt-tokens N] [--cycles K]
//                  [--kv-format int8|bf16] [--sampled] [--single-request]
//
// SPLASH_DISPATCH_LIST=PATH appends each profiled decode cycle's dispatches
// (title, index, pipeline) to PATH, for comparing dispatch lists of builds.
//
// Drives the real model runtime with Metal dispatch profiling enabled, so
// every dispatch of a prefill command and B1 through B4 DFlash cycles
// is replayed as its own command and attributed to its pipeline.
// The fused (unprofiled) GPU time of the same work is reported alongside, so
// the gap between the sum of parts and the fused command shows how much a
// cycle pays in dispatch boundaries rather than kernel work.

#include "engine/Types.hpp"
#include "model/Runtime.hpp"
#include "ops/PageStorage.hpp"
#include "metal/BackendInstrumentation.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;
using namespace splash::engine;
using metal::BackendInstrumentation;

namespace {

struct Attribution final {
  uint64_t dispatches = 0;
  double gpuSeconds = 0.0;
};

using Table = std::map<std::string, Attribution>;

void accumulate(Table &table, std::span<const metal::DispatchTiming> timings) {
  for (const metal::DispatchTiming &timing : timings) {
    Attribution &entry = table[timing.pipelineName];
    ++entry.dispatches;
    entry.gpuSeconds += timing.gpuSeconds;
  }
}

// `fusedGpuSeconds` is already per unit of work; the table accumulated
// `divisor` units.
void print(const std::string &title, const Table &table, double divisor,
           double fusedGpuSeconds) {
  std::vector<std::pair<std::string, Attribution>> rows(table.begin(),
                                                        table.end());
  std::sort(rows.begin(), rows.end(), [](const auto &left, const auto &right) {
    return left.second.gpuSeconds > right.second.gpuSeconds;
  });
  double total = 0.0;
  uint64_t dispatches = 0;
  for (const auto &[_, entry] : rows) {
    total += entry.gpuSeconds;
    dispatches += entry.dispatches;
  }
  std::printf("\n== %s: %.2f ms fused, %.2f ms as %llu separate dispatches ==\n",
              title.c_str(), fusedGpuSeconds * 1e3, total * 1e3 / divisor,
              static_cast<unsigned long long>(dispatches / divisor));
  std::printf("%-44s %9s %10s %6s\n", "pipeline", "count", "gpu_ms", "share");
  for (const auto &[name, entry] : rows) {
    std::printf("%-44s %9.1f %10.3f %5.1f%%\n", name.c_str(),
                entry.dispatches / divisor, entry.gpuSeconds * 1e3 / divisor,
                entry.gpuSeconds / total * 100.0);
  }
}

std::vector<uint32_t> pageRange(uint32_t first, uint32_t count) {
  std::vector<uint32_t> result(count);
  for (uint32_t index = 0; index < count; ++index)
    result[index] = first + index;
  return result;
}

// A chat-formatted request that asks for a long answer, so decode cycles keep
// producing tokens instead of a stop token right after prefill. The user text
// repeats one sentence until the prompt reaches the requested length.
// `variant` rotates the repeated sentence: distinct prompts of one length.
std::vector<uint32_t> chatPrompt(uint32_t tokens, uint32_t variant = 0) {
  static constexpr std::array<uint32_t, 3> kUserHeader{248045, 846, 198};
  static constexpr std::array<uint32_t, 17> kSentence{
      7734, 264, 11346, 11,    7072,  12, 26829, 8627, 883,
      279,  3712, 314,   279,   12386, 19825, 13,    220};
  static constexpr std::array<uint32_t, 9> kAssistantHeader{
      248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271};
  if (tokens < kUserHeader.size() + kAssistantHeader.size() + 1)
    throw std::invalid_argument("prompt is too short for a chat request");
  std::vector<uint32_t> prompt(kUserHeader.begin(), kUserHeader.end());
  const uint32_t body = tokens - kUserHeader.size() - kAssistantHeader.size();
  for (uint32_t index = 0; index < body; ++index)
    prompt.push_back(kSentence[(index + 3 * variant) % kSentence.size()]);
  prompt.insert(prompt.end(), kAssistantHeader.begin(), kAssistantHeader.end());
  return prompt;
}

uint32_t parseCount(std::string_view text, std::string_view label) {
  uint32_t value = 0;
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      !value) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return value;
}

struct Lane final {
  uint64_t id = 0;
  uint32_t stateLane = 0;
  uint64_t position = 0;
  std::vector<uint32_t> pages;
  // The pages never change, so the page table keeps its first revision.
  uint64_t pageTableRevision = 1;
};

// Requests stop after this many new tokens (the A/B modes raise it).
uint32_t maxNewTokensPerRequest = 256;

void prefill(model::Runtime &executor, Lane &lane,
             std::span<const uint32_t> prompt, SamplingParameters sampling) {
  EngineRequest request;
  request.id = lane.id;
  request.prompt.assign(prompt.begin(), prompt.end());
  request.maxNewTokens = maxNewTokensPerRequest;
  request.sampling = sampling;
  executor.beginColdRequest(request.modelView(), lane.stateLane);
  uint32_t offset = 0;
  while (offset < prompt.size()) {
    const uint32_t count = std::min<uint32_t>(
        model::ExecutionLimits::prefillTokenBudget,
        static_cast<uint32_t>(prompt.size()) - offset);
    BatchPlan plan{.kind = WorkKind::Prefill,
                   .items = {{lane.id, count, offset}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{lane.id, offset, count, lane.pages,
                        lane.pageTableRevision};
    item.inputTokens = prompt.subspan(offset, count);
    auto results =
        executor.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    if (results.size() != 1 || results[0].consumedPromptTokens != count)
      throw std::runtime_error("prefill consumed the wrong row count");
    if (!results[0].failure.empty())
      throw std::runtime_error(results[0].failure);
    offset += count;
  }
  lane.position = prompt.size();
}

struct CycleTiming final {
  double gpuSeconds = 0.0;
  double wallSeconds = 0.0;
  uint64_t commands = 0;
  uint32_t outputs = 0;    // output tokens this cycle (all lanes)
  uint32_t retained = 0;   // of which kept with KV (the lanes' position advance)
  uint64_t tokenHash = 0;  // FNV-1a over the output token ids (the A/B work witness)
  bool finished = false;   // a lane's answer ended (A/B modes; otherwise it throws)
};

CycleTiming decodeCycle(metal::MetalBackend &backend,
                        model::Runtime &executor,
                        std::span<Lane> lanes, bool allowFinish = false) {
  const uint64_t submissionsBefore =
      BackendInstrumentation::submittedCommands(backend);
  const auto started = std::chrono::steady_clock::now();
  BatchPlan plan;
  plan.kind = WorkKind::Decode;
  std::vector<ModelBatchItem> items;
  for (Lane &lane : lanes) {
    plan.items.push_back({lane.id, 0, 0});
    items.push_back({lane.id, lane.position, 0, lane.pages,
                     lane.pageTableRevision});
  }
  auto results = executor.decode(plan, items);
  if (results.size() != lanes.size())
    throw std::runtime_error("decode width changed");
  CycleTiming timing;
  timing.tokenHash = 1469598103934665603ull;
  for (size_t index = 0; index < lanes.size(); ++index) {
    if (!results[index].failure.empty())
      throw std::runtime_error(results[index].failure);
    if (results[index].finished && !allowFinish)
      throw std::runtime_error("the answer ended before profiling finished");
    timing.finished = timing.finished || results[index].finished;
    const auto retained = static_cast<uint32_t>(results[index].outputTokens.size() -
                                                results[index].outputTokensWithoutKv);
    lanes[index].position += retained;
    timing.retained += retained;
    timing.outputs += static_cast<uint32_t>(results[index].outputTokens.size());
    for (const uint32_t token : results[index].outputTokens)
      timing.tokenHash = (timing.tokenHash ^ token) * 1099511628211ull;
    // SPLASH_PROFILE_TOKENS (diagnostic): each lane's output tokens, to
    // compare the decode of two builds.
    if (std::getenv("SPLASH_PROFILE_TOKENS")) {
      std::printf("CYCLE_TOKENS width=%zu lane=%zu", lanes.size(), index);
      for (const uint32_t token : results[index].outputTokens) std::printf(" %u", token);
      std::printf("\n");
    }
  }
  const auto finished = std::chrono::steady_clock::now();
  timing.gpuSeconds = executor.telemetry().lastDecodeGpuSeconds;
  timing.wallSeconds = std::chrono::duration<double>(finished - started).count();
  timing.commands = BackendInstrumentation::submittedCommands(backend) - submissionsBefore;
  return timing;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 3) {
        std::cerr << "usage: decode-profile METALLIB MODEL_ROOT "
                     "[--prompt-tokens N] [--cycles K] "
                     "[--kv-format int8|bf16] [--sampled] [--single-request]\n";
        return 2;
      }
      uint32_t promptTokens = 512;
      uint32_t cycles = 4;
      kv::Format format = kv::Format::Int8;
      bool sampled = false;
      bool singleRequest = false;
      for (int index = 3; index < argc;) {
        const std::string_view option(argv[index]);
        if (option == "--sampled" || option == "--single-request") {
          (option == "--sampled" ? sampled : singleRequest) = true;
          ++index;
          continue;
        }
        if (index + 1 >= argc)
          throw std::invalid_argument(std::string(option) + " requires a value");
        if (option == "--prompt-tokens")
          promptTokens = parseCount(argv[index + 1], "--prompt-tokens");
        else if (option == "--cycles")
          cycles = parseCount(argv[index + 1], "--cycles");
        else if (option == "--kv-format") {
          const std::string_view value(argv[index + 1]);
          if (value != "int8" && value != "bf16")
            throw std::invalid_argument("--kv-format takes int8 or bf16");
          format = value == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;
        } else
          throw std::invalid_argument("unknown option");
        index += 2;
      }
      // A/B modes run long greedy generations.
      const char *abMode = std::getenv("SPLASH_AB_MODE");
      if (abMode && std::getenv("SPLASH_AB_SWITCH")) {
        maxNewTokensPerRequest = 4096;
        std::setvbuf(stderr, nullptr, _IOLBF, 0);
      }
      const SamplingParameters sampling =
          sampled ? SamplingParameters{.temperature = 1.0F, .topP = 0.95F,
                                       .topK = 20, .seed = 20260926}
                  : SamplingParameters{};

      metal::MetalBackend backend(argv[1]);
      const std::filesystem::path root(argv[2]);
      model::LoadedModel model =
          model::loadModel(backend, root, model::inspectModelRoot(root));
      ops::ExecutionPlans operators(backend.capabilities());

      // Enough Page32 pages for four lanes (one with --single-request, which
      // fits small Macs) of prompt plus generated rows.
      const uint32_t pagesPerLane =
          (promptTokens + maxNewTokensPerRequest + model::ExecutionLimits::targetVerifyRows) /
              kv::kPageTokens +
          2;
      // Whole extents of the largest size the pool rule picks, as a large
      // pool's would be.
      const kv::Layout kvLayout = model.targetKvLayout(format);
      const uint32_t extentPages = kvLayout.maximumExtentPages();
      const uint32_t pageCount =
          (pagesPerLane * (singleRequest ? 1 : 4) + extentPages - 1) / extentPages * extentPages;
      MemoryGovernor governor(
          backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1,
          queryHostAvailableMemory, 0);
      kv::PageStorage pages(backend, governor.allocationAdmission(), kvLayout,
                            pageCount, extentPages);
      for (uint32_t extent = 0; extent < pageCount / extentPages; ++extent) {
        if (!pages.allocateExtent(extent))
          throw std::runtime_error("could not allocate the KV extents");
      }
      model::QwenStateStorage states(backend,
                                      governor.allocationAdmission(),
                                      model.stateLayout(), nullptr);
      model::RuntimeContext context{backend, model, pages, states, operators};
      model::Runtime executor(context);

      std::printf("device %s, %u prompt tokens, %u cycles per width, "
                  "policy %s\n",
                  backend.capabilities().deviceName.c_str(), promptTokens,
                  cycles,
                  sampled ? "sampled T=1 top_p=0.95 top_k=20 seed=20260926"
                          : "greedy T=0");

      std::array<Lane, 4> lanes;
      for (uint32_t index = 0; index < lanes.size(); ++index) {
        lanes[index] = {index + 1, index, 0,
                        pageRange(index * pagesPerLane, pagesPerLane)};
      }
      const std::vector<uint32_t> prompt = chatPrompt(promptTokens);

      // Warm prefill and B1 decode with real work before measuring.
      prefill(executor, lanes[0], prompt, sampling);
      static_cast<void>(decodeCycle(backend, executor, std::span<Lane>(&lanes[0], 1)));
      executor.end(lanes[0].id);

      // SPLASH_AB_SWITCH=VAR SPLASH_AB_MODE=lockstep|block (diagnostic A/B of
      // a default-on switch, ab-harness-v2): VAR=1 is the on arm, VAR=0 off.
      // SPLASH_AB_WIDTH=W (1 or 2) lanes per batch, SPLASH_AB_PROMPTS (6)
      // greedy prompts of the prompt length, SPLASH_AB_CYCLES (48) cycles
      // each, after a >= 3 GPU-s warm-up. One LS_STEP line per cycle.
      //  lockstep (exact switches): two groups of W lanes take the same
      //   prompts and decode alternately, one B_W cycle each, one arm per
      //   group (the on group alternates by prompt, the first arm by cycle),
      //   so cycle k of both arms does identical work about one cycle apart.
      //   Drafting is inline in both arms: a draft-ahead block launches only
      //   when every resident request is in its batch.
      //  block (switches lockstep can't pair, DRAFT_AHEAD): one group of W
      //   lanes runs the arms in blocks of SPLASH_AB_BLOCK cycles (4),
      //   alternating; slot = the cycle's position in its block (0 is the
      //   transition).
      if (abMode && std::getenv("SPLASH_AB_SWITCH")) {
        const char *abVariable = std::getenv("SPLASH_AB_SWITCH");
        const std::string_view mode(abMode);
        const auto count = [](const char *name, uint32_t fallback) {
          const char *text = std::getenv(name);
          return text ? parseCount(text, name) : fallback;
        };
        const uint32_t width = count("SPLASH_AB_WIDTH", 1);
        const uint32_t prompts = count("SPLASH_AB_PROMPTS", 6);
        const uint32_t abCycles = count("SPLASH_AB_CYCLES", 48);
        const uint32_t block = count("SPLASH_AB_BLOCK", 4);
        if ((mode != "lockstep" && mode != "block") || width > 2)
          throw std::invalid_argument("SPLASH_AB_MODE is lockstep or block, SPLASH_AB_WIDTH 1 or 2");
        const auto arm = [&](bool on) { setenv(abVariable, on ? "1" : "0", 1); };
        const auto group = [&](uint32_t index) { return std::span<Lane>(&lanes[index * width], width); };
        const auto start = [&](uint32_t index, uint32_t prompt) {
          for (uint32_t lane = 0; lane < width; ++lane)
            prefill(executor, lanes[index * width + lane], chatPrompt(promptTokens, prompt + lane), sampling);
        };
        const auto stop = [&](uint32_t index) {
          for (Lane &lane : group(index)) executor.end(lane.id);
        };
        const uint32_t groups = mode == "lockstep" ? 2 : 1;
        double warmGpu = 0.0;
        uint32_t warmCycles = 0;
        for (uint32_t index = 0; index < groups; ++index) start(index, 99);
        for (; warmGpu < 3.0; ++warmCycles) {
          const uint32_t index = warmCycles % groups;
          arm(groups == 2 ? index == 1 : (warmCycles / block) & 1);
          const CycleTiming t = decodeCycle(backend, executor, group(index), true);
          if (std::isfinite(t.gpuSeconds) && t.gpuSeconds > 0.0) warmGpu += t.gpuSeconds;
          if (t.finished) {
            stop(index);
            start(index, 99);
          }
        }
        for (uint32_t index = 0; index < groups; ++index) stop(index);
        std::fprintf(stderr, "LS_WARMUP cycles=%u gpu_s=%.3f rejected=0 mode=%s width=%u\n", warmCycles, warmGpu,
                     abMode, width);
        const auto step = [&](uint32_t prompt, uint32_t cycle, uint32_t index, bool on, uint32_t slot) {
          arm(on);
          const uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
          const CycleTiming t = decodeCycle(backend, executor, group(index), true);
          const uint64_t t1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
          const bool valid = std::isfinite(t.gpuSeconds) && t.gpuSeconds > 0.0;
          std::fprintf(stderr,
                       "LS_STEP prompt=%u cycle=%u lane=%u arm=%s slot=%u gpu_ms=%.6f wall_ms=%.6f t0_ns=%llu "
                       "t1_ns=%llu valid=%d outputs=%u retained=%u tokens=%016llx finished=%d\n",
                       prompt, cycle, index, on ? "on" : "off", slot, valid ? t.gpuSeconds * 1e3 : 0.0,
                       t.wallSeconds * 1e3, static_cast<unsigned long long>(t0),
                       static_cast<unsigned long long>(t1), int(valid), t.outputs, t.retained,
                       static_cast<unsigned long long>(t.tokenHash), int(t.finished));
          return t.finished;
        };
        for (uint32_t prompt = 0; prompt < prompts; ++prompt) {
          for (uint32_t index = 0; index < groups; ++index) start(index, prompt * width);
          bool done = false;
          for (uint32_t cycle = 0; cycle < abCycles && !done; ++cycle) {
            if (groups == 2) {
              const uint32_t onGroup = prompt & 1;
              for (uint32_t k = 0; k < 2; ++k) {
                const bool on = (cycle & 1) ? k == 0 : k == 1;  // off first on even cycles
                done = step(prompt, cycle, on ? onGroup : 1 - onGroup, on, k) || done;
              }
            } else {
              done = step(prompt, cycle, 0, ((cycle / block) + prompt) & 1, cycle % block);
            }
          }
          for (uint32_t index = 0; index < groups; ++index) stop(index);
        }
        unsetenv(abVariable);
        return 0;
      }

      // SPLASH_AB_SWITCH=VAR SPLASH_AB_B3_ROUNDS=N (diagnostic): three lanes
      // (SPLASH_AB_WIDTH, 1-4) decode as one batch for N rounds of two cycles,
      // VAR=0 and VAR=SPLASH_AB_ON (default 1), the order alternating by
      // round; one B3_STEP line per cycle. The lanes are
      // re-prefilled before their 256-token budget runs out. With an exact
      // switch the two cycles of a round do nearly the same work (adjacent
      // positions); an A/A run switches a variable nothing reads.
      const char *abSwitch = std::getenv("SPLASH_AB_SWITCH");
      const char *b3RoundsText = std::getenv("SPLASH_AB_B3_ROUNDS");
      if (abSwitch && b3RoundsText) {
        const uint32_t rounds = parseCount(b3RoundsText, "SPLASH_AB_B3_ROUNDS");
        const char *widthText = std::getenv("SPLASH_AB_WIDTH");
        const uint32_t width = widthText ? parseCount(widthText, "SPLASH_AB_WIDTH") : 3;
        if (width > (singleRequest ? 1u : 4u))
          throw std::invalid_argument("SPLASH_AB_WIDTH exceeds the allocated lanes");
        const char *onText = std::getenv("SPLASH_AB_ON");
        const std::span<Lane> batch(&lanes[0], width);
        for (Lane &lane : batch)
          prefill(executor, lane, prompt, sampling);
        for (uint32_t round = 0; round < rounds; ++round) {
          uint64_t generated = 0;
          for (const Lane &lane : batch)
            generated = std::max<uint64_t>(generated, lane.position - prompt.size());
          if (generated + 2 * model::ExecutionLimits::maximumStepTokens >= 256) {
            for (Lane &lane : batch) {
              executor.end(lane.id);
              prefill(executor, lane, prompt, sampling);
            }
          }
          for (uint32_t k = 0; k < 2; ++k) {
            const bool on = (round & 1) ? k == 0 : k == 1;
            // "0", not unset: most switches default on.
            setenv(abSwitch, on ? (onText ? onText : "1") : "0", 1);
            const CycleTiming t = decodeCycle(backend, executor, batch);
            std::printf("B3_STEP round=%u arm=%s gpu_ms=%.6f wall_ms=%.6f commands=%llu\n",
                        round, on ? "on" : "off", t.gpuSeconds * 1e3, t.wallSeconds * 1e3,
                        static_cast<unsigned long long>(t.commands));
          }
        }
        unsetenv(abSwitch);
        return 0;
      }

      // Prefill: fused timing first, then the attributed replay. Both cover
      // every chunk of the prompt.
      const double prefillBefore = executor.telemetry().totalPrefillGpuSeconds;
      prefill(executor, lanes[0], prompt, sampling);
      const double prefillFused =
          executor.telemetry().totalPrefillGpuSeconds - prefillBefore;
      executor.end(lanes[0].id);
      BackendInstrumentation::setDispatchProfiling(backend, true);
      prefill(executor, lanes[0], prompt, sampling);
      BackendInstrumentation::setDispatchProfiling(backend, false);
      Table prefillTable;
      accumulate(prefillTable,
                 BackendInstrumentation::takeDispatchProfile(backend));
      print("prefill " + std::to_string(promptTokens) + " rows", prefillTable,
            1.0, prefillFused);

      auto profileWidth = [&](const char *title, std::span<Lane> active) {
        std::vector<CycleTiming> fused;
        for (uint32_t cycle = 0; cycle < cycles; ++cycle)
          fused.push_back(decodeCycle(backend, executor, active));
        std::sort(fused.begin(), fused.end(),
                  [](const CycleTiming &left, const CycleTiming &right) {
                    return left.gpuSeconds < right.gpuSeconds;
                  });
        const CycleTiming median = fused[fused.size() / 2];
        std::printf("\n%s: median fused gpu %.2f ms, wall %.2f ms, %llu "
                    "command(s) per cycle\n",
                    title, median.gpuSeconds * 1e3, median.wallSeconds * 1e3,
                    static_cast<unsigned long long>(median.commands));
        BackendInstrumentation::setDispatchProfiling(backend, true);
        for (uint32_t cycle = 0; cycle < cycles; ++cycle)
          static_cast<void>(decodeCycle(backend, executor, active));
        BackendInstrumentation::setDispatchProfiling(backend, false);
        const std::vector<metal::DispatchTiming> profile =
            BackendInstrumentation::takeDispatchProfile(backend);
        if (const char *listPath = std::getenv("SPLASH_DISPATCH_LIST")) {
          std::ofstream list(listPath, std::ios::app);
          for (size_t index = 0; index < profile.size(); ++index)
            list << title << '\t' << index << '\t' << profile[index].pipelineName << '\n';
        }
        Table table;
        accumulate(table, profile);
        print(title, table, cycles, median.gpuSeconds);
      };
      profileWidth("B1 decode cycle", std::span<Lane>(&lanes[0], 1));
      // Add one lane at a time so M16 and M24 paths are measured too.
      for (uint32_t index = 1; index < (singleRequest ? 1 : lanes.size()); ++index) {
        prefill(executor, lanes[index], prompt, sampling);
        const std::string title = "B" + std::to_string(index + 1) + " decode cycle";
        profileWidth(title.c_str(), std::span<Lane>(lanes.data(), index + 1));
      }

      for (Lane &lane : lanes)
        executor.end(lane.id);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "decode-profile: " << error.what() << '\n';
      return 1;
    }
  }
}
