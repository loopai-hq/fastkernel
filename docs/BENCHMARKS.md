# Benchmarks

We measured every number here on our own Macs. The tests use Qwen3.8-27B (4-bit), plus one run with Qwen3.6-35B-A3B.

## How to read the numbers

| Term | What it means |
|---|---|
| tok/s | Tokens per second: how fast the answer is written (the decode speed). A token is a word or part of a word. |
| Sampled | The model picks each token at random, weighted by its own probabilities (temperature 1). |
| Greedy | The model always picks its most likely next token. |
| Draft model | A small model that guesses the next few tokens. The full model checks every guess before it keeps it. |
| Context | The most text the model can hold at once: your prompt plus its answer. |

Most M5 Max tests use the same 6 prompts. They cover chat, math, code, a code file, a long agent task and Portuguese.
Some tables call the Portuguese prompt "multilingual". The Splash 1.3.0 tests add 4 more chat prompts and 2
long-context prompts, 12 in all. Each test's fine print, such as seeds and builds, is in its "Details" fold. A 95%
confidence interval, written [low, high], is the range the true average falls in, with 95% confidence.

## M5 Max (40-core GPU, 128 GB)

### Pulsar vs lithos-metal, greedy, 10 prompts

On our 128 GB M5 Max running macOS 27.2, Pulsar delivered 1.30–1.46× the client-observed decode throughput of
lithos-metal in every session we ran, and was faster on every prompt in every session (geometric mean over 10 prompts;
each one ABBA session, greedy decoding, thinking off, output capped at 1,024 tokens).

| Session | Pulsar | lithos-metal | Pulsar tok/s | lithos-metal tok/s | Pulsar is (95% range) |
|---|---|---|---:|---:|---|
| 1: 2026-10-09 03:42 | 1.1.3 | main 236475f, built from source | 108.5 | 80.2 | **1.36×** (1.26–1.48×) |
| 2: 2026-10-09 14:27 | 1.1.4 | 0.1.2, official release | 95.6 | 74.7 | **1.30×** (1.23–1.38×) |
| 3: 2026-10-09 15:37 | 1.1.4 | 0.1.2, official release | 105.8 | 73.3 | **1.46×** (1.36–1.56×) |
| 4: 2026-10-10 00:12 | 1.1.5 | 0.1.2, official release | 108.3 | 75.7 | **1.44×** (1.35–1.53×) |

tok/s is the mean of the 10 prompts' means; "Pulsar is" is the geometric mean of the per-run ratios, and its 95% range
is a bootstrap over the 10 prompts within that session. The sessions differ in lithos-metal's build and warm-up (see
Settings below). Median time to first token in session 4: 0.22 s on Pulsar, 0.39 s on lithos-metal.

Per prompt, decode tok/s as lithos-metal / Pulsar (ratio), mean of two runs per prompt:

| Prompt | Session 1 | Session 2 | Session 3 | Session 4 |
|---|---:|---:|---:|---:|
| lithos-metal's launch-video prompt (tip calculator) | 105.2 / 136.0 (1.29×) | 96.6 / 116.0 (1.20×) | 100.6 / 130.1 (1.29×) | 104.2 / 131.8 (1.26×) |
| code | 144.3 / 180.8 (1.25×) | 140.3 / 159.8 (1.14×) | 136.8 / 173.0 (1.26×) | 137.9 / 182.9 (1.33×) |
| math | 116.3 / 166.0 (1.43×) | 112.6 / 147.2 (1.31×) | 96.4 / 163.7 (1.71×) | 104.8 / 168.6 (1.61×) |
| code file, 2,247 tokens | 84.8 / 101.8 (1.20×) | 79.7 / 99.1 (1.24×) | 78.5 / 107.4 (1.37×) | 79.4 / 106.2 (1.34×) |
| chat 1 | 67.0 / 85.0 (1.27×) | 57.6 / 75.1 (1.30×) | 56.4 / 82.4 (1.46×) | 57.0 / 84.7 (1.49×) |
| chat 2 | 61.3 / 80.4 (1.31×) | 55.8 / 70.0 (1.25×) | 54.8 / 77.3 (1.41×) | 57.0 / 78.3 (1.37×) |
| chat 3 | 56.3 / 91.4 (1.62×) | 52.4 / 78.0 (1.49×) | 53.2 / 88.2 (1.66×) | 54.0 / 89.3 (1.65×) |
| chat 4 | 55.7 / 88.8 (1.60×) | 52.8 / 74.7 (1.41×) | 54.9 / 85.3 (1.55×) | 56.5 / 86.5 (1.53×) |
| chat 5 | 55.7 / 92.4 (1.66×) | 51.7 / 79.5 (1.54×) | 53.8 / 88.6 (1.65×) | 55.8 / 91.0 (1.63×) |
| multilingual | 55.1 / 62.2 (1.13×) | 47.5 / 56.9 (1.20×) | 48.0 / 62.0 (1.29×) | 50.2 / 63.7 (1.27×) |

Decode tok/s = (completion tokens − 1) / (last − first streamed chunk).

<details>
<summary>Settings and limits</summary>

- **Machine:** M5 Max (40-core GPU, 128 GB), macOS 27.2, AC power, thermal state nominal. No other model server was
  running.
- **Order:** one server at a time: Pulsar, lithos-metal, lithos-metal, Pulsar. Each block ran at least 120 s of
  warm-up requests, then the 10 prompts.
- **Pulsar:** the release package (1.1.3, 1.1.4, then 1.1.5), `SPLASH_DRAFT_HEAD_IDS=$PWD/data/head-ranked.u32
  ./pulsar serve --model incoai/Qwen3.8-27B-Splash`, plus `--port` and `--max-context 40K`. 1.1.4 runs the same GPU
  code as 1.1.3 (identical metallib).
- **lithos-metal, session 1:** main 236475f (14 commits after the v0.1.2 tag), built from source, `lithos-metal serve
  --model nvidia/Qwen3.8-27B-NVFP4` plus `--port` and `--max-context 34816` (its default is 32,768). Its warm-up was 2
  requests per block.
- **lithos-metal, sessions 2 to 4:** the official v0.1.2 prebuilt, installed like its Homebrew formula, at its
  default context, with at least 6 warm-up requests per block. lithos-metal compiles one session per sampling seed; in
  session 2 each warm-up request used its own seed, and in sessions 3 and 4 every warm-up request used the seed then
  measured, with a ~2.5K-token prompt every third request.
- **Checkpoints:** lithos-metal runs NVIDIA's checkpoint (revision 482ca0f) with its DSpark draft head
  (LithosAI/Qwen3.8-27B-DSpark-NVFP4, b169bc4). That checkpoint is NVFP4 in the MLP and LM head and FP8 in the attention
  and linear-attention projections, so it has a 17.6 GB logical target-weight payload per verification pass, against
  14.4 GB for the 4-bit package Pulsar runs (mlx-community/Qwen3.8-27B-4bit, packed by Inco AI) with DFlash 2.
- **lithos-metal on this Mac:** its own round benchmark (v0.1.2) measured 48.04 ms per round at 128 tokens, against the
  42.6 ms its authors published for a 48 GB M5 Max on macOS 26.5.1. In session 2 its serving rounds (48–56 ms) matched
  that tool.
- **Requests:** greedy (temperature 0), reasoning_effort "none" (lithos-metal serves without thinking), streamed, at most
  1,024 tokens. Four prompts reach the cap on both engines, including the tip calculator, so this measures decode
  speed over a bounded answer, not time to a finished answer.
- **Prompts:** the 12-prompt Splash 1.3.0 set without its three long prompts, plus the prompt from lithos-metal's launch
  video. Answers differ in length and content between the engines, so this compares delivered throughput, not
  identical work. Answer quality was not compared.

</details>

### Pulsar 1.1.5 vs Splash 1.3.0, MTPLX, AX Engine, MLX-LM and llama.cpp

Each engine ran against the Pulsar 1.1.5 release package, one pair per session, with the same prompts, seeds, settings
and engine versions as the Pulsar 1.1.0 runs below. Against Splash 1.3.0, one server ran at a time in alternating blocks
(ABBA, 8 blocks, the 12 prompts). Against the others, both servers ran together and took turns on each prompt: the 6
standard prompts sampled, or 5 of them greedy for AX Engine.

| Engine | Answers | Its tok/s | Pulsar 1.1.5 tok/s | Pulsar 1.1.5 is |
|---|---|---:|---:|---|
| Splash 1.3.0 (DFlash 2) | sampled, 12 prompts | 86.5 | 98.6 | **1.13× faster** (95% range 1.09–1.17×) |
| MTPLX 2.12.0 (Bare-Speed 4-bit + MTP) | sampled | 64.9 | 125.0 | **1.93× faster** (95% range 1.70–2.10×) |
| AX Engine 7.5.7 (MXFP4 + MTP) | greedy | 40.1 | 131.0 | **3.27× faster** (95% range 2.91–3.67×) |
| MLX-LM 0.31.3 (4-bit) | sampled | 30.0 | 115.0 | **3.84× faster** (95% range 3.04–4.63×) |
| llama.cpp 0.5.0 (Q4_K_M GGUF) | sampled | 27.1 | 121.7 | **4.49× faster** (95% range 3.53–5.40×) |

Conditions: 2026-10-10, 03:59–06:28 IST, on AC power. Every block started with the Mac at thermal state "nominal".
In chat, Pulsar 1.1.5 writes 74.9 tok/s, 1.13× Splash 1.3.0's 66.4.

<details>
<summary>Details</summary>

- Splash row: mean of per-prompt means; the ratio is the geometric mean of matched prompt-and-seed pairs, with a 95%
  range clustered by seed block. Other rows: mean of per-request client decode speeds, ratio of those means, with a
  95% bootstrap range over prompts.
- Pulsar 1.1.5: the release package (sha256 6a01d6c4…) with the Quick start flags.

Per prompt, in tok/s (Pulsar 1.1.5 / the other engine, mean of 2 seeds):

| Prompt | vs MTPLX | vs AX Engine | vs MLX-LM | vs llama.cpp |
|---|---:|---:|---:|---:|
| chat | 77.5 / 49.7 | 85.6 / 33.0 | 74.7 / 30.4 | 77.3 / 27.2 |
| math | 173.9 / 79.0 | 169.6 / 54.9 | 163.6 / 31.1 | 169.8 / 28.4 |
| code | 167.5 / 79.4 | 168.6 / 52.5 | 146.4 / 30.8 | 158.4 / 28.1 |
| code file | 140.2 / 70.3 | 136.0 / 34.4 | 122.9 / 30.0 | 137.8 / 27.7 |
| agent (32K prompt) | 91.2 / 47.5 | | 90.4 / 26.0 | 89.8 / 23.5 |
| multilingual | 99.7 / 63.7 | 95.3 / 25.5 | 91.7 / 31.4 | 96.8 / 27.7 |

</details>

### Sampled answers: Pulsar 1.1.0 vs Splash 1.3.0 and Pulsar 1.0.0, 12 prompts

Pulsar 1.1.0 is this release, built on Splash 1.3.0. All three servers ran at the same time, taking turns on each
prompt.

| Engine | tok/s | |
|---|---:|---|
| **Pulsar 1.1.0** | **99.2** | |
| Pulsar 1.0.0 | 100.8 | same speed: 0.98× (95% range 0.96–1.00×) |
| Splash 1.3.0 | 89.0 | Pulsar 1.1.0 is **1.11× faster** (95% range 1.07–1.14×) |

| | Pulsar 1.1.0 | Pulsar 1.0.0 | Splash 1.3.0 |
|---|---:|---:|---:|
| Chat (5 prompts), tok/s | **76.7** | 79.0 | 68.5 |

Time to first token, for each prompt's first request in the run (nothing cached), in seconds:

| Prompt | Pulsar 1.1.0 | Pulsar 1.0.0 | Splash 1.3.0 |
|---|---:|---:|---:|
| code file, 2,247 tokens | 3.1 | 3.2 | 2.7 |
| long context 2, 4,702 tokens | 6.3 | 6.5 | 5.4 |
| long context 1, 6,645 tokens | 8.5 | 8.8 | 7.7 |
| agent, 32,678 tokens | 45.8 | 46.3 | 40.1 |
| code file again (second request, prompt cache) | 0.1 | 0.4 | 0.1 |

Each prompt runs twice, and the second request can reuse the first one's prompt cache, so the first four rows take
the first request of each prompt, and the last row the code file's second request.

Conditions: 2026-10-07, 03:39–04:01 IST, on AC power, Chrome open.

<details>
<summary>Details</summary>

- Settings: temperature 1, top-p 0.95, top-k 20, up to 1,024 tokens, reasoning effort medium, seeds 20261501 and
  20261502. The engines took turns on each prompt, and the client timed each answer.
- tok/s is the mean over prompts of each prompt's mean. A ratio is the geometric mean of the per-prompt ratios, and
  its 95% range is a bootstrap over prompts. In chat, Pulsar 1.1.0 is 1.12× Splash 1.3.0
  and 0.97× Pulsar 1.0.0.
- Pulsar 1.1.0: a release-candidate build with the same GPU kernels as the release, with the official `incoai/Qwen3.8-27B-Splash` package and the Quick
  start flags (`SPLASH_DRAFT_HEAD_IDS`). Neural Engine prefill was on, its default; its startup timing on this Mac kept prefill on the GPU alone, where no split was faster.
- Pulsar 1.0.0: the release package with the same model and flags.
- Splash 1.3.0: the release package, running `mlx-community/Qwen3.8-27B-4bit` (revision 3e6447f) with its DFlash 2
  draft, the same weights as the Splash package. Its defaults stayed on, including the Neural Engine prefill split and
  int8 K/V.
- Each server had a 40K-token context and a 40 GB memory limit.

Per prompt, in tok/s:

| Prompt | Pulsar 1.1.0 | Pulsar 1.0.0 | Splash 1.3.0 |
|---|---:|---:|---:|
| chat 1 | 74.7 | 78.6 | 65.1 |
| chat 2 | 75.6 | 74.8 | 65.8 |
| chat 3 | 77.0 | 81.6 | 68.4 |
| chat 4 | 77.3 | 78.8 | 70.1 |
| chat 5 | 78.8 | 81.4 | 73.2 |
| math | 166.4 | 164.9 | 145.9 |
| code | 153.9 | 150.0 | 127.5 |
| code file | 125.4 | 133.5 | 115.6 |
| agent (32K prompt) | 81.5 | 87.6 | 85.1 |
| multilingual | 92.7 | 90.4 | 82.9 |
| long context 1 | 88.9 | 88.4 | 82.4 |
| long context 2 | 97.7 | 99.7 | 86.0 |

</details>

### Sampled answers: Pulsar 1.0.0 vs Splash 1.3.0, 12 prompts

The same 12 prompts and settings, with the Pulsar 1.0.0 release package against Splash 1.3.0.

| Engine | tok/s | |
|---|---:|---|
| **Pulsar 1.0.0** | **92.5** | |
| Splash 1.3.0 | 82.6 | Pulsar 1.0.0 is **1.12× faster** (95% range 1.09–1.15×) |

In chat (5 prompts), Pulsar 1.0.0 writes 72.3 tok/s against 62.9, 1.15× faster. On the 6 prompts of the tests
below, it writes 109.4 against 97.8.

Conditions: 2026-10-06, 21:24–21:35 IST, on AC power, Chrome and other apps open.

<details>
<summary>Details</summary>

- Settings and the Splash 1.3.0 setup are as in the test above. Both servers ran at the same time, taking turns on
  each prompt.

Per prompt, in tok/s:

| Prompt | Pulsar 1.0.0 | Splash 1.3.0 |
|---|---:|---:|
| chat 1 | 76.7 | 61.8 |
| chat 2 | 69.0 | 60.4 |
| chat 3 | 72.1 | 62.4 |
| chat 4 | 70.1 | 63.3 |
| chat 5 | 73.8 | 66.6 |
| math | 153.4 | 138.4 |
| code | 138.9 | 120.0 |
| code file | 122.5 | 108.6 |
| agent (32K prompt) | 80.6 | 79.4 |
| multilingual | 84.0 | 78.4 |
| long context 1 | 79.3 | 74.6 |
| long context 2 | 90.2 | 77.7 |

</details>

### Pulsar 1.1.0 vs MTPLX, AX Engine, MLX-LM and llama.cpp

Each engine ran side by side with the Pulsar 1.1.0 release package, one pair per session, taking turns on each
prompt: the 6 standard prompts sampled, or 5 of them greedy for AX Engine (it takes prompts of up to 16K tokens, so the
agent task is left out).

| Engine | Answers | Its tok/s | Pulsar 1.1.0 tok/s | Pulsar 1.1.0 is |
|---|---|---:|---:|---|
| MTPLX 2.12.0 (Bare-Speed 4-bit + MTP) | sampled | 65.7 | 118.2 | **1.80× faster** (95% range 1.62–1.97×) |
| AX Engine 7.5.7 (MXFP4 + MTP) | greedy | 38.3 | 119.6 | **3.12× faster** (95% range 2.78–3.48×) |
| MLX-LM 0.31.3 (4-bit) | sampled | 30.9 | 114.9 | **3.72× faster** (95% range 2.93–4.51×) |
| llama.cpp 0.5.0 (Q4_K_M GGUF) | sampled | 26.7 | 110.7 | **4.14× faster** (95% range 3.23–5.07×) |

Conditions: 2026-10-07, 04:33–05:33 IST, on AC power, Chrome and LM Studio open. The Mac was warm ("fair") for most
requests.

<details>
<summary>Details</summary>

- Settings: tools/head2head's request body (temperature 1, top-p 0.95, top-k 20, up to 1,024 tokens, reasoning effort
  medium; temperature 0 for AX Engine), seeds 20261501 and 20261502, the engine order alternating by prompt and seed.
  tok/s is the mean of the per-request client decode speeds; the ratio is of those means, with a 95% bootstrap range.
- Pulsar 1.1.0: the release package, the official `incoai/Qwen3.8-27B-Splash` package and the Quick start flags.
- MTPLX 2.12.0: `Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed`, its server's defaults (04:49–04:55; 22 of 24 requests warm).
- AX Engine 7.5.7: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP` with `--mlx-mtp-policy required --speculation-profile
  coding` (05:28–05:33; 20 of 20 requests warm).
- MLX-LM 0.31.3 with mlx 0.32.3: `mlx-community/Qwen3.8-27B-4bit`, the weights Pulsar's package was converted from
  (05:19–05:28; 17 of 24 requests warm).
- llama.cpp 0.5.0 (build 11146, Metal): `lmstudio-community/Qwen3.8-27B-GGUF` Q4_K_M, a 40,960-token context and one
  slot (04:33–04:42; 24 of 24 requests warm).

Per prompt, in tok/s (Pulsar 1.1.0 / the other engine, mean of 2 seeds):

| Prompt | vs MTPLX | vs AX Engine | vs MLX-LM | vs llama.cpp |
|---|---:|---:|---:|---:|
| chat | 78.8 / 50.2 | 77.4 / 32.2 | 78.1 / 30.9 | 70.2 / 27.3 |
| math | 167.7 / 80.1 | 155.4 / 52.0 | 165.0 / 32.4 | 158.1 / 27.5 |
| code | 154.4 / 77.2 | 156.8 / 49.8 | 151.9 / 31.9 | 148.8 / 27.6 |
| code file | 121.3 / 71.8 | 122.8 / 33.2 | 120.6 / 31.5 | 114.4 / 27.0 |
| agent (32K prompt) | 90.5 / 52.9 | | 86.0 / 27.5 | 83.7 / 23.4 |
| multilingual | 96.3 / 61.9 | 85.8 / 24.4 | 88.1 / 31.4 | 89.1 / 27.5 |

</details>

### Qwen3.6-35B-A3B: Pulsar 1.1.0 vs Splash 1.3.0, 6 prompts

| Answers | Pulsar 1.1.0 tok/s | Splash 1.3.0 tok/s | Ratio (95% range) |
|---|---:|---:|---|
| Sampled | 268.1 | 260.9 | 1.03× (1.00–1.05×) |
| Greedy | 299.6 | 276.1 | 1.09× (0.99–1.23×) |

Conditions: 2026-10-07, 05:33–05:36 IST, on AC power, Chrome and LM Studio open, the Mac warm ("fair").

<details>
<summary>Details</summary>

- Both servers ran the official `incoai/Qwen3.6-35B-A3B-Splash` package in one session, taking turns on each prompt,
  with the settings of the test above (temperature 1 and 0).

Per prompt, in tok/s (Pulsar 1.1.0 / Splash 1.3.0, mean of 2 seeds):

| Prompt | Sampled | Greedy |
|---|---:|---:|
| chat | 199.7 / 203.5 | 233.6 / 200.7 |
| math | 351.9 / 333.8 | 362.7 / 374.1 |
| code | 377.5 / 357.4 | 358.9 / 360.3 |
| code file | 270.3 / 262.8 | 385.4 / 275.5 |
| agent (32K prompt) | 166.1 / 168.1 | 192.3 / 185.0 |
| multilingual | 242.9 / 240.0 | 264.5 / 260.9 |

</details>

### Demo videos

The videos at the top of the README replay measured token timings in real time. Each engine's server ran alone, one
after another, with greedy answers and 3 runs per prompt (2026-10-07, 04:57–05:14 IST, on AC power, the Mac cool
("nominal") for every run). The prompt for each video was picked by a rule fixed before any run.

| Video | Prompt | Pulsar 1.1.0 | Splash 1.3.0 | MLX-LM |
|---|---|---:|---:|---:|
| Same answer | a Python command-line todo app; all three engines write the same 708 tokens | 3.97 s (184.5 tok/s) | 4.07 s (179.7 tok/s) | 23.1 s (30.8 tok/s) |
| Galaxy app | a one-shot particle-galaxy web app | 3,124 tokens in 20.9 s (150.4 tok/s) | the same 3,124 tokens in 22.1 s (141.2 tok/s) | |
| Code edit | an edit to a pasted 210-line Python file | 8.2 s (315.6 tok/s) | 11.7 s (196.4 tok/s) | 64.5 s (30.9 tok/s) |

<details>
<summary>Details</summary>

- Same answer: the first of 4 prompts whose text matched on all three engines; the median-finish run of each.
- Galaxy app: Splash 1.3.0 reached the 3,500-token limit, so the video compares the time to the same token count. The
  two texts differ from about token 322.
- Code edit: each engine's first run, so no prompt cache was reused. The texts differ from line 4 (1,894 tokens for
  Pulsar 1.1.0, 1,884 for the others).
- Times are from the run each video shows; tok/s is each engine's median decode speed over its 3 runs.

</details>

The tests below ran Pulsar 1.0.0 and its pre-release builds, 2026-09-26 to 2026-09-30.

### Sampled answers vs Splash 1.1.0, 6 prompts

Splash 1.1.0 was the latest Splash on 2026-09-30, the day of this test. Both servers ran at the same time, taking turns
on each prompt, with other apps and servers closed. Pulsar is the released 1.0.0 package.

| Engine | tok/s | |
|---|---:|---|
| **Pulsar** | **123.2** | |
| Splash 1.1.0 | 93.8 | Pulsar is **1.31× faster** (95% range 1.28–1.35×) |

| Task | Pulsar | Splash 1.1.0 | Pulsar is |
|---|---:|---:|---:|
| chat | 80.6 | 57.2 | 1.41× |
| code | 158.8 | 116.8 | 1.36× |
| multilingual | 96.4 | 73.1 | 1.32× |
| math | 172.7 | 132.5 | 1.30× |
| code file | 140.0 | 110.4 | 1.27× |
| agent (32K prompt) | 90.8 | 72.9 | 1.25× |

- Settings: temperature 1, top-p 0.95, top-k 20, up to 1,024 tokens, seeds 20261501 and 20261502; 12 of 12 pairs.

### Sampled answers vs Splash and MTPLX, 6 prompts

We sent the 6 prompts, twice each, to each engine in turn. MTPLX uses MTP, the model's built-in way to guess its next
few tokens.

| Engine | tok/s | |
|---|---:|---|
| **Pulsar** | **123.5** | |
| Splash 1.0.2 | 86.9 | Pulsar is **1.42× faster** |
| MTPLX (4-bit + MTP) | 64.0 | Pulsar is **1.93× faster** |

Conditions: nothing else used the GPU during the run.

<details>
<summary>Details</summary>

- Settings: temperature 1, top-p 0.95, top-k 20, up to 1,024 tokens per answer.
- Seeds 20261501 and 20261502. A seed fixes the random picks, so a run can be repeated.
- The engines took turns on each prompt. The client (the program sending the prompts) timed each answer, and we
  averaged over all requests.
- Builds: a Pulsar 1.0.0 pre-release build of 2026-09-27; Splash 1.0.2 as installed; MTPLX from a local install (Bare-Speed
  4-bit + MTP). This run didn't record the thermal state or whether Chrome was open.

Per prompt, in tok/s:

| Prompt | Pulsar | Splash | MTPLX |
|---|---:|---:|---:|
| math | 163 | 122 | 79 |
| code | 168 | 116 | 76 |
| code file | 133 | 95 | 70 |
| agent | 103 | 70 | 48 |
| Portuguese | 93 | 66 | 62 |
| chat | 81 | 53 | 50 |

</details>

### Reading a long prompt vs MTPLX

Before an engine writes anything, it reads your whole prompt. We timed that step on a new 16.7K-token prompt, three
times on each engine. Speed here is prompt tokens read per second.

| Engine | Prompt tok/s | |
|---|---:|---|
| **Pulsar** | **775** | |
| MTPLX (4-bit + MTP) | 704 | Pulsar is **1.10× faster** |

Conditions: a new prompt each time, so neither engine could reuse earlier work.

<details>
<summary>Details</summary>

- 16,693 prompt tokens. Each request carried a new random string (a nonce). So neither engine could reuse work saved
  from an earlier prompt (no prefix-cache hit).
- Each answer was one token long. The engines took turns.
- Seconds per request: Pulsar 22.37 / 21.60 / 20.69; MTPLX 24.82 / 23.65 / 22.67. The speeds above are the averages.

</details>

### Greedy answers vs AX Engine, 5 prompts

We sent 5 of the prompts, twice each, to Pulsar and to AX Engine 7.5.7. AX takes prompts of up to 16K tokens, so
we left out the long agent task.

| Engine | tok/s | |
|---|---:|---|
| **Pulsar** | **126.0** | |
| AX Engine 7.5.7 (MXFP4 + MTP) | 40.5 | Pulsar is **3.11× faster** |

Conditions: nothing else used the GPU during the run.

<details>
<summary>Details</summary>

- AX model: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP`, the only public AX build of this model. MXFP4 is its
  4-bit number format.
- AX flags: `--mlx-mtp-policy required --speculation-profile coding`. AX's own guessing was on, and it kept 92% of its
  guesses.
- Pulsar: a 1.0.0 pre-release build of 2026-09-26. With the long agent task included, Pulsar's greedy average is 120.8.
- AX's answers are shorter and show no written-out reasoning. That changes how long an answer is, not how fast it is
  written.

Per prompt, in tok/s:

| Prompt | Pulsar | AX |
|---|---:|---:|
| chat | 82.6 | 33.9 |
| Portuguese | 93.3 | 25.7 |
| math | 167.2 | 55.6 |
| code | 150.7 | 52.2 |
| code file | 136.2 | 35.2 |

</details>

### Sampled answers vs AX Engine, 5 prompts

The same 5 prompts, once each, with sampled answers.

| Engine | tok/s | |
|---|---:|---|
| **Pulsar** | **115.0** | |
| AX Engine 7.5.7 (MXFP4 + MTP) | 27.9 | Pulsar is **4.12× faster** |

Conditions: as in the greedy run above.

<details>
<summary>Details</summary>

- One seed per prompt.
- AX uses its guessing only for greedy answers. Here it wrote 2,824 of its 2,928 steps without it.

</details>

### Sampled answers vs llama.cpp and MLX-LM, 6 prompts

We ran Pulsar, llama.cpp and MLX-LM side by side. We sent the 6 prompts, twice each, to each engine in turn.

| Engine | tok/s | |
|---|---:|---|
| **Pulsar** | **116.8** | |
| MLX-LM 0.31.3 | 31.2 | Pulsar is **3.74× faster** |
| llama.cpp 0.5.0 | 27.0 | Pulsar is **4.33× faster** |

Conditions: Chrome was open for part of the run, and the Mac was warm ("fair") for most requests.

<details>
<summary>Details</summary>

Setup:

- MLX-LM ran `mlx-community/Qwen3.8-27B-4bit` (4-bit, group 64, 4.50 bits per weight). Pulsar's package was
  converted from these same weights.
- llama.cpp ran `lmstudio-community/Qwen3.8-27B-GGUF` Q4_K_M (4.92 bits per weight). GGUF is llama.cpp's model file
  format, and Q4_K_M is one of its 4-bit types. Of the public Q4_K_M files, this one is closest to 4.5 bits; Qwen
  publishes no official GGUF.
- Versions: llama.cpp 0.5.0 (build 11146, Metal); mlx-lm 0.31.3 with mlx 0.32.3.
- Both ran at their default settings. Neither turns on speculative decoding (draft-and-check) by default. llama.cpp
  used our server's 40,960-token context and one slot (one request at a time).
- MLX-LM's server ignores the request's `reasoning_effort`. So its prompts were 38–42 tokens longer, with the chat
  template's default instruction. That changes the text, not the speed per token.

Results:

- 95% confidence intervals: Pulsar 116.8 [92.7, 142.8]; ratio vs llama.cpp 4.33 [3.46, 5.20], vs MLX-LM 3.74
  [3.01, 4.47].
- Conditions: the Mac was cool ("nominal") for the first 4 requests, then warm ("fair") for 32 of 36. Chrome was open
  for 16 of 36 requests.
- Pulsar: the 1.0.0 pre-release build of 2026-09-29, with the release settings.

Per prompt, in tok/s:

| Prompt | Pulsar | MLX-LM | llama.cpp |
|---|---:|---:|---:|
| chat | 80 | 31 | 28 |
| math | 161 | 33 | 28 |
| code | 150 | 32 | 28 |
| code file | 132 | 32 | 27 |
| agent | 87 | 28 | 24 |
| multilingual | 91 | 32 | 27 |

</details>

### Sampled and greedy answers vs Splash, 6 prompts

We ran the first Splash test again with the same prompts. Then we repeated it with greedy answers.

| Engine | Sampled tok/s | Greedy tok/s | |
|---|---:|---:|---|
| **Pulsar** | **121.0** | **124.8** | |
| Splash 1.0.2 | 86.6 | 90.3 | Pulsar is **1.40×** (sampled) and **1.38×** (greedy) **faster** |

Conditions: Chrome was open, and the Mac ran warm ("fair" thermal state) for most requests.

<details>
<summary>Details</summary>

95% confidence intervals (bootstrap over prompts):

| Answers | Pulsar | Splash | Ratio |
|---|---|---|---|
| Sampled | 121.0 [95.1, 148.9] | 86.6 [67.3, 107.2] | 1.398 [1.343, 1.452] |
| Greedy | 124.8 [98.3, 152.3] | 90.3 [72.2, 108.7] | 1.382 [1.316, 1.457] |

The speed ranges are wide because the prompts range from 54 to 171 tok/s. The ratio compares each prompt with itself,
so its range is narrow.

Per prompt, in tok/s:

| Prompt | Pulsar sampled | Splash sampled | Pulsar greedy | Splash greedy |
|---|---:|---:|---:|---:|
| chat | 81 | 54 | 77 | 59 |
| math | 171 | 122 | 166 | 122 |
| code | 154 | 114 | 169 | 110 |
| code file | 137 | 93 | 139 | 103 |
| agent | 89 | 70 | 99 | 78 |
| multilingual | 94 | 67 | 100 | 70 |

- Builds: the Pulsar 1.0.0 pre-release build of 2026-09-29; Splash 1.0.2 as installed. Both servers
  ran in one session, and all 48 requests completed cleanly.
- In all 12 sampled requests, Splash wrote the same number of tokens as in the first run. So it did the same work
  both times (86.6 vs 86.9 tok/s).
- On two prompts, the greedy text differs from an older Pulsar build. There the text is byte-identical to
  Pulsar's own reference path (1,533/1,533 positions), the path the model check below compares against.

</details>

### Qwen3.6-35B-A3B: sampled and greedy answers vs Splash, 6 prompts

The same test on Splash's other model, Qwen3.6-35B-A3B, against Splash 1.0.2. It is a mixture-of-experts (MoE)
model. It has many small expert blocks and uses only a few of them for each token.

| Engine | Sampled tok/s | Greedy tok/s | |
|---|---:|---:|---|
| **Pulsar** | **271.5** | **271.2** | |
| Splash 1.0.2 | 218.9 | 249.8 | Pulsar is **1.24×** (sampled) and **1.09×** (greedy) **faster** |

Conditions: Chrome was open. The Mac was cool for 30 requests and warm ("fair") for 18.

<details>
<summary>Details</summary>

- Model: `incoai/Qwen3.6-35B-A3B-Splash` (4-bit mixture of experts) with its DFlash 2 draft model.
- Pulsar: the 1.0.0 pre-release build of 2026-09-29, with the release settings.
- In this build, the draft's one-launch K/V math (part 3 of [What's inside](WHATS-INSIDE.md)) covered the 27B's layer
  shapes. On this model it used the regular path, which is also exact. K/V are the keys and values the model stores
  for each earlier token.

95% confidence intervals:

| Answers | Pulsar | Splash | Ratio |
|---|---|---|---|
| Sampled | 271.5 [210.7, 334.9] | 218.9 [173.8, 266.0] | 1.240 [1.201, 1.286] |
| Greedy | 271.2 [214.9, 328.7] | 249.8 [197.8, 301.5] | 1.086 [1.054, 1.118] |

Per prompt, in tok/s:

| Prompt | Pulsar sampled | Splash sampled | Pulsar greedy | Splash greedy |
|---|---:|---:|---:|---:|
| chat | 197 | 159 | 208 | 180 |
| math | 367 | 275 | 348 | 317 |
| code | 379 | 306 | 372 | 334 |
| code file | 283 | 236 | 281 | 268 |
| agent | 180 | 151 | 172 | 159 |
| multilingual | 222 | 186 | 247 | 240 |

</details>

## M5 Pro (16-core GPU, 24 GB)

### 24 GB M5 Pro with a 69K context

We let the GPU use 20 GB with `sudo sysctl iogpu.wired_limit_mb=20480` (it resets on reboot). We ran Pulsar
text-only (`SPLASH_TEXT_ONLY=1`), which skips the part of the model that reads images. We sent one greedy request per
prompt.

| Prompt | Pulsar tok/s |
|---|---:|
| math | **78.9** |
| code | **73.3** |
| chat | **39.2** |
| code edit (3.5K-token prompt) | **121.3** |
| agent task (32K-token prompt) | **43.0** |

Context: **69,625 tokens**.

Conditions: on battery, with Chrome and Safari quit. The Mac was warm ("fair") from the third request on.

<details>
<summary>Details</summary>

Memory plan, default settings vs the GPU given 20 GB:

| | Default | 20 GB |
|---|---:|---:|
| Working set: memory macOS lets the GPU use | 18,186 MiB | 20,480 MiB |
| Dynamic budget: memory for the context | 673 MB | 2,668 MB |
| Small draft head (the draft's short word list) | doesn't fit | fits |
| Context | 8,185 tokens | 69,625 tokens |

- Time to first token: math 0.8 s, code 1.1 s, chat 0.5 s, code edit 9.3 s, agent task 103.7 s.
- The code edit and the agent task moved 121 MB and 71 MB of other memory to swap, the disk space macOS uses as
  spare memory. After the agent task, macOS showed memory pressure at "warning".
- Tool calls, end to end including the prompt: code edit 84.3 tok/s, tool-copy 106.7, tool-edit 66.5.
- Exact on the real chip: 5,844/5,844 positions have the same numbers with and without the multi-row checks. A
  multi-row check tests many tokens in one pass.
- Flags: the release defaults plus `SPLASH_TEXT_ONLY=1` and a row logger.

</details>

### 24 GB M5 Pro at default settings

The same Mac with no memory setting changed. Pulsar runs text-only with an 8,185-token context.

| Prompt | Pulsar tok/s |
|---|---:|
| math | **77.6** |
| code | **72.0** |
| chat | **38.5** |

Conditions: on power, with Chrome, Safari and ChatGPT quit. The Mac stayed cool the whole time.

<details>
<summary>Details</summary>

- Tool calls, end to end including the prompt: tool-copy 115.3 tok/s, tool-edit 75.5.
- Exact on the real chip: 2,058/2,058 positions match with and without the multi-row checks. The M5 Pro also matched
  our M5 Max running as if it had 16 GPU cores, position for position.
- Build: a Pulsar 1.0.0 pre-release build with the same memory plan. Its single-request paths match the release on 16 cores.
- The small draft head didn't fit this memory plan, so Pulsar used the full one.
- A separate run with the Mac in normal use (Chrome open, CPU load up to 4.6): 56.6 tok/s sampled, 50.2 greedy.

</details>

## How long a prompt can be

| Setup | Longest context |
|---|---:|
| The model itself | 262,144 tokens |
| Our M5 Max server | 40,960 tokens |
| 24 GB Mac, GPU given 20 GB | 69,625 tokens |
| 24 GB Mac, default settings | 8,185 tokens |

On the M5 Max, we checked outputs for exactness up to 128K tokens.

## Is the output exact?

The small draft model only guesses. The full model checks every token before it keeps it.

- Greedy answers are exactly what the full model writes on its own; sampled answers follow the full model's probabilities exactly.
- Some of our early GPU code adds numbers in a different order than Splash. So rounding, and sometimes a word, can
  differ from Splash. That code passed long-document retrieval tests from 2K to 128K tokens, and quality checks.
- Every speed change after that gives byte-for-byte the same numbers as the code it replaced. This holds even when
  many guesses are checked at once.
- Neural Engine prefill is on by default since 1.1.0, as in Splash 1.3.0. At startup the engine times the GPU
  against a GPU and Neural Engine split, and uses the split only where it is faster. The split reads long Qwen3.8-27B
  prompts with part of each layer's FFN in 8-bit on the Neural Engine. Writing the answer stays on the GPU.
  `SPLASH_ANE=0` runs prefill on the GPU alone.

<details>
<summary>Check it yourself</summary>

The model check runs the real model through prompt reading, answer writing, batching and cache restore. It compares each
result with a reference path, and it takes 30 s on the GPU:

```bash
make -j12 BUILD=build build/engine-tests/model-runtime-oracle
build/engine-tests/model-runtime-oracle build/pulsar.metallib \
  "$HOME/Library/Application Support/Splash/models/incoai/Qwen3.8-27B-Splash"
# last line: model_runtime_oracle_test: PASS
```

</details>
