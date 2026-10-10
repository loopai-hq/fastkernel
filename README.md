<!-- Modified by Pulsar. (Splash's original README is docs/SPLASH-README.md.) -->
<h1 align="center">Pulsar</h1>

<h3 align="center">Qwen3.8-27B on your Mac: 153 tok/s writing code, 344 tok/s editing it.</h3>

<p align="center"><b>The fastest engine we have measured for Qwen3.8-27B on Apple silicon (M5 Max)</b>, a fork of Inco AI's
<a href="https://github.com/incoai/splash">Splash</a> 1.3.0.<br>
Up to 477 tok/s editing code · 1.44× faster than lithos-metal 0.1.2 (official release) · 1.13× Splash 1.3.0<br>
every token still checked by the full model</p>

<p align="center">
<a href="https://github.com/loopai-hq/pulsar/releases/latest"><img src="https://img.shields.io/github/v/release/loopai-hq/pulsar?label=release" alt="Latest release"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="Apache 2.0 license"></a>
<img src="https://img.shields.io/badge/Apple%20silicon-M3%20or%20newer-black" alt="Apple silicon, M3 or newer">
<img src="https://img.shields.io/badge/API-OpenAI%20%2B%20Anthropic-555" alt="OpenAI and Anthropic APIs">
</p>

<div align="center">

| **153 tok/s** | **344 tok/s** | **1.44×** | **0.1 s** |
|:---:|:---:|:---:|:---:|
| writing code, average | editing a pasted file, average | faster than lithos-metal 0.1.2 (official release) | to start a repeated prompt |

</div>

<p align="center"><a href="docs/media/lithos-tip.mp4"><img src="docs/media/lithos-tip.webp" width="720" alt="Pulsar 1.1.5 and lithos-metal 0.1.2 answer lithos-metal's launch-video prompt, a tip calculator, side by side on the same M5 Max. Pulsar writes 3,000 tokens in 19.7 s, lithos-metal in 25.3 s."></a><br>
<sub><b>lithos-metal's own launch-video prompt, same Mac:</b> Pulsar 1.1.5 writes 3,000 tokens in 19.7 s
(153 tok/s), lithos-metal 0.1.2 in 25.3 s (120 tok/s).
<a href="docs/media/lithos-tip.mp4">Video</a></sub></p>

<p align="center"><a href="docs/media/lithos-edit.mp4"><img src="docs/media/lithos-edit.webp" width="720" alt="Pulsar 1.1.5 and lithos-metal 0.1.2 edit the same pasted 210-line Python file. Pulsar finishes in 7.3 s, lithos-metal in 16.1 s."></a><br>
<sub><b>Editing a pasted 210-line file:</b> Pulsar 1.1.5 finishes its 1,894-token answer in 7.3 s (344 tok/s, up to
477), lithos-metal 0.1.2 its 1,884 tokens in 16.1 s (151 tok/s). Pulsar takes most of this answer straight from the
pasted file (prompt lookup).
<a href="docs/media/lithos-edit.mp4">Video</a></sub></p>

<p align="center"><sub>Qwen3.8-27B on an M5 Max (128 GB, macOS 27.2), greedy, thinking off, one request at a time (2026-10-10).
lithos-metal 0.1.2 is the official release, running NVIDIA's checkpoint (NVFP4 MLP, FP8 attention) with its DSpark
draft head. Real-time replays of measured token timings.</sub></p>

<p align="center"><a href="docs/media/speed-race.mp4"><img src="docs/media/speed-race.webp" width="720" alt="Pulsar 1.1.0, Splash 1.3.0 and MLX-LM write the same 708-token answer side by side on an M5 Max. Pulsar finishes in 3.97 s, Splash in 4.07 s, MLX-LM in 23.1 s."></a><br>
<sub><b>Same prompt, same 708-token answer:</b> Pulsar 3.97 s · Splash 1.3.0 4.07 s · MLX-LM 23.1 s (MLX-LM without
a draft model, its default).
<a href="docs/media/speed-race.mp4">Video</a></sub></p>

<p align="center"><a href="docs/media/galaxy-app.mp4"><img src="docs/media/galaxy-app.webp" width="720" alt="Pulsar 1.1.0 and Splash 1.3.0 write a particle-galaxy app side by side on an M5 Max. Pulsar writes its 3,124 tokens in 20.9 s; Splash takes 22.1 s for the same number of tokens. Then the galaxy Pulsar wrote runs."></a><br>
<sub><b>Building a particle galaxy app in one shot:</b> Pulsar writes its 3,124 tokens in 20.9 s, Splash 1.3.0
takes 22.1 s for the same number of tokens, then the galaxy Pulsar wrote runs.
<a href="docs/media/galaxy-app.mp4">Video</a></sub></p>

<p align="center"><a href="docs/media/code-edit.mp4"><img src="docs/media/code-edit.webp" width="720" alt="Pulsar 1.1.0, Splash 1.3.0 and MLX-LM edit the same pasted 210-line Python file. Pulsar finishes in 8.2 s, Splash in 11.7 s, MLX-LM in 64.5 s."></a><br>
<sub><b>Editing a pasted 210-line file:</b> Pulsar 8.2 s · Splash 1.3.0 11.7 s · MLX-LM 64.5 s (MLX-LM without a
draft model, its default).
<a href="docs/media/code-edit.mp4">Video</a></sub></p>

<p align="center"><sub>Qwen3.8-27B 4-bit on an M5 Max, one engine's server per pane, greedy answers (2026-10-07). Real-time
replays of measured token timings.</sub></p>

> **Jump to:** [How fast?](#m5-max-128-gb) · [What's new in 1.1.6](#whats-new-in-116) · [Requirements](#requirements) ·
> [Quick start](#quick-start) · [What's different](#whats-different-from-splash) · [Models](#models-on-hugging-face) ·
> [Switches](docs/SWITCHES.md) · [All the numbers](docs/BENCHMARKS.md)

Pulsar (formerly fastkernel) is a fork of Inco AI's [Splash](https://github.com/incoai/splash) 1.3.0. It uses Splash's
model packages and DFlash 2 draft models unchanged and adds new engine code: GPU kernels, drafting and prompt lookup.
It runs the Qwen3.8-27B and Qwen3.6-35B-A3B AI models on your own Mac. Chat with them in your browser, or connect a
coding agent or any other app.

A model writes its answer one token at a time; a token is a word or part of a word. Pulsar pairs the full model with a
small guesser, a draft model that proposes the next few tokens. The full model checks every guess in one step and keeps
only the ones it agrees with, so you get the full model's answers, faster.

On an M5 Max, Pulsar 1.1.5 averages **108.0 tok/s** across 13 prompts, from short chat to a 32K-token agent task
(greedy, 2026-10-09), and a repeated prompt starts in 0.1 s from its prompt cache.

## M5 Max (128 GB)

**Speed you see** (Qwen3.8-27B 4-bit, Pulsar 1.1.5, greedy, measured from the stream, 2026-10-10):

| Task | Peak | Average | First token |
|---|---:|---:|---:|
| Edit a pasted 210-line file | **477 tok/s** | **344 tok/s** | 1.82 s |
| Write a tip-calculator app (3,000 tokens) | 208 tok/s | 153 tok/s | 0.12 s |

Peak is the most tokens in any one second; average runs from the first token to the last. The edit row is the first
run (nothing cached), so its first token includes reading the 1,825-token pasted file; the app row is the median of
three runs.

**Pulsar 1.1.5 vs lithos-metal 0.1.2, Qwen3.8-27B** (2026-10-10):

On our 128 GB M5 Max running macOS 27.2, Pulsar 1.1.5 delivered **1.44×** the client-observed decode throughput of
lithos-metal 0.1.2, its official release (geometric mean over 10 prompts; one ABBA session, greedy decoding, thinking
off, output capped at 1,024 tokens). Decode tok/s, mean of two runs per prompt:

| Prompt | lithos-metal 0.1.2 | Pulsar 1.1.5 | Pulsar is |
|---|---:|---:|---:|
| lithos-metal's launch-video prompt (tip calculator) | 104.2 | **131.8** | 1.26× |
| Code | 137.9 | **182.9** | 1.33× |
| Math | 104.8 | **168.6** | 1.61× |
| Code file | 79.4 | **106.2** | 1.34× |
| Chat (5 prompts) | 54.0–57.0 | **78.3–91.0** | 1.37–1.65× |
| Multilingual | 50.2 | **63.7** | 1.27× |
| **All 10 prompts** | 75.7 | **108.3** | **1.44×** |

The first token arrives in 0.22 s on Pulsar and 0.39 s on lithos-metal (median). lithos-metal runs NVIDIA's checkpoint,
which keeps attention in FP8: it has a 17.6 GB logical target-weight payload per verification pass, against 14.4 GB
for Pulsar's package, and part of Pulsar's lead comes from that smaller payload. The drafters differ (DSpark vs
DFlash 2), and answer quality was not compared.

All sessions: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#pulsar-vs-lithos-metal-greedy-10-prompts).

Pulsar vs other engines:

| Engine | Guesses ahead with | Its speed | Pulsar's speed | Pulsar is |
|---|---|---:|---:|---|
| lithos-metal 0.1.2 | DSpark draft head | 75.7 | 108.3 | **1.44× faster** |
| Splash 1.3.0 | DFlash 2 draft model | 86.5 | 98.6 | **1.13× faster** |
| MTPLX | MTP | 64.9 | 125.0 | **1.93× faster** |
| AX Engine | MTP | 40.1 | 131.0 | **3.27× faster** |
| MLX-LM | no draft model (its default) | 30.0 | 115.0 | **3.84× faster** |
| llama.cpp | no draft model (its default) | 27.1 | 121.7 | **4.49× faster** |

Speeds are in tok/s (tokens per second). All rows are Pulsar 1.1.5 (2026-10-10). MTP is the model's own built-in guesser. Each pair ran in one session, taking
turns on the same prompts: 12 prompts against Splash 1.3.0 (chat, math, code, a code file, a 32K-token agent task, a
multilingual prompt and two long-context prompts), the 6 standard prompts against the others (5 greedy prompts against
AX Engine). In chat, Pulsar 1.1.5 writes
74.9 tok/s, 1.13× Splash 1.3.0's 66.4.

Sent again, a 2,247-token prompt starts in 0.1 s: Pulsar 1.1.0 reuses its prompt cache.

**Also: Qwen3.6-35B-A3B (MoE)**

MoE means "mixture of experts". The model has many small expert blocks and uses only a few of them for each token.
That makes it fast for its size. On Qwen3.6-35B-A3B, Pulsar 1.1.5 writes **264.3 tok/s** on average across 13 prompts,
from short chat to a 32K-token agent task (greedy, 2026-10-10).

Full numbers and how we measured: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

## What's new in 1.1.6

Long prompts write faster, and prompts are read faster. Measured against 1.1.5 on the same M5 Max, taking turns on
the same prompts (greedy, 2026-10-10):

| | Pulsar 1.1.5 | Pulsar 1.1.6 | 1.1.6 is |
|---|---:|---:|---|
| Qwen3.8-27B, 32K-token agent task | 77.9 tok/s | 83.3 tok/s | **6.9% faster** |
| Qwen3.8-27B, a code file | 101.7 tok/s | 104.3 tok/s | 2.6% faster |
| Qwen3.8-27B, 13 prompts, average | 101.6 tok/s | 102.4 tok/s | 0.8% faster |
| Qwen3.6-35B-A3B, 32K-token agent task | 135.1 tok/s | 139.9 tok/s | **3.6% faster** |
| Reading a 32K-token prompt | 40.6 s | 40.1 s | 1.2% faster |
| Reading an 8K-token prompt | 9.12 s | 8.92 s | 2.2% faster |

Answers match 1.1.5 on all 52 Qwen3.8-27B requests in this run. The Neural Engine split calibrates more reliably at
startup; the first start after installing spends about 6 s on it.

**Download and source.** Pulsar 1.1.6 ships as a ready-to-run download, the latest Pulsar. This repository holds the
Apache-2.0 source of Pulsar 1.1.4, which builds and runs as before.

## Requirements

| | |
|---|---|
| Mac | Apple silicon, M3 or newer (measured on M5 Max and M5 Pro) |
| macOS | 26.4 or later |
| Memory | 24 GB or more for Qwen3.8-27B (24 GB Macs need one extra setting, see Quick start) |
| Disk | 17.4 GB for Qwen3.8-27B, 20.9 GB for Qwen3.6-35B-A3B |
| Python | 3.12 to 3.14 |
| Xcode | Not needed for the download. Building the 1.1.4 source needs the full Xcode app. |

## Quick start

Both ways below start a server at <http://127.0.0.1:8000>. It speaks the OpenAI and Anthropic APIs, so apps built for
either one can use it. To chat, open that address in your browser. The first run downloads the model (17.4 GB).

**24 GB Mac?** First, run `sudo sysctl iogpu.wired_limit_mb=20480`. It lets the GPU use 20 GB until you restart the
Mac. Then put `SPLASH_TEXT_ONLY=1` in front of the serve command. It skips the part of the model that reads images.

For Qwen3.6-35B-A3B, run `./pulsar serve --model incoai/Qwen3.6-35B-A3B-Splash`, without `SPLASH_DRAFT_HEAD_IDS`.

### Download Pulsar 1.1.6 (latest, no Xcode)

Download `pulsar-1.1.6-macos-arm64.tar.gz` from
[Releases](https://github.com/loopai-hq/pulsar/releases/latest). Then unpack it and start the server:

```bash
tar -xzf pulsar-1.1.6-macos-arm64.tar.gz && cd pulsar
SPLASH_DRAFT_HEAD_IDS=$PWD/data/head-ranked.u32 ./pulsar serve --model incoai/Qwen3.8-27B-Splash
```

Got the file through a browser or AirDrop? Then macOS blocks it until you run
`/usr/bin/xattr -dr com.apple.quarantine .` once in the `pulsar` folder. Run it before the serve command.

The first run also sets up its Python packages, which takes 20 seconds.

### Build Pulsar 1.1.4 from source

```bash
git clone https://github.com/loopai-hq/pulsar && cd pulsar
make install MODEL=incoai/Qwen3.8-27B-Splash
SPLASH_DRAFT_HEAD_IDS=$PWD/data/head-ranked.u32 ./pulsar serve --model incoai/Qwen3.8-27B-Splash
```

### Use it with a coding agent

A coding agent is an AI tool that reads and edits your code, like OpenCode, Claude Code or Codex. Leave the server
running. Open a second terminal in the pulsar folder and run:

```bash
./pulsar opencode    # or: ./pulsar claude / ./pulsar codex / ./pulsar hermes
```

The agent opens already connected to Pulsar. Other apps can use the OpenAI API at `http://127.0.0.1:8000/v1` or
the Anthropic API at `http://127.0.0.1:8000`, with the model `incoai/Qwen3.8-27B-Splash`.

### Skill for coding agents

A skill is an instruction file that a coding agent reads.
[`.claude/skills/pulsar`](.claude/skills/pulsar/SKILL.md) teaches an agent to install, start, connect and stop
Pulsar. Claude Code and OpenCode find it when you open them in this folder. To use it from any folder:

```bash
cp -R .claude/skills/pulsar ~/.claude/skills/    # Claude Code and OpenCode
cp -R .claude/skills/pulsar ~/.agents/skills/    # Codex
```

## What's different from Splash

- **Built on Splash 1.3.0.** Pulsar brings its speedups onto the latest Splash, with its prompt reading and prompt
  cache: a repeated prompt starts in 0.1 s.
- **Same model, same checking.** The guesser proposes the next few tokens. The full model checks every guess and keeps
  only the ones it agrees with.
- **Guesses from your prompt.** Some answers repeat your input, like an edit to a file you pasted. There the engine
  takes its guesses straight from your prompt. The full model can then keep many tokens in one step.
- **A lighter guesser.** The guesser picks from a shorter list of the tokens the model writes most
  (`data/head-ranked.u32`), so each guess costs less.
- **Faster GPU code.** New GPU kernels, tuned by measurement on an M5 Max, do the model's math in less time.
- **Text-only mode.** `SPLASH_TEXT_ONLY=1` skips the image weights, which leaves more memory for the context on smaller
  Macs.
- **New engine code.** The 1.1.4 source adds 6,191 lines of C++ and Metal, Apple's GPU language, to Splash 1.3.0,
  including 38 new GPU kernels, small programs that run on the GPU. What each part does and what it gained:
  [docs/WHATS-INSIDE.md](docs/WHATS-INSIDE.md). Every change has a switch: [docs/SWITCHES.md](docs/SWITCHES.md).

## Models on Hugging Face

Each model has a page with its speed table and quick start:

- [loopai-hq/Qwen3.8-27B-Pulsar](https://huggingface.co/loopai-hq/Qwen3.8-27B-Pulsar)
- [loopai-hq/Qwen3.6-35B-A3B-Pulsar](https://huggingface.co/loopai-hq/Qwen3.6-35B-A3B-Pulsar)

They carry Inco AI's Splash model packages unchanged, with full credit, plus Pulsar's guesser token list for
Qwen3.8-27B.

## Credits & license

Pulsar is licensed under the Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). The download includes
LICENSE, NOTICE and THIRD_PARTY_NOTICES. If you build on Pulsar, keep the NOTICE file and credit Pulsar. Pulsar was
called fastkernel until version 1.1.3.

Thank you to [Inco AI](https://github.com/incoai) for [Splash](https://github.com/incoai/splash), the Apache-2.0 engine
Pulsar is built on, and for the DFlash 2 draft models and Splash model packages Pulsar runs. Pulsar 1.1.6 is built on
Splash 1.3.0. Splash's own README: [docs/SPLASH-README.md](docs/SPLASH-README.md). Thank you also to the Qwen team for
the Qwen models and to mlx-community for the 4-bit conversions. Third-party code that Splash ships keeps its own
license: [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Files we changed from Splash say "Modified by Pulsar."; the
full list, including data files that can't hold a comment, is in [docs/CHANGED-FILES.md](docs/CHANGED-FILES.md).
