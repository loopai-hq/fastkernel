<!-- Modified by meowkernels. (Splash's original README is docs/SPLASH-README.md.) -->
<h1 align="center">fastkernel</h1>

<p align="center"><b>The fastest inference engine for Qwen3.8-27B and Qwen3.6-35B-A3B on Apple Silicon.</b><br>
Qwen3.8-27B: up to 452 tok/s editing code · 1.36× lithos-metal (greedy, 10 prompts) · 1.11× faster than Splash 1.3.0<br>
measured on an M5 Max · every token still checked by the full model</p>

<p align="center"><a href="docs/media/speed-race.mp4"><img src="docs/media/speed-race.webp" width="720" alt="fastkernel 1.1.0, Splash 1.3.0 and MLX-LM write the same 708-token answer side by side on an M5 Max. fastkernel finishes in 3.97 s, Splash in 4.07 s, MLX-LM in 23.1 s."></a><br>
<sub><b>Same prompt, same 708-token answer:</b> fastkernel 3.97 s · Splash 1.3.0 4.07 s · MLX-LM 23.1 s.
<a href="docs/media/speed-race.mp4">Video</a></sub></p>

<p align="center"><a href="docs/media/galaxy-app.mp4"><img src="docs/media/galaxy-app.webp" width="720" alt="fastkernel 1.1.0 and Splash 1.3.0 write a particle-galaxy app side by side on an M5 Max. fastkernel writes its 3,124 tokens in 20.9 s; Splash takes 22.1 s for the same number of tokens. Then the galaxy fastkernel wrote runs."></a><br>
<sub><b>Building a particle galaxy app in one shot:</b> fastkernel writes its 3,124 tokens in 20.9 s, Splash 1.3.0
takes 22.1 s for the same number of tokens, then the galaxy fastkernel wrote runs.
<a href="docs/media/galaxy-app.mp4">Video</a></sub></p>

<p align="center"><a href="docs/media/code-edit.mp4"><img src="docs/media/code-edit.webp" width="720" alt="fastkernel 1.1.0, Splash 1.3.0 and MLX-LM edit the same pasted 210-line Python file. fastkernel finishes in 8.2 s, Splash in 11.7 s, MLX-LM in 64.5 s."></a><br>
<sub><b>Editing a pasted 210-line file:</b> fastkernel 8.2 s · Splash 1.3.0 11.7 s · MLX-LM 64.5 s.
<a href="docs/media/code-edit.mp4">Video</a></sub></p>

<p align="center"><sub>Qwen3.8-27B 4-bit on an M5 Max, one engine's server per pane, greedy answers (2026-10-07). Real-time
replays of measured token timings.</sub></p>

> **Jump to:** [How fast?](#m5-max-128-gb) · [Requirements](#requirements) · [Quick start](#quick-start) ·
> [What's different](#whats-different-from-splash) · [Switches](docs/SWITCHES.md) · [All the numbers](docs/BENCHMARKS.md)

fastkernel runs the Qwen3.8-27B and Qwen3.6-35B-A3B AI models on your own Mac. Chat with them in your browser, or
connect a coding agent or any other app.

fastkernel 1.1.3 is built on Splash 1.3.0, the latest Splash. On an M5 Max it writes answers 1.11×
faster than Splash 1.3.0, and a repeated prompt starts in 0.1 s from its prompt cache. It brings everything new in
Splash 1.3.0. A small draft model guesses ahead, and the full model checks every token before it
keeps it.

## M5 Max (128 GB)

**Speed you see** (Qwen3.8-27B 4-bit, fastkernel 1.1.0, greedy, measured from the stream, 2026-10-07):

| Task | Peak | Average | First token |
|---|---:|---:|---:|
| Edit a pasted 210-line file | **452 tok/s** | **321 tok/s** | 2.3 s |
| Write a Python todo app | 205 tok/s | 184 tok/s | 0.14 s |
| Write a particle-galaxy app | 198 tok/s | 149 tok/s | 0.21 s |

Peak is the most tokens in any one second; average runs from the first token to the last. Each row is one run, the
first (nothing cached); the first token includes reading the prompt, here a 1,825-token pasted file.

**fastkernel 1.1.3 vs lithos-metal 0.1.2, Qwen3.8-27B** (2026-10-09):

On our 128 GB M5 Max running macOS 27.2, fastkernel 1.1.3 delivered 1.36× the client-observed decode throughput of
lithos-metal 0.1.2 across these 10 prompts (geometric mean; one ABBA session, greedy decoding, thinking off, output
capped at 1,024 tokens). The engines used different quantized checkpoints and drafters; task-quality parity was not
evaluated.

| Prompt | lithos-metal 0.1.2 | fastkernel 1.1.3 |
|---|---:|---:|
| lithos-metal's launch-video prompt (tip calculator) | 105.2 | **136.0** |
| Code | 144.3 | **180.8** |
| Math | 116.3 | **166.0** |
| Code file | 84.8 | **101.8** |
| Chat (5 prompts) | 55.7–67.0 | **80.4–92.4** |
| Multilingual | 55.1 | **62.2** |

Decode tok/s, mean of two runs per prompt. Same Mac, one server at a time in the order fastkernel, lithos-metal,
lithos-metal, fastkernel, each started with its README command plus port and context size, after at least 120 s of
warm-up. lithos-metal runs NVIDIA's NVFP4 checkpoint with its DSpark draft head; fastkernel runs the 4-bit Splash
package with DFlash2. Settings and limits: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#fastkernel-113-vs-lithos-metal-012-greedy-10-prompts).

fastkernel 1.1.0 vs other engines (2026-10-07):

| Engine | Its speed | fastkernel 1.1.0's speed | fastkernel 1.1.0 is |
|---|---:|---:|---|
| Splash 1.3.0 | 89.0 | 99.2 | **1.11× faster** |
| MTPLX | 65.7 | 118.2 | **1.80× faster** |
| AX Engine | 38.3 | 119.6 | **3.12× faster** |
| MLX-LM | 30.9 | 114.9 | **3.72× faster** |
| llama.cpp | 26.7 | 110.7 | **4.14× faster** |

Speeds are in tok/s (tokens per second). A token is a word or part of a word. Each pair ran in one session, taking
turns on the same prompts: 12 prompts against Splash 1.3.0 (chat, math, code, a code file, a 32K-token agent task, a
multilingual prompt and two long-context prompts), the 6 standard prompts against the others (5 greedy prompts against
AX Engine). In chat, fastkernel 1.1.0 writes 76.7 tok/s, 1.12× Splash 1.3.0's 68.5.

Sent again, a 2,247-token prompt starts in 0.1 s: fastkernel 1.1.0 reuses its prompt cache.

**Also: Qwen3.6-35B-A3B (MoE)**

MoE means "mixture of experts". The model has many small expert blocks and uses only a few of them for each token.
That makes it fast for its size. On Qwen3.6-35B-A3B, fastkernel 1.1.0 writes 268.1 tok/s sampled and 299.6 greedy,
against Splash 1.3.0's 260.9 and 276.1 on the same 6 prompts (2026-10-07).

Full numbers and how we measured: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

## Requirements

| | |
|---|---|
| Mac | Apple silicon, M3 or newer |
| macOS | 26.4 or later |
| Memory | 24 GB or more for Qwen3.8-27B (24 GB Macs need one extra setting, see Quick start) |
| Disk | 17.4 GB for Qwen3.8-27B, 20.9 GB for Qwen3.6-35B-A3B |
| Python | 3.12 to 3.14 |
| Xcode | Not needed for the prebuilt download. Building from source needs the full Xcode app. |

## Quick start

Both ways below start a server at <http://127.0.0.1:8000>. It speaks the OpenAI and Anthropic APIs, so apps built for
either one can use it. To chat, open that address in your browser. The first run downloads the model (17.4 GB).

**24 GB Mac?** First, run `sudo sysctl iogpu.wired_limit_mb=20480`. It lets the GPU use 20 GB until you restart the
Mac. Then put `SPLASH_TEXT_ONLY=1` in front of the serve command. It skips the part of the model that reads images.

For Qwen3.6-35B-A3B, use `incoai/Qwen3.6-35B-A3B-Splash` instead of `incoai/Qwen3.8-27B-Splash`.

### Prebuilt download (no Xcode)

Download `fastkernel-1.1.3-macos-arm64.tar.gz` from
[Releases](https://github.com/loopai-hq/fastkernel/releases). Then unpack it and start the server:

```bash
tar -xzf fastkernel-1.1.3-macos-arm64.tar.gz && cd fastkernel
SPLASH_DRAFT_HEAD_IDS=$PWD/data/head-ranked.u32 ./splash serve --model incoai/Qwen3.8-27B-Splash
```

Got the file through a browser or AirDrop? Then macOS blocks it until you run
`/usr/bin/xattr -dr com.apple.quarantine .` once in the `fastkernel` folder. Run it before the serve command.

The first run also sets up its Python packages, which takes 20 seconds.

### Build from source

```bash
git clone https://github.com/loopai-hq/fastkernel && cd fastkernel
make install MODEL=incoai/Qwen3.8-27B-Splash
SPLASH_DRAFT_HEAD_IDS=$PWD/data/head-ranked.u32 ./splash serve --model incoai/Qwen3.8-27B-Splash
```

### Use it with a coding agent

A coding agent is an AI tool that reads and edits your code, like OpenCode, Claude Code or Codex. Leave the server
running. Open a second terminal in the fastkernel folder and run:

```bash
./splash opencode    # or: ./splash claude / ./splash codex / ./splash hermes
```

The agent opens already connected to fastkernel. Other apps can use the OpenAI API at `http://127.0.0.1:8000/v1` or
the Anthropic API at `http://127.0.0.1:8000`, with the model `incoai/Qwen3.8-27B-Splash`.

### Skill for coding agents

A skill is an instruction file that a coding agent reads.
[`.claude/skills/fastkernel`](.claude/skills/fastkernel/SKILL.md) teaches an agent to install, start, connect and stop
fastkernel. Claude Code and OpenCode find it when you open them in this folder. To use it from any folder:

```bash
cp -R .claude/skills/fastkernel ~/.claude/skills/    # Claude Code and OpenCode
cp -R .claude/skills/fastkernel ~/.agents/skills/    # Codex
```

## What's different from Splash

- **Built on Splash 1.3.0.** fastkernel 1.1.3 brings fastkernel's speedups onto the latest Splash, with its prompt
  reading and prompt cache: a repeated prompt starts in 0.1 s.
- **Same model, same checking.** A small draft model guesses the next few tokens. The full model checks every guess
  and keeps only the ones it agrees with.
- **Guesses from your prompt.** Some answers repeat your input, like an edit to a file you pasted. There the engine
  takes its guesses straight from your prompt. The full model can then keep many tokens in one step.
- **Faster GPU code.** New GPU code, hand-tuned for Apple silicon, does the model's math in less time.
- **Text-only mode.** `SPLASH_TEXT_ONLY=1` skips the image weights, which leaves more memory for the context on smaller
  Macs.
- **New engine code.** 6,187 new lines of C++ and Metal, Apple's GPU language, on top of Splash 1.3.0.
  They include 38 new GPU kernels, small programs that run on the GPU. What each part does and
  what it gained: [docs/WHATS-INSIDE.md](docs/WHATS-INSIDE.md). Every change has a switch:
  [docs/SWITCHES.md](docs/SWITCHES.md).

## Credits & license

fastkernel is licensed under the Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). If you build on it, keep
the NOTICE file and credit fastkernel.

Thank you to [Inco AI](https://github.com/incoai) for [Splash](https://github.com/incoai/splash), the Apache-2.0 engine
fastkernel is built on. fastkernel 1.1.3 is built on Splash 1.3.0. Splash's own README:
[docs/SPLASH-README.md](docs/SPLASH-README.md). Third-party code that Splash ships keeps its own license:
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Each file we changed from Splash says "Modified by meowkernels."
