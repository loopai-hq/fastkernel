<!-- Modified by meowkernels. (Splash's original README is docs/SPLASH-README.md.) -->
<h1 align="center">fastkernel</h1>

<p align="center"><b>The fastest inference engine for Qwen3.8-27B and Qwen3.6-35B-A3B on Apple Silicon.</b><br>
1.11× faster than Splash 1.3.0 on an M5 Max · every token still checked by the full model</p>

> **Jump to:** [How fast?](#m5-max-128-gb) · [Requirements](#requirements) · [Quick start](#quick-start) ·
> [What's different](#whats-different-from-splash) · [Switches](docs/SWITCHES.md) · [All the numbers](docs/BENCHMARKS.md)

fastkernel runs the Qwen3.8-27B and Qwen3.6-35B-A3B AI models on your own Mac. Chat with them in your browser, or
connect a coding agent or any other app.

fastkernel 1.1.0 is built on Splash 1.3.0, the latest Splash. On an M5 Max it writes answers 1.11×
faster than Splash 1.3.0 and matches fastkernel 1.0.0's decode speed. Compared with fastkernel 1.0.0, it reuses its prompt cache better.  It brings
everything new in Splash 1.3.0. A small draft model guesses ahead, and the full model checks every token before it
keeps it.

## M5 Max (128 GB)

fastkernel 1.1.0, side by side with Splash 1.3.0 and fastkernel 1.0.0 (2026-10-07):

| Engine | Its speed | fastkernel 1.1.0's speed | fastkernel 1.1.0 is |
|---|---:|---:|---|
| Splash 1.3.0 | 89.0 | 99.2 | **1.11× faster** |
| fastkernel 1.0.0 | 100.8 | 99.2 | **same speed** (0.98×) |

Speeds are in tok/s (tokens per second). A token is a word or part of a word. All three servers ran in one session,
taking turns on the same 12 prompts: chat, math, code, a code file, a 32K-token agent task, a multilingual prompt and
two long-context prompts. In chat, fastkernel 1.1.0 writes 76.7 tok/s, 1.12×
Splash 1.3.0's 68.5.

**Time to first token**, for a new prompt with nothing cached:

| Prompt | fastkernel 1.1.0 | fastkernel 1.0.0 |
|---|---:|---:|
| 2,247 tokens | 3.1 s | 3.2 s |
| 4,702 tokens | 6.3 s | 6.5 s |
| 6,645 tokens | 8.5 s | 8.8 s |
| 32,678 tokens | 45.8 s | 46.3 s |

Sent again, the 2,247-token prompt starts in 0.1 s on fastkernel 1.1.0, against 0.4 s on 1.0.0: 1.1.0 reuses its prompt cache better.

**fastkernel 1.0.0 vs other engines**

| Engine | Its speed | fastkernel 1.0.0's speed | fastkernel 1.0.0 is |
|---|---:|---:|---|
| Splash 1.1.0 | 93.8 | 123.2 | **1.31× faster** |
| MTPLX | 64.0 | 123.5 | **1.93× faster** |
| AX Engine | 40.5 | 126.0 | **3.11× faster** |
| MLX-LM | 31.2 | 116.8 | **3.74× faster** |
| llama.cpp | 27.0 | 116.8 | **4.33× faster** |

Speeds are in tok/s, measured 2026-09-26 to 2026-09-30. Each pair ran in one session, on the same prompts.

**Also: Qwen3.6-35B-A3B (MoE)**

MoE means "mixture of experts". The model has many small expert blocks and uses only a few of them for each token.
That makes it fast for its size.

On Qwen3.6-35B-A3B, fastkernel 1.0.0 is **1.24× faster** than Splash 1.0.2: 271.5 vs 218.9 tok/s, sampled
(2026-09-29).

## M5 Pro (24 GB)

The context is how much text the model can hold at once: your prompt plus its answer. With one memory setting (see
Quick start), a 24 GB M5 Pro running fastkernel 1.0.0 holds a 69K context and writes (greedy answers, one request per
task):

| Task | tok/s |
|---|---:|
| Code edit | 121 |
| Math question | 79 |
| Coding question | 73 |
| 32K-token coding-agent task | 43 |
| Chat | 39 |

The code edit is fastest because its answer repeats text from the prompt.

At default settings, the same Mac holds an 8K context. It writes 77.6 tok/s on a math question, 72.0 on a coding
question and 38.5 in chat.

**Base M5 (10-core GPU, 32 GB):** fastkernel 1.0.0 is 2–5% faster than Splash 1.1.0 in two runs on the same prompts
(35.3 vs 33.5 tok/s in the second).

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

Download `fastkernel-1.1.0-macos-arm64.tar.gz` from
[Releases](https://github.com/loopai-hq/fastkernel/releases). Then unpack it and start the server:

```bash
tar -xzf fastkernel-1.1.0-macos-arm64.tar.gz && cd fastkernel
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

- **Built on Splash 1.3.0.** fastkernel 1.1.0 brings fastkernel's speedups onto the latest Splash, with its prompt
  reading and prompt cache: a repeated prompt starts in 0.1 s.
- **Same model, same checking.** A small draft model guesses the next few tokens. The full model checks every guess
  and keeps only the ones it agrees with.
- **Guesses from your prompt.** Some answers repeat your input, like an edit to a file you pasted. There the engine
  takes its guesses straight from your prompt. The full model can then keep many tokens in one step.
- **Faster GPU code.** New GPU code, hand-tuned for Apple silicon, does the model's math in less time.
- **Runs on 24 GB Macs.** Text-only mode skips the image weights. On a 24 GB M5 Pro, fastkernel 1.0.0 fit the model
  and a 69K context.
- **New engine code.** 5,314 new lines of C++ and Metal, Apple's GPU language, on top of Splash 1.3.0.
  They include 34 new GPU kernels, small programs that run on the GPU. What each part does and
  what it gained: [docs/WHATS-INSIDE.md](docs/WHATS-INSIDE.md). Every change has a switch:
  [docs/SWITCHES.md](docs/SWITCHES.md).

## Credits & license

fastkernel is licensed under the Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). If you build on it, keep
the NOTICE file and credit fastkernel.

Thank you to [Inco AI](https://github.com/incoai) for [Splash](https://github.com/incoai/splash), the Apache-2.0 engine
fastkernel is built on. fastkernel 1.1.0 is built on Splash 1.3.0. Splash's own README:
[docs/SPLASH-README.md](docs/SPLASH-README.md). Third-party code that Splash ships keeps its own license:
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Each file we changed from Splash says "Modified by meowkernels."
