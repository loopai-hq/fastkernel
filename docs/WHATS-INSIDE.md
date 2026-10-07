# What we built

fastkernel is new engine code, not a settings change. Compared with Splash 1.3.0, the engine's
`runtime/` folder has **54 files changed, 6,187 lines added and 379 removed**.

- 2,700 of the new lines are Metal shaders and GPU headers. Metal is Apple's language for GPU code.
- There are **38 new GPU kernels**: 137 `kernel void` functions, up from
  Splash 1.3.0's 99. A kernel is a small program that runs on the GPU.
- The model weights, the draft model and the rule that the full model checks every token stay the same.

The biggest pieces, in lines added:

| File | Lines added |
|---|---:|
| `runtime/model/Runtime.cpp` | 1,170 |
| `runtime/metal/kernels/decode/sampling.metal` | 651 |
| `runtime/metal/kernels/common/gdn_wide.h` | 627 |
| `runtime/metal/kernels/decode/linear_q4_split.metal` | 620 |
| `runtime/metal/kernels/common/gdn_value_parts.h` | 388 |
| `runtime/metal/MetalBackend.mm` | 353 |

## How the engine writes an answer

The engine writes an answer in steps. A token is a word or part of a word.

1. A small draft model guesses the next few tokens.
2. The full model checks all the guesses at once and keeps the ones it agrees with.
3. The CPU hands this work to the GPU as GPU launches. A GPU launch is one job for the GPU to run.

Speed is in tok/s: tokens written per second. We measured every gain below on an M5 Max with fastkernel 1.0.0, one change at a time. We
also checked each change's output (see "Is the output exact?" in [BENCHMARKS.md](BENCHMARKS.md)).

## 1. New GPU kernels for the 4-bit matrix multiplications

The model's weights are big grids of numbers (matrices), stored at 4 bits per number. Each step multiplies the tokens
through them.

- Long, thin multiplications now split across more GPU cores, and the parts are added up at the end.
- Each kernel hands its partial sums straight to the next kernel. It no longer writes them out and reads them back.

| Change | Gain |
|---|---|
| Sums of each multiplication's input, handed straight on | each step 2.1% faster |
| The draft's narrow multiplications, split | each step 1.24% faster |
| Three or four requests at once | each step takes 20% less time; the total goes from 163.9 to 208.2 tok/s (27% more) |
| Scheduling fixes inside the kernels | another 0.15–0.31 ms saved per step, each |

Code: `runtime/metal/kernels/decode/linear_q4_split.metal`, `runtime/metal/kernels/shared/normalization.metal`,
`runtime/ops/Linear.cpp`.

## 2. New GPU kernels for the linear-attention (GDN) layers

The model mixes two kinds of layers. Attention layers store keys and values (K/V) for every earlier token, and look
back at them. GDN layers are linear attention: they keep a running summary instead, and update it with each token.

- The update, called the scan, now runs in four parallel parts: the server runs 1.07% faster.
- When many tokens are checked at once, a single pass replaces a chain of small launches. Code edits get 2.39%
  faster, and tool copies (tool calls that copy text) 2.74% faster.
- The GDN output kernel also prepares the next multiplication's input: 0.08 ms saved per step.

Code: `runtime/metal/kernels/decode/gdn.metal`, `runtime/metal/kernels/common/gdn_value_parts.h`, `gdn_wide.h`,
`runtime/ops/GDN.cpp`.

## 3. Smarter drafting

- **Block verification.** For sampled answers, the full model judges the guesses as one block instead of one at a
  time. It keeps more of each block: 1.29% more tokens per step.
- **Drafting ahead.** The next draft runs on the GPU while the CPU reads the current result: 0.26 ms saved per step.
- **A smaller draft vocabulary.** The draft chooses from a shorter list of the most frequent tokens
  (`data/head-ranked.u32`): 2.4% more tok/s.
- **Sharper guesses.** The draft gets its own temperature and top-p, which set how bold its picks are. That gives 0.8%
  and 0.3% more tokens accepted.

Code: `runtime/model/DFlashDraft.cpp`, `runtime/ops/Sampling.cpp`, `runtime/ops/DraftSelector.cpp`,
`runtime/metal/kernels/decode/sampling.metal`, `draft.metal`.

## 4. Prompt lookup: exact wide checking for copy-heavy text

Some answers repeat your input. Say you paste a file and ask for one change: most of the answer is your file again.
Tool calls that copy text work the same way.

For these answers, the engine guesses the next words straight from your prompt, and the full model checks them. When
the guesses match, the full model keeps many words in one step.

For example, with fastkernel 1.0.0 a code edit on a 24 GB M5 Pro ran at 121 tok/s, and a coding question at 73.

How the wide check stays exact:

- A normal check covers 8 tokens, one row each. Prompt lookup checks 16 or 32 rows in one pass.
- Every row gets exactly the bytes a normal 8-row check would give it, so the output stays exact.
- The engine picks the widest check that stays exact on your Mac's GPU.

| Change | Gain |
|---|---|
| 16 rows | rewrites 52% and edits 24% faster |
| The step up to 32 rows | code edits 12.7% and tool copies 27.4% faster again |

Code: `runtime/model/Runtime.cpp` and `PromptLookup.hpp` (lookup and scheduling), `runtime/model/QwenTarget.cpp`
(the exact-row inputs and the width check), `runtime/ops/PagedAttention.cpp` (row-stable verify attention).

## 5. Fewer, larger GPU launches

Each change below cuts time the CPU and GPU spend setting up or waiting on each other.

- The GPU starts a step before the CPU finishes building it: 0.07–0.2 ms per step.
- A tool-call step runs as a single GPU command: tool calls 1.36% faster.

Code: `runtime/metal/MetalBackend.mm`, `runtime/metal/CommandGraph.hpp`, `runtime/model/Runtime.cpp`.

## 6. A memory plan that fits 24 GB Macs

- **Text-only mode** skips the weights for reading images.
- **Safety margins scale with the Mac's memory**, so a small Mac keeps more of its memory for the model.
- **The engine plans the small draft vocabulary up front.** If it doesn't fit, the engine uses the full one.
- **The engine keeps prompt checkpoints as a cache.** A checkpoint is the model's state, saved while it reads a
  prompt. A second agent sharing an 11K-token prompt gets its first token in 4.4 s instead of 13.3 s.

With fastkernel 1.0.0 on a 24 GB M5 Pro, this gave 8,185 tokens of context at default settings. With the GPU given
20 GB, it gave 69,625. The context is the most text the model can hold at once: your prompt plus its answer.

Code: `runtime/engine/MemoryPlan.cpp`, `runtime/engine/RuntimeResources.mm`, `runtime/model/DFlashDraft.cpp`,
`runtime/main.mm`.

## 7. Rules that follow the GPU

The engine picks its kernels and memory plan from what the GPU reports: its family (chip generation) and core count.
It does not use a fixed table for one Mac.

- We picked a few tuned numbers on the M5 Max, such as how many GPU jobs to send early.
- On M3 and M4 GPUs, the engine keeps Splash's plans where ours don't apply.
- Wide checking stops at the widest size that stays exact: 32 rows on M5, 16 on M3/M4.

Code: `runtime/ops/Linear.cpp`, `runtime/model/QwenTarget.cpp`.

## Switches

Every change above has a switch. [SWITCHES.md](SWITCHES.md) lists them all, with their defaults.
