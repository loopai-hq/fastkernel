# Switches

fastkernel's changes to Splash each have an environment variable. Set it in front of the serve command, for example
`SPLASH_DRAFT_AHEAD=0 ./splash serve --model incoai/Qwen3.8-27B-Splash`. The defaults are the tuned settings.

- Speed switches are on by default. `=0` turns one off and runs Splash 1.3.0's path for that part.
- With every fork switch off, execution matches Splash 1.3.0. The memory plan's margin policy (2% of memory in place
  of a fixed 1 GiB) has no switch and stays on.
- [WHATS-INSIDE.md](WHATS-INSIDE.md) explains what each part does and what it gained.

## Speed

| Switch | Default | Effect |
|---|---|---|
| `SPLASH_ONE_LANE_SPLIT` | on | One request: long, thin 4-bit multiplications split across more GPU cores. |
| `SPLASH_NARROW_SPLIT` | on | Extends that split to narrow outputs, such as the draft model's. |
| `SPLASH_M16_NARROW_SPLIT` | on | The same split for two requests at once. |
| `SPLASH_M24_NARROW_SPLIT` | on | The same split for three requests at once. |
| `SPLASH_SPLIT128` | auto | Splash 1.3.0's Split128 kernels where Splash 1.3.0 picks them. `0`: fastkernel's split kernels only. `1`: Split128 first. |
| `SPLASH_INPUT_FUSED_SUMS` | on | The input norm hands each projection the sums of its input, so the split multiplication skips recomputing them. |
| `SPLASH_M16_INPUT_SUMS` | on | The same for two requests. |
| `SPLASH_M24_INPUT_SUMS` | on | The same for three requests. |
| `SPLASH_M24_PAD3` | on | Three requests run their input projections as four lanes, on the faster 32-row kernel. |
| `SPLASH_FFN_FUSED_SUMS` | on | The FFN's gate/up kernel also writes the input sums the down projection needs. |
| `SPLASH_M16_FFN_SUMS` | on | The same for two requests. |
| `SPLASH_SPLIT4_INPUT_DIV` | 4 | One request: the split kernel's thread groups, one per 4 tiles. `0` turns it off. |
| `SPLASH_SPLIT4_HOIST` | on | Scheduling variant of the split kernels. Same output bytes. |
| `SPLASH_SPLIT4_FOOTER` | on | Scheduling variant of the split kernels' last step. Same output bytes. |
| `SPLASH_SPLIT4_M16` | on | Both variants for two requests. |
| `SPLASH_STAGED_NORM_WIDE` | on | Wide RMS norms stage their row in GPU threadgroup memory. Same output bytes. |
| `SPLASH_GDN_VALUE_PARTS` | 4 | The GDN layers' state update runs in four parallel parts. `0` turns it off. |
| `SPLASH_GDN_FUSED_SUMS` | on | The GDN output kernel also prepares the next projection's input sums. |
| `SPLASH_GDN_SCAN_NOSTORE` | on | The GDN layers' checking scans don't store their running state; the commit replays the kept rows instead, with the same bits. `0`: the scans store it. |
| `SPLASH_GDN_DEFER` | on | One request: each step's GDN state commit runs inside the next step's scan instead of as its own GPU work. `0`: it commits at once. |
| `SPLASH_WIDE_GDN_SINGLE` | parts | Wide checks run each GDN layer as one pass, in four parts. `1`: one pass. `0`: Splash's chain of small launches. |
| `SPLASH_BLOCK_VERIFY` | on | Sampled answers: the full model judges the draft's guesses as one block. |
| `SPLASH_SAMPLER_TOPK32` | on | Sampled answers with top-k at most 32 and no min-p: the top-k and top-p search reads each vocabulary shard's 32 most likely tokens instead of the whole vocabulary; the picks are the same. `0`: the whole vocabulary. |
| `SPLASH_DRAFT_AHEAD` | on | The next draft runs on the GPU while the CPU reads the current result. |
| `SPLASH_DRAFT_AHEAD_GRAMMAR` | on | Drafting ahead also runs for constrained requests, such as tool calls. |
| `SPLASH_DRAFT_AHEAD_LOOKUP_QUIET` | 8 | Drafting ahead starts after this many steps without a prompt-lookup step. |
| `SPLASH_DRAFT_TAU` | 0.85 | The draft model's own sampling temperature. `1` keeps its probabilities unchanged. |
| `SPLASH_DRAFT_TOP_P` | 0.99 | The draft model's own top-p. `1` turns it off. |
| `SPLASH_PROMPT_LOOKUP` | on | One request: guesses can come straight from your prompt instead of the draft model. |
| `SPLASH_WIDE_PROMPT_LOOKUP` | on | Prompt-lookup guesses are checked 16 rows at once, each row byte-exact with an 8-row check. |
| `SPLASH_WIDE_LOOKUP32` | 1 | Wide checks go up to 32 rows where the GPU keeps them exact. `0`: 16 rows. |
| `SPLASH_LOOKUP_ADAPTIVE` | on | Each request adapts how long a repeated stretch must be before lookup guesses from it. |
| `SPLASH_LOOKUP_MIN_MATCH` | 8 | The shortest repeated stretch, in tokens, that lookup guesses from. |
| `SPLASH_WIDE_LOOKUP_MIN_MATCH` | 16 | The same for wide checks, with `SPLASH_LOOKUP_ADAPTIVE=0`. |
| `SPLASH_STRIPED_VERIFY` | on | Checking attention splits by absolute page, so a row's bytes don't depend on its place; wide checks need it. `0`: Splash's balanced splits. |
| `SPLASH_CHUNKED_SUBMIT` | 48 | A long GPU command sends its first 48 jobs before the rest is encoded. `0` turns it off. |
| `SPLASH_STREAMED_SUBMIT` | 48 | Each step's first 48 GPU jobs start while the CPU builds the rest. `0` turns it off. |
| `SPLASH_GRAMMAR_CHAIN` | on | A constrained step (tool calls, JSON) runs as one GPU command. |
| `SPLASH_KEEP_PREFILL_CHECKPOINTS` | on | Prompt checkpoints stay as a cache, so a second request that shares a long prompt starts sooner. |
| `SPLASH_ANE` | on | Qwen3.8-27B prefill: the engine times the GPU against a GPU and Neural Engine split at startup and uses the split where it is faster. `0` (or `serve --disable-ane`): GPU alone. |

## Settings

| Switch | Default | Effect |
|---|---|---|
| `SPLASH_TEXT_ONLY` | off | `1` skips the image-reading weights, and image requests are refused. For 24 GB Macs. |
| `SPLASH_DRAFT_HEAD_IDS` | unset | A token-ID file for the draft's smaller vocabulary. The Quick start sets it to `data/head-ranked.u32`. |
| `SPLASH_DRAFT_HEAD_ROWS` | 98304 | How many IDs from that file the draft uses. |

## Diagnostics

| Switch | Default | Effect |
|---|---|---|
| `SPLASH_ROW_HASH` | off | `1` prints a hash of each accepted row's output, to compare two runs row by row. |
| `SPLASH_GPU_GAP_LOG` | off | Set: prints each GPU command's start and end time. |
| `SPLASH_HOST_PHASE_LOG` | off | Set: prints the CPU's phase marks on the same clock. |
| `SPLASH_REQUEST_STATS` | off | `1` prints per-request draft_ahead / prompt_lookup / wide_lookup lines at request end, and the wide lookup's row-stable verify rows at startup. |

## Benchmark tool (`dev/benchmarks/decode_profile.mm`)

| Switch | Default | Effect |
|---|---|---|
| `SPLASH_DISPATCH_LIST` | unset | A file path: writes each profiled step's list of GPU jobs. |
| `SPLASH_PROFILE_TOKENS` | unset | Set: prints each lane's output tokens. |
| `SPLASH_AB_SWITCH`, `SPLASH_AB_MODE`, `SPLASH_AB_ON`, `SPLASH_AB_WIDTH`, `SPLASH_AB_PROMPTS`, `SPLASH_AB_CYCLES`, `SPLASH_AB_BLOCK`, `SPLASH_AB_B3_ROUNDS` | unset | An in-process A/B of one switch: its name, lockstep or block order, its on value, lanes, prompts, steps, block length, rounds. |
