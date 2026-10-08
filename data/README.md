# data

`head-ranked.u32`: the token ids for the drafter's small vocabulary head (`SPLASH_DRAFT_HEAD_IDS`).

- **Format:** 248,320 little-endian u32 values, one per token id of the Qwen3.8-27B tokenizer, each id exactly once.
  Special tokens come first, then ids by how often the model emitted them, then the rest of the vocabulary.
  993,280 bytes; sha256 `9eca7a0f6a9671fc8a1202d49cbad59151db5df67ba3e39ef1e86384bdb0c2c5`.
- **Where it comes from:** token counts of Qwen3.8-27B's own regenerated responses to general, multilingual, code and
  agent prompts, which we produced. The file holds ids only, no text.
- **Use:** `SPLASH_DRAFT_HEAD_IDS=$PWD/data/head-ranked.u32` on the `pulsar serve` command. The engine gathers the
  top 98,304 rows (`SPLASH_DRAFT_HEAD_ROWS`) from the target model's head at startup. Every token is still checked by
  the full model, so output stays exact. The file only works with this model's tokenizer.
