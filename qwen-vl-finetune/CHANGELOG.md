# Continued-Pretraining Changelog

Running log of changes made on this branch for Qwen3-VL-32B continued
pretraining (see DATA.md for the data rationale). Append an entry whenever a
training default, script, or pipeline behavior changes.

## Current defaults (2026-09-01)

| Setting | Value | Where |
|---|---|---|
| Mode | `full` \| `llm` (required arg) | `scripts/cpt_32b_sbatch.sh` |
| Nodes / time / partition | 16 / 1h / `36x2-a01r` | sbatch header |
| Model | `$LUSTRE/checkpoints/hf/Qwen3-VL-32B-Instruct` | `MODEL_PATH` |
| Data | `$LUSTRE/datasets/MINT-1T-HTML/processed` | `MINT1T_DATA_DIR` |
| Micro batch / grad accum | 2 / 8 (global 1,024 samples) | `cpt_32b.sh` |
| Sequence length | 8,192 | `--model_max_length` |
| Image budget | 200,704 px (≤196 tokens/image) | `MAX_PIXELS` |
| LR / schedule | 2e-5 peak, 10-step linear warmup, cosine→0 | `LR`, `WARMUP_STEPS` |
| Epochs | 10 | `--num_train_epochs` |
| Eval | 1,024 held-out samples, every 10 steps | `EVAL_SAMPLES`, `EVAL_STEPS` |
| Checkpointing | disabled; output to container-local `/results` | `OUTPUT_DIR` |
| Seed | 42 | `SEED` |

## 2026-09-02

- Warmup specified as an absolute step count (`WARMUP_STEPS`, default 10)
  instead of a 3% ratio: with ~325k samples the ratio meant ~95 warmup
  steps, so the 1-hour runs (~95 steps) spent their entire budget warming up.
- Eval hold-out split now uses the training `SEED` instead of a fixed
  constant (2024): each seed trains and evaluates on its own partition, so
  eval losses across seeds also reflect split variance.
- `tools/plot_losses.py`: one 2x2 figure (training loss, eval loss, learning
  rate, grad norm) from a directory of `slurm_*.out` logs — one line per
  run, shared legend = seed, title = mode.
- **Fixed: image pixel budget was silently ignored on transformers v5.**
  `update_processor_pixels` guarded its updates on `hasattr(min_pixels)`
  and `isinstance(size, dict)`; v5 pops those attributes into a non-dict
  `size`, so both branches skipped and every image was processed at native
  resolution (up to ~16.7M px, i.e. thousands of tokens per image). This is
  the actual root cause of the oversized-sample skips, the 53 GiB
  cross-entropy OOM, and the ZeRO-3 hang that followed a skipped image
  anchor. The budget is now set unconditionally through every
  representation and verified functionally at startup (a 2048x2048 probe
  must not exceed the token limit, else training aborts). No data changes.

## 2026-09-01

- Run-configuration banner printed by `cpt_32b.sh` on node 0 (seed, LR,
  batch, tune flags, data, eval settings) so logs record what ran; `SEED`
  exported explicitly by the sbatch launcher.
- **Exact token verification in preprocessing** (`--tokenizer`, wrapper env
  `TOKENIZER`): counts each sample with the real tokenizer and drops
  over-budget ones — the heuristic can still under-estimate rare scripts.
  Training-side skip hardened: oversized samples skip without retries and
  the dataset scans up to 32 neighbors (consecutive bad chunks from one
  document previously crashed the loader).
- **Fixed OOM from mis-chunked samples**: the converter counted whitespace
  words, so CJK/space-free text and unbroken junk (URLs, base64) produced
  chunks tokenizing to ~87k tokens, and the fp32 loss over them allocated
  50+ GiB. The converter now uses a token-aware `effective_words` estimate
  with character-level fallback splitting (re-chunk existing data — images
  are reused), and the dataset skips any sample that still tokenizes past
  `model_max_length` (truncation would break image/pixel alignment).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the sbatch env:
  variable-length packed sequences fragment the caching allocator, a common
  cause of OOM despite ample free memory (investigating persistent OOM at
  micro batch 1).
- Micro batch back to 2 (needed for the text-only filler slot in `full`
  mode); paid for by reducing `MAX_PIXELS` 451,584 → 200,704 (441 → 196
  tokens/image) — batch 2 at the larger budget OOMed because `data_flatten`
  packs the micro-batch into one sequence.
- Default LR raised 2e-6 → 2e-5.
- Eval switched from per-epoch to every N optimizer steps (`EVAL_STEPS`,
  default 3).
- `MINT1T_DATA_DIR` default moved to `datasets/MINT-1T-HTML/processed`.
- sbatch launcher requires a `full`/`llm` mode argument (exits otherwise);
  each mode sets a consistent bundle of tune/text-only flags.

## 2026-08-30

- **S1 mimicry** (Qwen3-VL tech report, arXiv:2511.21631): whole model
  trainable by default at seq len 8,192 with text-only data mixed in.
- **Image-guaranteed batches** (`--require_image_per_batch`): every train
  micro-batch is anchored by ≥1 image sample and the eval split is
  image-only, so a trainable vision tower is safe under ZeRO-3 with
  text-only data present (gradient collectives stay ordered identically
  across ranks). Excess text-only samples are dropped per epoch.
- Fixed per-image token accounting: 576×28×28 = up to 441 Qwen3-VL tokens
  (one per 32×32 px), not 144; converter `--image-word-cost` default set to
  352 so chunks stay within the 8,192 context.
- Preprocessing speedups: download timeout 20s → 3s (dead hosts dominate
  wall time), workers 16 → 64.
- Added `DATA.md` (data rationale) and `scripts/preprocess_mint1t.sh`
  (converts `<data_dir>/data_v1_1/*.parquet` → `<data_dir>/processed`).

## 2026-08-28 .. 2026-08-29

- Slurm launcher `scripts/cpt_32b_sbatch.sh`: head-node rendezvous,
  container launch, lustre mount; `cpt_32b.sh` made multi-node aware
  (`--nnodes/--node_rank` from Slurm env).
- Guard: `allow_text_only` is rejected with a trainable vision tower unless
  image-guaranteed batching is on (prevents a silent 30-min NCCL hang from
  divergent gradient reduce-scatter order).
- Held-out evaluation (`--eval_samples`, fixed-seed split, disjoint from
  training); dummy vision forward extended to eval mode.
- Training output moved to container-local `/results` (no stale-checkpoint
  resume issues); checkpoint saving disabled for dev runs.
- `tune_llm_last_n_layers` (`LLM_LAST_N`) for memory-constrained
  single-node development.
- Env knobs added: `SEED`, `LR`, `REPORT_TO` (default `none`),
  `OUTPUT_DIR`, `MAX_PIXELS`/`MIN_PIXELS`.

## 2026-08-27

- **Continued pretraining support**: `--train_on_all_tokens` (full-sequence
  next-token loss; vision pad positions excluded) + `scripts/cpt_32b.sh`.
- **MINT-1T preprocessing** (`tools/preprocess_mint1t.py`): downloads and
  validates images (content-hash dedup, dead links dropped with `<image>`
  tags kept aligned), chunks documents to a word budget, writes the repo's
  annotation format; progress bar with parquet-metadata totals.
- **Transformers v5 compatibility**: `--warmup_ratio` → `--warmup_steps`
  (float = ratio), no `use_fast=False`, `.visual`/`.language_model` resolved
  through the inner model, signature-agnostic `create_causal_mask`
  replacement, `ModelOutput`-aware dummy vision forward.
- **ZeRO-3 text-only hang diagnosed**: ranks whose batch has no images skip
  the vision tower and miss its parameter all-gathers → NCCL watchdog
  timeout. First fix: zero-weighted dummy vision forward
  (`--allow_text_only`); converter drops text-only chunks by default
  (`--keep-text-only` to retain).
