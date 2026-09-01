# Data Choice: MINT-1T for Qwen3-VL Continued Pretraining

## Goal

Mimic Stage 1 (S1, "Multimodal Pre-Training") of the Qwen3-VL training
recipe (arXiv:2511.21631): full-parameter training on interleaved
vision-language data at a sequence length of 8,192, with text-only data
mixed in.

## Why interleaved image-text documents

S1's core data modality is interleaved documents — web pages where images
appear in their natural textual context — rather than isolated caption
pairs. Training on them teaches the model to ground images in surrounding
discourse, handle multiple images per context, and carry information across
image boundaries; this is also the format that motivates Qwen3-VL's native
interleaved-context design. Full-sequence next-token loss
(`--train_on_all_tokens`) makes this continued *pretraining* rather than
instruction tuning: every text token is a target, and only vision
placeholder tokens are excluded.

## Why MINT-1T

MINT-1T is the largest open interleaved corpus (~1T tokens — the same order
as S1's token budget), so it is the closest public proxy for Qwen's
proprietary interleaved data. Its documents ship as parallel
`images`/`texts` arrays with content hashes, which map directly onto Qwen's
`<image>`-placeholder training format.

## Why text-only samples are kept (`--keep-text-only`)

The Qwen3-VL report mixes text-only data into every pretraining stage "to
maintain the LLM's strong language abilities" — heavy image-text training
measurably degrades a model's pure-text quality otherwise. Chunks whose
images are unavailable become that text-only mix. At train time an
image-guaranteed batch sampler keeps at least one image in every
micro-batch, which is what makes this mixture safe under DeepSpeed ZeRO-3
with a trainable vision tower.

## Preprocessing choices

- **Images are downloaded and validated offline** (dead URLs from the
  2011–2024 crawl are dropped, with `<image>` tags kept aligned); training
  never touches the network. Content-hash filenames dedup identical images
  across documents.
- **Documents are chunked to a ~5,000-word budget** (images costed at 352
  words each, sized for the training pixel budget) so every sample
  tokenizes below the 8,192 context — truncating mid-image would break
  pixel/token alignment.
- **Native resolution is preserved on disk**; the per-image pixel budget
  (default 576×28×28 → up to 441 vision tokens for Qwen3-VL) is applied at
  training time. Changing it only requires re-chunking with a matching
  `--image-word-cost` — downloaded images are reused.

## Known deltas vs. the real S1

Token scale (we train epochs over a subset, not 1T tokens), undisclosed
LR/schedule/batch size in the paper, an upper bound on the text-only ratio
(one text sample per image anchor per batch), and a pixel ceiling in place
of unbounded native resolution.

## Usage

```bash
# raw MINT-1T parquet shards expected at <data_dir>/data_v1_1/*.parquet
bash scripts/preprocess_mint1t.sh <data_dir>
```

This downloads/validates the images and writes `annotations.jsonl` plus
`images/` to `<data_dir>/processed`. Point training at it via
`MINT1T_DATA_DIR=<data_dir>/processed sbatch scripts/cpt_32b_sbatch.sh <full|llm>`
(the default already matches
`/lustre/fsw/coreai_mlperf_training/users/dfridman/datasets/MINT-1T-HTML/processed`).
