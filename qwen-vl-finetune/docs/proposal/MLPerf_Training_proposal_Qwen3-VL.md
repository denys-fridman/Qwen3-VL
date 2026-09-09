# MLPerf Training new benchmark proposal: Vision-Language Model (Qwen3-VL)

## Motivation

Vision-language models (VLMs) have become the default form of frontier LLMs:
current flagship models are natively multimodal, and the open-weight
ecosystem has followed (Qwen-VL, InternVL, Llama 3.2 Vision, Gemma 3, Pixtral,
Molmo). Multimodal pretraining — training a vision encoder, a projector and a
language model jointly on interleaved image-text documents — is now a
substantial share of large-scale training compute, yet the MLPerf Training
suite contains no vision-language *training* workload: the LLM benchmarks
(Llama 3.1 8B/405B pretraining, Llama 2 70B LoRA) are text-only, and the
vision benchmarks (Stable Diffusion, FLUX, RetinaNet) do not exercise a
language model.

**Core concept.** A VLM couples three components: a vision transformer (ViT)
that turns each image into a variable number of visual tokens (native /
dynamic resolution), a small projector ("merger") that maps them into the
LLM embedding space, and a decoder-only LLM that consumes the interleaved
sequence of text and visual tokens with multimodal positional encoding
(M-RoPE). Training on interleaved web documents has system characteristics
that no current benchmark covers: two heterogeneous networks with very
different shapes per step, sequences whose composition (text vs. image
tokens) varies sample to sample, a data pipeline dominated by image decoding
and resizing, and modality-dependent execution paths whose collectives must
stay aligned across data-parallel ranks.

## Literature study

Possible model candidates (open weights, dense unless noted):

* **Qwen3-VL (2025/10; technical report arXiv:2511.21631)**
  * Size: dense 2B / 4B / 8B / **32B** and MoE 30B-A3B / 235B-A22B.
  * Architecture: 27-layer ViT with 16-px patches and 2×2 spatial merge
    (one token per 32×32 px, native dynamic resolution), MLP merger,
    Qwen3 decoder LLM; interleaved-MRoPE; **DeepStack** injects multi-level
    ViT features into early LLM layers; 256K context.
  * Availability: weights public (Apache-2.0); training recipe described in
    detail (4 pretraining stages with data mixture, token budgets and sequence
    lengths); an official fine-tuning framework (HF Trainer + DeepSpeed) is
    public, full pretraining code is not.
* **InternVL3 (2025/04)**
  * Size: 1B – 78B (InternViT-300M/6B encoder + Qwen2.5 / InternLM3 LLMs).
  * Architecture: ViT + MLP projector + LLM, "native multimodal pretraining".
  * Availability: weights and the InternVL training code are public.
* **Llama 3.2 Vision (2024/09)**
  * Size: 11B and 90B.
  * Architecture: frozen-LLM-friendly cross-attention vision adapter rather
    than early fusion — a less representative architecture for today's
    natively multimodal models.
  * Availability: weights under the Llama license; no training code.
* **Gemma 3 (2025/03)**
  * Size: 4B / 12B / 27B with a 400M SigLIP encoder.
  * Availability: weights public; no training code or recipe details.

## Why should we add a VLM to MLPerf Training?

1. **Relevance to modern LLM training.** Frontier models are trained
   multimodally from pretraining onward; the benchmark suite currently
   measures text-only LLM training and image generation, but not the joint
   vision-language training that dominates real workloads.
2. **Architectural and system coverage.** VLM training stresses components
   dense LLM benchmarks do not: a second (vision) network with its own
   parallelism and memory profile, variable-length packed sequences of mixed
   modality, image preprocessing throughput in the input pipeline, and
   cross-rank alignment of modality-dependent computation. It also exercises
   long-sequence attention (8K here, up to 256K in the model's later
   pretraining stages) on realistic interleaved data.

## Preferred VLM model characteristics

1. **Model size.** Tens of billions of dense parameters — large enough to
   require multi-node training and represent production-scale VLM
   pretraining, small enough for a reasonable compute budget per run.
2. **Availability.** Public weights and a detailed description of the
   training methodology (stages, data mixture, sequence lengths).
3. **Training code / recipe.** A public training framework or a recipe
   reproducible in open-source frameworks.
4. **Data.** An openly licensed interleaved image-text corpus matching the
   model's own pretraining data type.

## Proposal: Qwen3-VL-32B on MINT-1T

Given the above, we propose **continued multimodal pretraining of
Qwen3-VL-32B on the MINT-1T interleaved corpus**, mimicking Stage 1
("Multimodal Pre-Training") of the Qwen3-VL recipe:

* **Model:** Qwen3-VL-32B-Instruct public checkpoint (33.4B parameters:
  32.8B in the LLM — 64 layers, hidden 5120, 64 heads / 8 KV heads, vocab
  151,936 — plus a 27-layer ViT and merger). All parameters trained.
* **Codebase:** the official Qwen3-VL fine-tuning framework (HF Trainer +
  DeepSpeed ZeRO-3), extended on our branch with full-sequence
  next-token loss (continued pretraining rather than SFT), interleaved-data
  preprocessing, image-guaranteed batch construction, held-out evaluation and
  transformers-v5 compatibility. A Megatron-Bridge / NeMo recipe would be the
  natural submission-grade reference and is the planned next step.
* **Dataset:** MINT-1T (HTML subset) — the largest open interleaved
  image-text corpus (~1T tokens, CC-BY-4.0), i.e. the public proxy for the
  interleaved data that dominates Qwen3-VL's Stage 1. Documents are chunked
  to fit the 8,192-token context; images are stored at native resolution and
  capped at training time; text-only chunks are retained (as in the paper's
  data mixture) and made safe for ZeRO-3 by anchoring every micro-batch with
  an image sample.
* **Quality metric:** per-token cross-entropy (eval loss) on a held-out set
  of MINT-1T samples, evaluated periodically during training; the benchmark
  target is a fixed eval loss.

## Qwen3-VL-32B details

### Goals

1. Reasonable compute budget: **64–96 GB200 GPU-hours per run (BF16)**; the
   target loss is chosen so that a run lands inside this budget (see table).
2. Low variance: measured CV of samples-to-target of 2–4% over 10 seeds at
   the candidate targets.
3. Evaluation cost below 5% of the training budget (currently ~14%; see
   open items).
4. Scalability: reference convergence points for GBS 1k / 2k / 4k to be
   produced.

### Training

Scenario: **continued pretraining from the publicly available checkpoint**
(weights only; no optimizer state). The checkpoint is already available to
everyone, so no additional artifact needs distributing. The
public checkpoint is post-trained (Instruct), but on interleaved web data the
loss is far from saturated: eval loss falls from 2.98 to 2.42 over ~125k
samples. Training from scratch is not viable at this scale, and Qwen3-VL base
checkpoints are not released.

All experiments (10 seeds) were run with the following setup:

* 64 GB200 GPUs (16 nodes × 4), DeepSpeed ZeRO-3, BF16
* Global batch size 1,024 samples (micro batch 2 × grad accumulation 8 × 64)
* Sequence length 8,192 (documents packed per micro-batch with per-document
  attention boundaries); image budget 200,704 px per image (≤196 visual tokens)
* All parameters trainable (vision encoder, merger, LLM)
* Optimizer: AdamW (β₁ 0.9, β₂ 0.999, weight decay 0.01), gradient clipping 1.0
* LR schedule: cosine with linear warmup — peak LR 5e-6, warmup 10 steps,
  150 planned steps
* Evaluation: every 5 iterations on 4,096 held-out samples (seed-specific
  held-out split, disjoint from training)

Performance:

* Train step (1,024 samples): ~40 s median (≈25 samples/s at 64 GPUs)
* One evaluation (4,096 samples): ~30 s
* Runs reached 120–130 steps (~125k samples) before the 1.5 h wall clock;
  by then the cosine schedule was essentially complete (LR ≈ 3e-7).
* 9 of 10 seeds completed; seed 10 stopped at step 44 with an NCCL
  collective timeout (see open items).

### Learning curves (10 seeds)

![Training curves: training loss, evaluation loss, learning rate and gradient
norm vs. samples processed, one line per seed](training_curves.png)

Training loss drops from ~3.0 to ~2.4 and the ten seeds are statistically
indistinguishable after the first ~20k samples. Gradient norm shows isolated
spikes (up to ~700, clipped to 1.0) during the first ~8k samples of warmup
and stays below ~10 afterwards.

### Convergence variance

Samples to first reach a target eval loss, over 10 seeds (eval every 5,120
samples; targets ≥ 2.60 are reached within the first few evaluations and are
therefore quantized to 0% CV):

| Target eval loss | Seeds reached | Mean samples | Std | CV | Min – max |
|---|---|---|---|---|---|
| 2.43 | 9/10 | 103,538 | 2,258 | **2.2%** | 102,400 – 107,520 |
| 2.44 | 9/10 | 88,747 | 2,560 | 2.9% | 87,040 – 92,160 |
| 2.45 | 9/10 | 79,644 | 2,698 | **3.4%** | 76,800 – 81,920 |
| 2.50 | 9/10 | 46,649 | 1,707 | 3.7% | 46,080 – 51,200 |
| 2.55 | 10/10 | 27,648 | 2,644 | 9.6% | 25,600 – 30,720 |
| 2.60 | 10/10 | 20,480 | 0 | 0.0% | 20,480 |

(9/10 at the lower targets reflects the seed-10 hang, not divergence.)

Candidate targets: **eval loss 2.43** (~104k samples, ~80 GB200 GPU-hours
including evaluation, CV 2.2%) sits inside the 64–96 GPU-hour goal;
**2.45** (~80k samples, ~60 GPU-hours, CV 3.4%) falls just below it. The CV
values are upper bounds set by the 5,120-sample evaluation granularity;
denser evaluation near the target would tighten them further.

## Summary

| | Continued pretraining from the public Qwen3-VL-32B checkpoint |
|---|---|
| Compute budget | 64–96 GB200 GPU-hours per run (goal); ~80 GPU-hours at target loss 2.43 (~105k samples at GBS 1,024) |
| Variance | CV = 3.4% at loss 2.45; 2.2% at loss 2.43 (10 seeds) |
| Evaluation cost | ~30 s per 4,096-sample eval; ~14% of run time at the current cadence, <5% with eval every 10 steps or 1,024 samples |
| Scalability | To be checked (RCPs for GBS 1k / 2k / 4k) |

## Open items / next steps

1. **Evaluation cost:** move to every 10 steps and/or 1,024 samples to meet
   the 5% goal (current 24 evals × 30 s ≈ 12 min of an ~88 min run).
2. **Data robustness:** ~3–4% of samples per run still exceed the 8,192
   context after heuristic chunking (non-Latin scripts) and are skipped at
   load time; the skip substitution can break the image-per-batch guarantee
   and caused the seed-10 NCCL hang. Fix: exact tokenizer-based chunking in
   preprocessing and image-aware substitution in the loader.
3. **Reference implementation:** port the recipe to Megatron-Bridge / NeMo
   (TP/PP/CP support, FP8 policy, checkpoint format) and define the
   reference convergence points for GBS 1k / 2k / 4k.
4. **Target selection:** choose the target eval loss (2.45 vs. 2.43) and
   fix the held-out evaluation set (currently seed-specific) as a shared
   benchmark artifact.
5. **Scale sensitivity:** repeat the variance study at 2× and 4× GBS with
   LR scaling to establish RCPs.
