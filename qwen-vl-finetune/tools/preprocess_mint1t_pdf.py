#!/usr/bin/env python3
"""Preprocess the MINT-1T PDF subset into the qwen-vl-finetune format.

Input: a directory of .tar shards. Each shard holds <id>.json / <id>.tiff
pairs, one per document:

  * json: {"texts": [str|null, ...], "images": [null|"page_<p>_image_<xref>", ...],
           "image_metadata": [{"page": p, "xref": x, "sha256": ..., "width": w,
           "height": h}, ...], "language_id_whole_page_fasttext": {"en": 0.87}, ...}
    `texts` and `images` are parallel arrays (one non-null per position).
  * tiff: a multi-frame TIFF with one frame per image_metadata entry, in order.

Images are extracted from the TIFF (no downloads), saved as
images/<sha256>.jpg (deduplicated by content hash), and documents are chunked
exactly like tools/preprocess_mint1t.py, writing:

    <output_dir>/images/<sha256>.jpg
    <output_dir>/annotations.jsonl

Usage:
    python tools/preprocess_mint1t_pdf.py \
        --data-files "/path/to/MINT-1T-PDF/*.tar" \
        --output-dir /path/to/MINT-1T-PDF/processed \
        --tokenizer /path/to/Qwen3-VL-32B-Instruct --keep-text-only
"""

import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from itertools import zip_longest
from pathlib import Path

from PIL import Image, ImageSequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_mint1t import (  # noqa: E402  (shared chunking / verification)
    IMAGE_WORD_COST,
    MIN_IMAGE_SIDE,
    blocks_to_samples,
    tqdm,
)

IMAGE_REF = re.compile(r"page_(\d+)_image_(\d+)")

# per-process state (set by the pool initializer)
_TOKENIZER = None


def _init_worker(tokenizer_path):
    global _TOKENIZER
    if tokenizer_path:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)


def save_frame(frame, out_path):
    if out_path.exists():
        return
    img = frame.convert("RGB")
    if min(img.size) < MIN_IMAGE_SIDE:
        raise ValueError(f"image too small: {img.size}")
    tmp = out_path.with_name(out_path.name + ".tmp")
    img.save(tmp, "JPEG", quality=95)
    os.replace(tmp, out_path)


def doc_to_blocks(doc, tiff_bytes, images_dir, stats):
    """Flatten a document into ordered ("image", relpath) / ("text", str) blocks,
    extracting referenced images from the document's multi-frame TIFF."""
    texts = doc.get("texts") or []
    images = doc.get("images") or []
    metadata = doc.get("image_metadata") or []
    # "page_<p>_image_<xref>" -> frame index (position in image_metadata)
    ref_to_index = {(m.get("page"), m.get("xref")): i for i, m in enumerate(metadata)}

    frames = None
    if tiff_bytes and any(images):
        try:
            tiff = Image.open(io.BytesIO(tiff_bytes))
            frames = [f.copy() for f in ImageSequence.Iterator(tiff)]
        except Exception:
            stats["tiff_failed"] += 1
            frames = None

    blocks = []
    for ref, text in zip_longest(images, texts):
        if ref:
            m = IMAGE_REF.fullmatch(ref)
            idx = ref_to_index.get((int(m.group(1)), int(m.group(2)))) if m else None
            ok = frames is not None and idx is not None and idx < len(frames)
            if ok:
                name = metadata[idx].get("sha256") or f"{doc.get('pdf_name', 'doc')}_{ref}"
                try:
                    save_frame(frames[idx], images_dir / f"{name}.jpg")
                    blocks.append(("image", f"images/{name}.jpg"))
                    stats["images_ok"] += 1
                except Exception:
                    stats["images_failed"] += 1
            else:
                stats["images_failed"] += 1
        if text:
            text = text.replace("﻿", "").strip()
            if text:
                blocks.append(("text", text))
    return blocks


def process_tar(args_tuple):
    """Process one shard: returns (samples, stats)."""
    tar_path, images_dir, max_words, keep_text_only, image_word_cost, min_en_score = args_tuple
    stats = {"docs": 0, "docs_skipped_lang": 0, "images_ok": 0, "images_failed": 0,
             "tiff_failed": 0, "samples_dropped_too_long": 0}
    samples = []
    with tarfile.open(tar_path) as tar:
        members = {}
        for m in tar.getmembers():
            if not m.isfile():
                continue
            stem, ext = os.path.splitext(os.path.basename(m.name))
            members.setdefault(stem, {})[ext.lower()] = m
        for stem, parts in members.items():
            if ".json" not in parts:
                continue
            doc = json.load(tar.extractfile(parts[".json"]))
            if min_en_score > 0:
                lang = doc.get("language_id_whole_page_fasttext") or {}
                if lang.get("en", 0.0) < min_en_score:
                    stats["docs_skipped_lang"] += 1
                    continue
            tiff_member = parts.get(".tiff") or parts.get(".tif")
            tiff_bytes = tar.extractfile(tiff_member).read() if tiff_member else None
            blocks = doc_to_blocks(doc, tiff_bytes, images_dir, stats)
            doc_samples, dropped = blocks_to_samples(
                blocks, max_words, keep_text_only, image_word_cost, _TOKENIZER
            )
            stats["samples_dropped_too_long"] += dropped
            stats["docs"] += 1
            samples.extend(doc_samples)
    return samples, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-files", required=True, help="Glob of input .tar shards")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 8,
                        help="Worker processes (one shard at a time each)")
    parser.add_argument("--max-words", type=int, default=5000,
                        help="Word budget per output sample; keep it below model_max_length after tokenization")
    parser.add_argument("--image-word-cost", type=int, default=IMAGE_WORD_COST,
                        help="Word-budget cost per image when chunking")
    parser.add_argument("--keep-text-only", action="store_true",
                        help="Keep samples without images (needs --allow_text_only or image-anchored batches in training)")
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer path for exact token counting; over-budget samples are dropped")
    parser.add_argument("--min-en-score", type=float, default=0.0,
                        help="Skip documents whose fastText English score is below this (0 disables)")
    parser.add_argument("--max-shards", type=int, default=None, help="Stop after N shards (smoke tests)")
    args = parser.parse_args()

    shards = sorted(glob.glob(args.data_files))
    if not shards:
        raise FileNotFoundError(f"no files match {args.data_files}")
    if args.max_shards:
        shards = shards[: args.max_shards]

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = output_dir / "annotations.jsonl"

    jobs = [(s, images_dir, args.max_words, args.keep_text_only, args.image_word_cost, args.min_en_score)
            for s in shards]
    totals = {"shards": 0, "docs": 0, "docs_skipped_lang": 0, "samples": 0, "images_ok": 0,
              "images_failed": 0, "tiff_failed": 0, "dropped_too_long": 0}
    progress = tqdm(total=len(shards), unit="shard", dynamic_ncols=True) if tqdm else None
    with open(annotation_path, "w") as fout, ProcessPoolExecutor(
        max_workers=args.num_workers, initializer=_init_worker, initargs=(args.tokenizer,)
    ) as pool:
        for samples, stats in pool.map(process_tar, jobs):
            totals["shards"] += 1
            totals["docs"] += stats["docs"]
            totals["docs_skipped_lang"] += stats["docs_skipped_lang"]
            totals["samples"] += len(samples)
            totals["images_ok"] += stats["images_ok"]
            totals["images_failed"] += stats["images_failed"]
            totals["tiff_failed"] += stats["tiff_failed"]
            totals["dropped_too_long"] += stats["samples_dropped_too_long"]
            for sample in samples:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if progress:
                progress.update(1)
                progress.set_postfix(docs=totals["docs"], samples=totals["samples"],
                                     img_ok=totals["images_ok"], img_fail=totals["images_failed"])
            else:
                print(f"shards={totals['shards']}/{len(shards)} docs={totals['docs']} "
                      f"samples={totals['samples']} images ok={totals['images_ok']} "
                      f"failed={totals['images_failed']}", flush=True)
    if progress:
        progress.close()

    print(f"\nWrote {totals['samples']} samples from {totals['docs']} documents to {annotation_path}")
    print(f"Images in {images_dir}: {totals['images_ok']} ok, {totals['images_failed']} failed/skipped, "
          f"{totals['tiff_failed']} unreadable TIFFs")
    if args.min_en_score > 0:
        print(f"Documents skipped by language filter: {totals['docs_skipped_lang']}")
    if args.tokenizer:
        print(f"Samples dropped as over token budget: {totals['dropped_too_long']}")
    extra = " --allow_text_only True" if args.keep_text_only else ""
    print("\nRegister in qwenvl/data/__init__.py as:\n"
          f'  MINT1T_PDF = {{"annotation_path": "{annotation_path}", "data_path": "{output_dir}"}}\n'
          f"and train with --dataset_use mint1t_pdf --train_on_all_tokens True{extra}")


if __name__ == "__main__":
    main()
