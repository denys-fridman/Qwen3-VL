#!/usr/bin/env python3
"""Preprocess MINT-1T-style interleaved documents into the qwen-vl-finetune format.

Input: parquet shards (or json/jsonl) where each row is a document with parallel
`images`/`texts` arrays — at each position one of the two is non-null, images
are URLs (see datapoint_example.json).

For every document this script downloads the images (skipping dead links and
tiny/broken files), splits the document into chunks that fit the training
context, and writes:

    <output_dir>/images/<image_hash>.jpg
    <output_dir>/annotations.jsonl

Each output line is one training sample in the repo's annotation format — the
whole interleaved chunk as a single human turn with <image> placeholders:

    {"image": ["images/abc.jpg", ...],
     "conversations": [{"from": "human", "value": "text\n\n<image>\n\ntext..."}]}

There is no assistant turn, so these samples carry no SFT loss — train them
with --train_on_all_tokens True (continued pretraining).

Usage:
    python tools/preprocess_mint1t.py \
        --data-files "./mint1t/data_v1_1/*.parquet" \
        --output-dir ./mint1t/processed \
        --num-workers 16
"""

import argparse
import glob
import hashlib
import io
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path

import requests
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; qwenvl-mint1t-preprocess)"}
# Budget estimate per image when chunking, in "words". Qwen3-VL spends one
# token per 32x32 pixels, so --max_pixels 576*28*28 costs up to 441 tokens
# per image (~340 words at ~1.3 tokens/word); 352 leaves margin. Scale it
# with --image-word-cost if you change the training max_pixels.
IMAGE_WORD_COST = 352
MIN_IMAGE_SIDE = 28  # below one ViT patch the processor rejects the image


def iter_documents(data_files):
    files = sorted(glob.glob(data_files))
    if not files:
        raise FileNotFoundError(f"no files match {data_files}")
    for path in files:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            yield from pq.read_table(path).to_pylist()
        elif path.endswith(".jsonl"):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        else:
            with open(path) as f:
                data = json.load(f)
            yield from (data if isinstance(data, list) else [data])


def count_documents(data_files):
    total = 0
    for path in sorted(glob.glob(data_files)):
        try:
            if path.endswith(".parquet"):
                import pyarrow.parquet as pq

                total += pq.ParquetFile(path).metadata.num_rows
            elif path.endswith(".jsonl"):
                with open(path) as f:
                    total += sum(1 for line in f if line.strip())
            else:
                with open(path) as f:
                    data = json.load(f)
                total += len(data) if isinstance(data, list) else 1
        except Exception:
            return None
    return total


def download_image(url, out_path, timeout):
    if out_path.exists():
        return
    resp = requests.get(url, timeout=timeout, headers=HEADERS)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    if min(img.size) < MIN_IMAGE_SIDE:
        raise ValueError(f"image too small: {img.size}")
    tmp = out_path.with_name(out_path.name + ".tmp")
    img.save(tmp, "JPEG", quality=95)
    os.replace(tmp, out_path)


def doc_to_blocks(doc, images_dir, timeout, stats):
    """Flatten a document into ordered ("image", relpath) / ("text", str) blocks."""
    images = doc.get("images") or []
    texts = doc.get("texts") or []
    hashes = doc.get("image_hashes") or []

    blocks = []
    img_idx = 0
    for url, text in zip_longest(images, texts):
        if url is not None:
            # image_hashes is aligned with the non-null entries of `images`;
            # hashing by content dedups identical images across documents
            if img_idx < len(hashes) and hashes[img_idx]:
                name = hashes[img_idx]
            else:
                name = hashlib.sha256(url.encode()).hexdigest()
            img_idx += 1
            try:
                download_image(url, images_dir / f"{name}.jpg", timeout)
                blocks.append(("image", f"images/{name}.jpg"))
                stats["images_ok"] += 1
            except Exception:
                stats["images_failed"] += 1
        if text:
            text = text.replace("﻿", "").strip()
            if text:
                blocks.append(("text", text))
    return blocks


def split_long_text(text, max_words):
    if len(text.split()) <= max_words:
        return [text]
    parts, current, count = [], [], 0
    for para in text.split("\n\n"):
        n = len(para.split())
        if current and count + n > max_words:
            parts.append("\n\n".join(current))
            current, count = [], 0
        if n > max_words:
            words = para.split()
            for i in range(0, len(words), max_words):
                parts.append(" ".join(words[i : i + max_words]))
        else:
            current.append(para)
            count += n
    if current:
        parts.append("\n\n".join(current))
    return parts


def blocks_to_samples(blocks, max_words, keep_text_only=False, image_word_cost=IMAGE_WORD_COST):
    """Chunk blocks at block boundaries so each sample fits the context budget."""
    normalized = []
    for kind, value in blocks:
        if kind == "text":
            normalized.extend(("text", piece) for piece in split_long_text(value, max_words))
        else:
            normalized.append((kind, value))

    chunks, current, count = [], [], 0
    for kind, value in normalized:
        cost = image_word_cost if kind == "image" else len(value.split())
        if current and count + cost > max_words:
            chunks.append(current)
            current, count = [], 0
        current.append((kind, value))
        count += cost
    if current:
        chunks.append(current)

    samples = []
    for chunk in chunks:
        if not any(kind == "text" for kind, _ in chunk):
            continue
        image_paths = [value for kind, value in chunk if kind == "image"]
        # Text-only samples desync ZeRO-3: ranks whose batch has no images skip
        # the vision tower and miss its parameter all-gathers, hanging NCCL.
        if not image_paths and not keep_text_only:
            continue
        value = "\n\n".join(
            "<image>" if kind == "image" else text for kind, text in chunk
        )
        sample = {"conversations": [{"from": "human", "value": value}]}
        if image_paths:
            sample["image"] = image_paths
        samples.append(sample)
    return samples


def process_doc(doc, images_dir, timeout, max_words, keep_text_only=False, image_word_cost=IMAGE_WORD_COST):
    stats = {"images_ok": 0, "images_failed": 0}
    blocks = doc_to_blocks(doc, images_dir, timeout, stats)
    return blocks_to_samples(blocks, max_words, keep_text_only, image_word_cost), stats


def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-files",
        default="./mint1t/data_v1_1/*.parquet",
        help="Glob of input shards (.parquet, .jsonl or .json)",
    )
    parser.add_argument("--output-dir", default="./mint1t/processed")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--timeout",
        type=float,
        default=3,
        help="Per-image download timeout (s); unresponsive hosts dominate wall "
        "time, so keep this low on a good network",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=5000,
        help="Word budget per output sample; keep it below model_max_length "
        "after tokenization",
    )
    parser.add_argument(
        "--image-word-cost",
        type=int,
        default=IMAGE_WORD_COST,
        help="Word-budget cost per image when chunking; size it to the training "
        "max_pixels (Qwen3-VL: max_pixels/1024 tokens/image, ~0.77 words/token)",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Stop after N documents (smoke tests)")
    parser.add_argument(
        "--keep-text-only",
        action="store_true",
        help="Keep samples without images (unsafe with DeepSpeed ZeRO-3: text-only "
        "batches skip the vision tower and hang its parameter all-gathers)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = output_dir / "annotations.jsonl"

    docs = iter_documents(args.data_files)
    total_docs = count_documents(args.data_files)
    if args.max_docs:
        docs = itertools.islice(docs, args.max_docs)
        if total_docs is not None:
            total_docs = min(total_docs, args.max_docs)

    progress = tqdm(total=total_docs, unit="doc", dynamic_ncols=True) if tqdm else None
    totals = {"docs": 0, "samples": 0, "images_ok": 0, "images_failed": 0}
    with open(annotation_path, "w") as fout, ThreadPoolExecutor(args.num_workers) as pool:
        for batch in batched(docs, args.num_workers * 8):
            results = pool.map(
                lambda d: process_doc(
                    d,
                    images_dir,
                    args.timeout,
                    args.max_words,
                    args.keep_text_only,
                    args.image_word_cost,
                ),
                batch,
            )
            for samples, stats in results:
                totals["docs"] += 1
                totals["samples"] += len(samples)
                totals["images_ok"] += stats["images_ok"]
                totals["images_failed"] += stats["images_failed"]
                for sample in samples:
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if progress:
                    progress.update(1)
                    progress.set_postfix(
                        samples=totals["samples"],
                        img_ok=totals["images_ok"],
                        img_fail=totals["images_failed"],
                    )
            if not progress:
                of_total = f"/{total_docs}" if total_docs is not None else ""
                print(
                    f"docs={totals['docs']}{of_total} samples={totals['samples']} "
                    f"images ok={totals['images_ok']} failed={totals['images_failed']}",
                    flush=True,
                )
    if progress:
        progress.close()

    print(f"\nWrote {totals['samples']} samples to {annotation_path}")
    print(f"Images in {images_dir}: {totals['images_ok']} ok, {totals['images_failed']} failed/skipped")
    extra_flags = " --allow_text_only True" if args.keep_text_only else ""
    print(
        "\nRegister in qwenvl/data/__init__.py as:\n"
        f'  MINT1T = {{"annotation_path": "{annotation_path}", "data_path": "{output_dir}"}}\n'
        f"and train with --dataset_use mint1t --train_on_all_tokens True{extra_flags}"
    )


if __name__ == "__main__":
    main()
