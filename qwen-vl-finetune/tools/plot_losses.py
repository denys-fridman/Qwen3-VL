#!/usr/bin/env python3
"""Plot training and evaluation loss curves from cpt_32b Slurm logs.

Parses the HF Trainer log dicts printed by rank 0 ({'loss': ...} every
optimizer step, {'eval_loss': ...} every eval) and the run-config banner
(seed, tune flags) from each slurm_*.out in a directory, then writes two
figures — one for training loss, one for evaluation loss — with one line per
run, labeled by seed.

Usage:
    python tools/plot_losses.py ../logs/qwen/full [--out-dir DIR] [--mode full|llm]
"""

import argparse
import ast
import glob
import os
import re
from bisect import bisect_right

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed-order categorical palette (colorblind-validated adjacent pairs). Runs
# beyond eight slots reuse the hues with a dashed line so identity never rests
# on color alone.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

LOG_DICT = re.compile(r"\{'(?:loss|eval_loss)':.*?\}")


def parse_log(path):
    seed, mode, train, evals = None, None, [], []
    with open(path, errors="replace") as f:
        for line in f:
            if seed is None:
                m = re.search(r"\bseed=(\d+)", line)
                if m:
                    seed = int(m.group(1))
            if mode is None:
                m = re.search(r"tune_mm_vision=(True|False)", line)
                if m:
                    mode = "full" if m.group(1) == "True" else "llm"
            m = LOG_DICT.search(line)
            if not m:
                continue
            try:
                rec = ast.literal_eval(m.group(0))
            except (ValueError, SyntaxError):
                continue
            rec = {k: float(v) for k, v in rec.items() if k in ("loss", "eval_loss", "epoch")}
            if "loss" in rec:
                train.append(rec)
            elif "eval_loss" in rec:
                evals.append(rec)
    return seed, mode, train, evals


def eval_steps_from_epochs(train, evals):
    """Logs carry epoch but not step; training logs once per step, so the step
    of an eval is the number of training records at or before its epoch."""
    epochs = [r["epoch"] for r in train]
    return [bisect_right(epochs, e["epoch"] + 1e-9) for e in evals]


def style(ax, title, ylabel):
    ax.set_title(title, color=INK, loc="left", fontsize=12, pad=10)
    ax.set_xlabel("optimizer step", color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.legend(title="seed", frameon=False, fontsize=9, title_fontsize=9, ncol=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", help="Directory containing slurm_*.out logs")
    parser.add_argument("--out-dir", default=None, help="Where to write PNGs (default: log_dir)")
    parser.add_argument("--mode", default=None, help="Override the mode shown in titles (full|llm)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.log_dir
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    for path in sorted(glob.glob(os.path.join(args.log_dir, "slurm_*.out"))):
        seed, mode, train, evals = parse_log(path)
        if not train:
            print(f"skipping {os.path.basename(path)}: no training records")
            continue
        runs.append((seed if seed is not None else os.path.basename(path), mode, train, evals))
    if not runs:
        raise SystemExit(f"no parsable logs in {args.log_dir}")

    runs.sort(key=lambda r: (isinstance(r[0], str), r[0]))
    mode = args.mode or next((m for _, m, _, _ in runs if m), None) or os.path.basename(
        os.path.normpath(args.log_dir)
    )

    fig_train, ax_train = plt.subplots(figsize=(9, 5), facecolor="#fcfcfb")
    fig_eval, ax_eval = plt.subplots(figsize=(9, 5), facecolor="#fcfcfb")
    for i, (seed, _, train, evals) in enumerate(runs):
        color = PALETTE[i % len(PALETTE)]
        dash = "-" if i < len(PALETTE) else "--"
        label = f"{seed}" + ("" if len(train) > 30 else f" ({len(train)} steps)")
        steps = range(1, len(train) + 1)
        ax_train.plot(steps, [r["loss"] for r in train], dash, color=color, linewidth=1.6, label=label)
        if evals:
            ax_eval.plot(
                eval_steps_from_epochs(train, evals),
                [r["eval_loss"] for r in evals],
                dash,
                color=color,
                linewidth=1.6,
                marker="o",
                markersize=4,
                label=label,
            )

    style(ax_train, f"Training loss — mode: {mode} ({len(runs)} runs)", "loss (per-token CE)")
    style(ax_eval, f"Evaluation loss — mode: {mode} ({len(runs)} runs)", "eval loss (per-token CE)")
    for fig, ax, name in ((fig_train, ax_train, "train_loss.png"), (fig_eval, ax_eval, "eval_loss.png")):
        ax.set_facecolor("#fcfcfb")
        fig.tight_layout()
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
