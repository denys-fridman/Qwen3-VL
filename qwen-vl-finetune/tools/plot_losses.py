#!/usr/bin/env python3
"""Plot training curves from cpt_32b Slurm logs as one 2x2 figure:
training loss, evaluation loss, learning rate, gradient norm.

Parses the HF Trainer log dicts printed by rank 0 ({'loss', 'grad_norm',
'learning_rate', 'epoch'} every optimizer step, {'eval_loss', ...} every eval)
and the run-config banner (seed, tune flags) from each slurm_*.out in a
directory. One line per run, labeled by seed; the title carries the mode.

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
SURFACE, INK, INK_MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

LOG_DICT = re.compile(r"\{'[a-z_]+':.*?\}")
TRAIN_KEYS = ("step", "loss", "grad_norm", "learning_rate", "epoch")


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
            if "loss" in rec:
                train.append({k: float(rec[k]) for k in TRAIN_KEYS if k in rec})
            elif "eval_loss" in rec:
                evals.append({k: float(rec[k]) for k in ("step", "eval_loss", "epoch") if k in rec})
    return seed, mode, train, evals


def train_steps(train):
    if all("step" in r for r in train):
        return [int(r["step"]) for r in train]
    return list(range(1, len(train) + 1))


def eval_steps(train, evals):
    """Prefer the logged step; older logs carry only epoch, in which case the
    step of an eval is the number of training records at or before its epoch
    (training logs once per step)."""
    if all("step" in e for e in evals):
        return [int(e["step"]) for e in evals]
    epochs = [r["epoch"] for r in train]
    return [bisect_right(epochs, e["epoch"] + 1e-9) for e in evals]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=11, pad=8)
    ax.set_xlabel("optimizer step", color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", help="Directory containing slurm_*.out logs")
    parser.add_argument("--out-dir", default=None, help="Where to write the PNG (default: log_dir)")
    parser.add_argument("--mode", default=None, help="Override the mode shown in the title (full|llm)")
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=SURFACE)
    (ax_train, ax_eval), (ax_lr, ax_gn) = axes
    for i, (seed, _, train, evals) in enumerate(runs):
        color = PALETTE[i % len(PALETTE)]
        dash = "-" if i < len(PALETTE) else "--"
        label = f"{seed}" + ("" if len(train) > 30 else f" ({len(train)} steps)")
        steps = train_steps(train)
        line = dict(color=color, linewidth=1.5, linestyle=dash)

        ax_train.plot(steps, [r["loss"] for r in train], label=label, **line)
        if evals:
            ax_eval.plot(
                eval_steps(train, evals),
                [r["eval_loss"] for r in evals],
                marker="o",
                markersize=3.5,
                **line,
            )
        if all("learning_rate" in r for r in train):
            ax_lr.plot(steps, [r["learning_rate"] for r in train], **line)
        if all("grad_norm" in r for r in train):
            ax_gn.plot(steps, [r["grad_norm"] for r in train], **line)

    style(ax_train, "Training loss", "loss (per-token CE)")
    style(ax_eval, "Evaluation loss", "eval loss (per-token CE)")
    style(ax_lr, "Learning rate", "lr")
    style(ax_gn, "Gradient norm", "grad norm (pre-clip)")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.suptitle(f"mode: {mode} — {len(runs)} runs", color=INK, fontsize=13, x=0.01, ha="left")
    handles, labels = ax_train.get_legend_handles_labels()
    fig.legend(
        handles, labels, title="seed", loc="lower center", ncol=min(len(labels), 10),
        frameon=False, fontsize=9, title_fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
