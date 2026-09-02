#!/usr/bin/env python3
"""Measure run-to-run variance in samples-to-target-loss across seed runs.

Adapted from the MLPerf DeepSeek-V3 find_loss_variance.py for the Qwen3-VL
continued-pretraining logs produced by scripts/measure_variance.sh.

Each *.out log under the experiments directory is one experiment. Eval
records are the HF Trainer dicts printed by rank 0:

    {'step': 20, 'samples': 20480, 'eval_loss': '2.641', ..., 'epoch': '0.0627'}

The x-axis is samples processed = step * BATCH_SIZE. For older logs without a
'step' key the step is inferred from the epoch of the preceding training
records (training logs once per optimizer step).

Usage:
    python tools/measure_coefficient_of_variance.py <experiments_dir> --target-loss 2.5
    python tools/measure_coefficient_of_variance.py <experiments_dir> \
        --sweep-range 2.45 2.85 0.05 --pivot-table --plot --plot-output cov.png
"""

import argparse
import ast
import os
import re
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BATCH_SIZE = 1024  # global batch: 2 per device x 8 grad accumulation x 64 GPUs

LOG_DICT = re.compile(r"\{'[a-z_]+':.*?\}")


def parse_log_dicts(file_path: Path) -> Tuple[List[dict], List[dict]]:
    """Return (train_records, eval_records) parsed from a log file."""
    train, evals = [], []
    with open(file_path, "r", errors="replace") as f:
        for line in f:
            m = LOG_DICT.search(line)
            if not m:
                continue
            try:
                rec = ast.literal_eval(m.group(0))
            except (ValueError, SyntaxError):
                continue
            if "eval_loss" in rec:
                evals.append(rec)
            elif "loss" in rec:
                train.append(rec)
    return train, evals


def eval_steps(train: List[dict], evals: List[dict]) -> List[int]:
    """Logged step when present; otherwise the number of training records at or
    before the eval's epoch."""
    if evals and all("step" in e for e in evals):
        return [int(e["step"]) for e in evals]
    epochs = [float(r["epoch"]) for r in train if "epoch" in r]
    return [bisect_right(epochs, float(e["epoch"]) + 1e-9) for e in evals]


def process_experiment(file_path: Path) -> Dict[int, float]:
    """Process a single experiment log. Returns samples_count -> eval loss."""
    step_loss_data = {}
    try:
        train, evals = parse_log_dicts(file_path)
        for step, rec in zip(eval_steps(train, evals), evals):
            step_loss_data[step * BATCH_SIZE] = float(rec["eval_loss"])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return step_loss_data


def find_first_step_to_reach_loss(step_loss_data: Dict[int, float], target_loss: float) -> Optional[int]:
    """First samples count where the loss is <= target_loss, or None."""
    for step in sorted(step_loss_data.keys()):
        if step_loss_data[step] <= target_loss:
            return step
    return None


def process_experiments_directory(experiments_dir: str) -> pd.DataFrame:
    """Collect eval points from every *.out log under experiments_dir."""
    experiments_path = Path(experiments_dir)
    if not experiments_path.exists():
        raise ValueError(f"Directory {experiments_dir} does not exist")

    all_data = []
    for log_file in sorted(experiments_path.rglob("*.out")):
        print(f"\nProcessing experiment: {log_file}")
        step_loss_data = process_experiment(log_file)
        for step, loss in step_loss_data.items():
            all_data.append({"experiment": log_file.stem, "step": step, "loss": loss})
        print(f"Found {len(step_loss_data)} eval points for {log_file}")

    if not all_data:
        print("No eval data found in any experiments")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    return df.sort_values(["experiment", "step"]).reset_index(drop=True)


def analyze_steps_to_target_loss(df: pd.DataFrame, target_loss: float, verbose: bool = True) -> Optional[pd.DataFrame]:
    """Variance in samples required to reach the target loss across experiments."""
    if df.empty:
        if verbose:
            print("No data to analyze")
        return None

    if verbose:
        print("\n" + "=" * 80)
        print(f"ANALYSIS: Samples to first reach loss <= {target_loss}")
        print("=" * 80)

    steps_to_target = []
    experiment_results = []
    for exp_name in df["experiment"].unique():
        exp_data = df[df["experiment"] == exp_name]
        step_loss_dict = dict(zip(exp_data["step"], exp_data["loss"]))
        first_step = find_first_step_to_reach_loss(step_loss_dict, target_loss)
        if first_step is not None:
            steps_to_target.append(first_step)
            experiment_results.append(
                {"experiment": exp_name, "first_step_to_target": first_step, "loss_at_step": step_loss_dict[first_step]}
            )
            if verbose:
                print(f"{exp_name}: reached loss <= {target_loss} at {first_step} samples (actual loss: {step_loss_dict[first_step]:.4f})")
        else:
            experiment_results.append({"experiment": exp_name, "first_step_to_target": None, "loss_at_step": None})
            if verbose:
                print(f"{exp_name}: NEVER reached target loss (minimum loss achieved: {min(step_loss_dict.values()):.4f})")

    if not steps_to_target:
        if verbose:
            print(f"\nNo experiments reached the target loss of {target_loss}")
        return None

    steps_array = pd.Series(steps_to_target)
    if verbose:
        print("\n" + "-" * 50)
        print("VARIANCE ANALYSIS")
        print("-" * 50)
        print(f"Experiments that reached target loss: {len(steps_to_target)} out of {df['experiment'].nunique()}")
        print(f"Mean samples to reach target: {steps_array.mean():.1f}")
        print(f"Standard deviation: {steps_array.std():.1f}")
        print(f"Minimum samples: {steps_array.min()}")
        print(f"Maximum samples: {steps_array.max()}")
        print(f"Range: {steps_array.max() - steps_array.min()} samples")
        print(f"Coefficient of variation: {(steps_array.std() / steps_array.mean() * 100):.1f}%")
        print("\nPercentiles:")
        print(f"25th percentile: {steps_array.quantile(0.25):.1f}")
        print(f"50th percentile (median): {steps_array.quantile(0.5):.1f}")
        print(f"75th percentile: {steps_array.quantile(0.75):.1f}")

    return pd.DataFrame(experiment_results)


def sweep_loss_range(df: pd.DataFrame, min_loss: float, max_loss: float, interval: float) -> pd.DataFrame:
    """Analyze variance for each target loss in a range."""
    loss_values = np.arange(min_loss, max_loss + interval, interval)

    print("\n" + "=" * 80)
    print(f"LOSS SWEEP ANALYSIS: {min_loss} to {max_loss} (interval: {interval})")
    print("=" * 80)

    n_experiments = len(df["experiment"].unique())
    sweep_results = []
    for target_loss in loss_values:
        print(f"\nAnalyzing target loss: {target_loss:.3f}")
        print("-" * 40)
        results_df = analyze_steps_to_target_loss(df, target_loss, verbose=False)
        successful = (
            results_df[results_df["first_step_to_target"].notna()] if results_df is not None else pd.DataFrame()
        )
        steps = successful["first_step_to_target"] if len(successful) > 0 else None
        sweep_results.append(
            {
                "target_loss": target_loss,
                "experiments_reached": len(successful),
                "total_experiments": n_experiments,
                "success_rate": len(successful) / n_experiments,
                "mean_steps": steps.mean() if steps is not None else None,
                "std_steps": steps.std() if steps is not None else None,
                "min_steps": steps.min() if steps is not None else None,
                "max_steps": steps.max() if steps is not None else None,
                "range_steps": (steps.max() - steps.min()) if steps is not None else None,
                "cv_percent": (steps.std() / steps.mean() * 100) if steps is not None else None,
                "median_steps": steps.median() if steps is not None else None,
                "q25_steps": steps.quantile(0.25) if steps is not None else None,
                "q75_steps": steps.quantile(0.75) if steps is not None else None,
            }
        )
        if steps is not None:
            r = sweep_results[-1]
            print(f"  Reached by {len(successful)}/{n_experiments} experiments")
            print(f"  Mean samples: {r['mean_steps']:.1f} ± {r['std_steps']:.1f}")
            print(f"  Range: {r['min_steps']} - {r['max_steps']} samples")
            print(f"  CV: {r['cv_percent']:.1f}%")
        else:
            print(f"  No experiments reached target loss {target_loss:.3f}")

    sweep_df = pd.DataFrame(sweep_results)

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    print(f"Loss values tested: {len(loss_values)}")
    successful_df = sweep_df[sweep_df["experiments_reached"] > 0]
    print(f"Loss values with successful experiments: {len(successful_df)}")
    if len(successful_df) > 0:
        print("\nVariance trends:")
        print(f"Lowest CV: {successful_df['cv_percent'].min():.1f}% at loss {successful_df.loc[successful_df['cv_percent'].idxmin(), 'target_loss']:.3f}")
        print(f"Highest CV: {successful_df['cv_percent'].max():.1f}% at loss {successful_df.loc[successful_df['cv_percent'].idxmax(), 'target_loss']:.3f}")
        print(f"\nSuccess rate range: {sweep_df['success_rate'].min():.1%} - {sweep_df['success_rate'].max():.1%}")
        print(f"Sample range across all targets: {successful_df['min_steps'].min():.0f} - {successful_df['max_steps'].max():.0f}")

    return sweep_df


def plot_sweep_results(sweep_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Target loss vs mean samples to reach it, with a ±1 std band."""
    if sweep_df.empty:
        print("No data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    successful_df = sweep_df[sweep_df["experiments_reached"] > 0].copy()
    unsuccessful_df = sweep_df[sweep_df["experiments_reached"] == 0].copy()

    if not successful_df.empty:
        successful_df = successful_df.sort_values("mean_steps")
        ax.plot(
            successful_df["mean_steps"], successful_df["target_loss"],
            color="steelblue", linewidth=2.5, marker="o", markersize=6, alpha=0.8,
            label=f"Reached target loss ({len(successful_df)} values)",
        )
        ax.fill_betweenx(
            successful_df["target_loss"],
            successful_df["mean_steps"] - successful_df["std_steps"],
            successful_df["mean_steps"] + successful_df["std_steps"],
            color="steelblue", alpha=0.2, label="± 1 standard deviation",
        )

    if not unsuccessful_df.empty:
        min_x = successful_df["mean_steps"].min() * 0.8 if not successful_df.empty else 0
        ax.scatter(
            [min_x] * len(unsuccessful_df), unsuccessful_df["target_loss"],
            marker="x", s=100, color="red", alpha=0.7,
            label=f"Never reached ({len(unsuccessful_df)} values)",
        )
        if not successful_df.empty:
            ax.axvline(x=min_x * 1.1, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Samples to reach target (mean ± std)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Target loss", fontsize=14, fontweight="bold")
    ax.set_title("Convergence variance: target loss vs samples required", fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=12)

    if not successful_df.empty:
        min_cv_idx = successful_df["cv_percent"].idxmin()
        max_cv_idx = successful_df["cv_percent"].idxmax()
        ax.annotate(
            f"Lowest CV: {successful_df.loc[min_cv_idx, 'cv_percent']:.1f}%",
            xy=(successful_df.loc[min_cv_idx, "mean_steps"], successful_df.loc[min_cv_idx, "target_loss"]),
            xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            arrowprops=dict(arrowstyle="->"), fontsize=10,
        )
        ax.annotate(
            f"Highest CV: {successful_df.loc[max_cv_idx, 'cv_percent']:.1f}%",
            xy=(successful_df.loc[max_cv_idx, "mean_steps"], successful_df.loc[max_cv_idx, "target_loss"]),
            xytext=(10, -20), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            arrowprops=dict(arrowstyle="->"), fontsize=10,
        )
        stats_text = (
            f"Successful targets: {len(successful_df)}/{len(sweep_df)}\n"
            f"Sample range: {successful_df['min_steps'].min():.0f} - {successful_df['max_steps'].max():.0f}\n"
            f"Loss range: {successful_df['target_loss'].min():.3f} - {successful_df['target_loss'].max():.3f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    plt.close(fig)


def create_detailed_sweep_table(df: pd.DataFrame, sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Long-form table: per target loss, per experiment, samples to reach it."""
    detailed_results = []
    for _, row in sweep_df.iterrows():
        target_loss = row["target_loss"]
        for exp_name in df["experiment"].unique():
            exp_data = df[df["experiment"] == exp_name]
            step_loss_dict = dict(zip(exp_data["step"], exp_data["loss"]))
            first_step = find_first_step_to_reach_loss(step_loss_dict, target_loss)
            detailed_results.append(
                {
                    "target_loss": target_loss,
                    "experiment": exp_name,
                    "first_step_to_target": first_step,
                    "reached_target": first_step is not None,
                    "min_loss_achieved": min(step_loss_dict.values()) if step_loss_dict else None,
                }
            )
    return pd.DataFrame(detailed_results)


def create_pivot_table(df: pd.DataFrame, sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Target losses as rows, experiments as columns, plus summary statistics."""
    detailed_df = create_detailed_sweep_table(df, sweep_df)
    pivot_data = detailed_df.pivot(index="target_loss", columns="experiment", values="first_step_to_target")
    experiments = df["experiment"].unique()

    summary_stats = []
    for target_loss in pivot_data.index:
        steps_for_loss = [
            pivot_data.loc[target_loss, exp]
            for exp in experiments
            if exp in pivot_data.columns and pd.notna(pivot_data.loc[target_loss, exp])
        ]
        if steps_for_loss:
            avg_steps = np.mean(steps_for_loss)
            std_steps = np.std(steps_for_loss, ddof=1) if len(steps_for_loss) > 1 else 0
            summary_stats.append(
                {
                    "target_loss": target_loss,
                    "experiments_reached": len(steps_for_loss),
                    "avg_steps": avg_steps,
                    "std_steps": std_steps,
                    "cv_percent": (std_steps / avg_steps * 100) if avg_steps > 0 else 0,
                    "min_steps": min(steps_for_loss),
                    "max_steps": max(steps_for_loss),
                }
            )
        else:
            summary_stats.append(
                {
                    "target_loss": target_loss,
                    "experiments_reached": 0,
                    "avg_steps": None,
                    "std_steps": None,
                    "cv_percent": None,
                    "min_steps": None,
                    "max_steps": None,
                }
            )

    summary_df = pd.DataFrame(summary_stats).set_index("target_loss")
    result_df = pd.concat([pivot_data, summary_df], axis=1)
    exp_columns = [col for col in result_df.columns if col in experiments]
    summary_columns = ["experiments_reached", "avg_steps", "std_steps", "cv_percent", "min_steps", "max_steps"]
    return result_df[exp_columns + summary_columns]


def print_pivot_table(pivot_df: pd.DataFrame) -> None:
    """Print the summary-statistics part of the pivot table."""
    if pivot_df.empty:
        print("No data to display in pivot table")
        return

    print("\n" + "=" * 80)
    print("PIVOT TABLE: Target Loss Statistics (samples)")
    print("=" * 80)

    summary_columns = ["experiments_reached", "avg_steps", "std_steps", "cv_percent", "min_steps", "max_steps"]
    display_df = pivot_df[summary_columns].copy()
    display_df["avg_steps"] = display_df["avg_steps"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    display_df["std_steps"] = display_df["std_steps"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    display_df["cv_percent"] = display_df["cv_percent"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
    display_df["min_steps"] = display_df["min_steps"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
    display_df["max_steps"] = display_df["max_steps"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
    display_df = display_df.rename(
        columns={
            "experiments_reached": "Reached",
            "avg_steps": "Avg Samples",
            "std_steps": "Std Dev",
            "cv_percent": "CV%",
            "min_steps": "Min Samples",
            "max_steps": "Max Samples",
        }
    )
    display_df.index.name = "Target Loss"
    print(display_df.to_string(max_rows=None, max_cols=None))

    successful_rows = pivot_df[pivot_df["experiments_reached"] > 0]
    if not successful_rows.empty:
        total_experiments = len([col for col in pivot_df.columns if col not in summary_columns])
        print("\n" + "-" * 50)
        print("SUMMARY STATISTICS")
        print("-" * 50)
        print(f"Total experiments: {total_experiments}")
        print(f"Target loss values tested: {len(pivot_df)}")
        print(f"Target loss values reached by at least one experiment: {len(successful_rows)}")
        print(f"Average CV across all reachable targets: {successful_rows['cv_percent'].mean():.1f}%")
        print(f"Target with lowest CV: {successful_rows['cv_percent'].idxmin():.3f} (CV: {successful_rows['cv_percent'].min():.1f}%)")
        print(f"Target with highest CV: {successful_rows['cv_percent'].idxmax():.3f} (CV: {successful_rows['cv_percent'].max():.1f}%)")
        success_rates = successful_rows["experiments_reached"] / total_experiments
        print(f"Success rate range: {success_rates.min():.1%} - {success_rates.max():.1%}")
        print(f"Average success rate: {success_rates.mean():.1%}")


def analyze_loss_variance(df: pd.DataFrame) -> None:
    """General eval-loss statistics per experiment and across experiments."""
    if df.empty:
        print("No data to analyze")
        return

    print("\n" + "=" * 60)
    print("GENERAL LOSS ANALYSIS")
    print("=" * 60)
    print("\nPer-experiment statistics:")
    print(df.groupby("experiment")["loss"].agg(["count", "min", "max", "mean", "std"]).round(4))

    if df["experiment"].nunique() > 1:
        print("\nVariance across experiments at each samples count:")
        print(df.groupby("step")["loss"].agg(["count", "mean", "std", "min", "max"]).round(4).head(10))
        print("\nOverall loss variance across all experiments:")
        print(f"Mean loss: {df['loss'].mean():.4f}")
        print(f"Standard deviation: {df['loss'].std():.4f}")
        print(f"Min loss: {df['loss'].min():.4f}")
        print(f"Max loss: {df['loss'].max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze variance in samples to reach target eval loss across seed runs")
    parser.add_argument("experiments_dir", help="Directory containing *.out logs (searched recursively)")
    loss_group = parser.add_mutually_exclusive_group(required=True)
    loss_group.add_argument("--target-loss", type=float, help="Single target loss value to analyze")
    loss_group.add_argument("--sweep-range", nargs=3, type=float, metavar=("MIN", "MAX", "INTERVAL"),
                            help="Sweep loss range: min_loss max_loss interval")
    parser.add_argument("--output", "-o", help="Output CSV file (relative to experiments_dir)")
    parser.add_argument("--general-analysis", "-g", action="store_true", help="Also print general loss analysis")
    parser.add_argument("--plot", "-p", action="store_true", help="Create plot for sweep results (requires --sweep-range)")
    parser.add_argument("--plot-output", default="coefficient_of_variance.png",
                        help="Plot image path relative to experiments_dir (requires --plot)")
    parser.add_argument("--detailed-table", "-d", help="Save detailed long-form table for sweep results (CSV)")
    parser.add_argument("--pivot-table", "-t", action="store_true", help="Show pivot table (requires --sweep-range)")
    parser.add_argument("--pivot-output", help="Save pivot table to CSV (requires --pivot-table)")
    args = parser.parse_args()

    if args.plot and args.target_loss is not None:
        print("Warning: --plot only works with --sweep-range, ignoring plot option")
        args.plot = False
    if args.pivot_table and args.target_loss is not None:
        print("Warning: --pivot-table only works with --sweep-range, ignoring pivot table option")
        args.pivot_table = False
    if args.pivot_output and not args.pivot_table:
        print("Warning: --pivot-output requires --pivot-table, ignoring pivot output option")
        args.pivot_output = None

    df = process_experiments_directory(args.experiments_dir)
    if df.empty:
        print("No eval data found")
        return 1

    results_df = None
    if args.target_loss is not None:
        results_df = analyze_steps_to_target_loss(df, args.target_loss)
    else:
        min_loss, max_loss, interval = args.sweep_range
        results_df = sweep_loss_range(df, min_loss, max_loss, interval)
        if args.plot and not results_df.empty:
            plot_sweep_results(results_df, os.path.join(args.experiments_dir, args.plot_output))
        if args.detailed_table and not results_df.empty:
            create_detailed_sweep_table(df, results_df).to_csv(
                os.path.join(args.experiments_dir, args.detailed_table), index=False
            )
            print(f"Detailed long-form table saved to {args.detailed_table}")
        if args.pivot_table and not results_df.empty:
            pivot_df = create_pivot_table(df, results_df)
            print_pivot_table(pivot_df)
            if args.pivot_output:
                pivot_df.to_csv(os.path.join(args.experiments_dir, args.pivot_output), index=True)
                print(f"Pivot table saved to {args.pivot_output}")

    if args.output and results_df is not None and not results_df.empty:
        results_df.to_csv(os.path.join(args.experiments_dir, args.output), index=False)
        print(f"\nResults saved to {args.output}")

    if args.general_analysis:
        analyze_loss_variance(df)

    print("\nDataset summary:")
    print(f"Total eval points collected: {len(df)}")
    print(f"Experiments processed: {df['experiment'].nunique()}")
    print(f"Samples range: {df['step'].min()} - {df['step'].max()}")
    print(f"Loss range: {df['loss'].min():.4f} - {df['loss'].max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
