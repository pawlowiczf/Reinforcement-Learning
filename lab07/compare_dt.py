"""
Compare each trained Decision Transformer against the PPO expert that generated
its training data, and against the demonstrations themselves.

For every level it reports the mean return (+/- std) over N fresh episodes for:
  - Dataset : the PPO demonstrations the DT learned from (offline data quality)
  - PPO     : the original online expert, re-run live (the "teacher")
  - DT      : the Decision Transformer we trained (the "student")

DT and PPO use the SAME per-episode seeds, so the comparison is apples-to-apples.
A grouped bar chart is saved to output/compare_returns.png.

    uv run python compare_dt.py                       # all levels, 20 episodes
    uv run python compare_dt.py --episodes 50
    uv run python compare_dt.py --levels expert       # just one level
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

import minari
from rich.console import Console
from rich.table import Table

from dt_eval import (
    LEVELS,
    SOLVED_THRESHOLD,
    dataset_returns,
    evaluate_dt,
    evaluate_ppo,
)

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Compare DT vs PPO expert vs dataset")
    p.add_argument(
        "--levels",
        nargs="+",
        default=list(LEVELS),
        choices=list(LEVELS),
        help="Which skill levels to compare (default: all)",
    )
    p.add_argument(
        "--episodes", type=int, default=20, help="Eval episodes per model (default: 20)"
    )
    p.add_argument("--seed", type=int, default=0, help="Base seed for eval episodes")
    p.add_argument(
        "--out", default="output/compare_returns.png", help="Where to save the chart"
    )
    return p.parse_args()


def summarize(returns):
    return {
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "solved_pct": float((returns >= SOLVED_THRESHOLD).mean() * 100),
    }


def main():
    args = parse_args()
    results = {}

    for level in args.levels:
        console.rule(f"[bold]{level}")
        # Dataset quality: the demonstrations the DT was trained on.
        ds = minari.load_dataset(LEVELS[level]["dataset"])
        ds_ret = dataset_returns(ds)

        console.print(f"Evaluating PPO expert over {args.episodes} episodes...")
        ppo_ret, _ = evaluate_ppo(level, args.episodes, base_seed=args.seed)
        console.print(f"Evaluating Decision Transformer over {args.episodes} episodes...")
        dt_ret, _ = evaluate_dt(level, args.episodes, base_seed=args.seed)

        results[level] = {
            "dataset": summarize(ds_ret),
            "ppo": summarize(ppo_ret),
            "dt": summarize(dt_ret),
        }

    print_table(results)
    plot(results, args.out)

    json_path = Path(args.out).with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2))
    console.print(f"\n[green]Saved chart   ->[/green] {args.out}")
    console.print(f"[green]Saved numbers ->[/green] {json_path}")


def print_table(results):
    table = Table(title="Mean return (+/- std) over eval episodes")
    table.add_column("Level", style="bold")
    table.add_column("Dataset (demos)", justify="right")
    table.add_column("PPO expert", justify="right")
    table.add_column("DT (ours)", justify="right")
    table.add_column("DT vs PPO", justify="right")
    table.add_column("DT solved%", justify="right")

    for level, r in results.items():
        diff = r["dt"]["mean"] - r["ppo"]["mean"]
        color = "green" if diff >= -5 else "yellow" if diff >= -20 else "red"
        table.add_row(
            level,
            f"{r['dataset']['mean']:.0f} ± {r['dataset']['std']:.0f}",
            f"{r['ppo']['mean']:.0f} ± {r['ppo']['std']:.0f}",
            f"{r['dt']['mean']:.0f} ± {r['dt']['std']:.0f}",
            f"[{color}]{diff:+.0f}[/{color}]",
            f"{r['dt']['solved_pct']:.0f}%",
        )
    console.print(table)


def plot(results, out_path):
    levels = list(results)
    groups = [
        ("Dataset (demos)", "dataset", "#9aa0a6"),
        ("PPO expert", "ppo", "#4285f4"),
        ("DT (ours)", "dt", "#ea4335"),
    ]
    x = np.arange(len(levels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (label, key, color) in enumerate(groups):
        means = [results[lv][key]["mean"] for lv in levels]
        stds = [results[lv][key]["std"] for lv in levels]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, means, width, yerr=stds, capsize=4, label=label, color=color
        )
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)

    ax.axhline(
        SOLVED_THRESHOLD,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"solved (≥{SOLVED_THRESHOLD:.0f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Episode return")
    ax.set_title("Decision Transformer vs PPO expert - LunarLanderContinuous-v3")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)


if __name__ == "__main__":
    main()
