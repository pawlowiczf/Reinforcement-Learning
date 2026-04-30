"""
Parametric study of multi-armed bandit algorithms, in the style of
Sutton & Barto Fig. 2.6.

For each (algorithm, parameter_value, seed) job we build a fresh K-armed
TopHitBandit (random Bernoulli probabilities) and run TIME_STEPS rounds.
The reported metric is the mean reward per step, averaged across seeds.

Each algorithm has its own scalar parameter (eps, c, alpha, m), but the
parameters all control roughly the same trade-off — exploration vs.
exploitation — so it is meaningful to put them on a shared log-scale x-axis
and compare curves directly.

Speed:

1. (algo, param, seed) jobs are independent, so we farm them out to a
   `multiprocessing.Pool`. On Slurm we honour `SLURM_CPUS_PER_TASK`.
2. Headless matplotlib (Agg) — no GUI window per worker.

Raw per-job results are appended to a CSV so a partial run is never lost.
Re-running with the same CSV path skips already-completed
(algo, param_value, seed) triples.
"""

from __future__ import annotations

import csv
import itertools
import multiprocessing as mp
import os
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no GUI window per worker
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- configuration
TIME_STEPS = 1000
NUM_ARMS = 10                 # Sutton & Barto-style 10-armed testbed
SEEDS = list(range(50))       # bump to 200+ for publication-quality curves

# Parameter grids per algorithm.
# `np.logspace(a, b, num, base=2)` gives `num` points geometrically spaced
# in [2^a, 2^b]. Increase `num` to make the curve denser without changing
# the range. With num = (b - a) + 1 you get integer powers of 2 (Sutton &
# Barto Fig. 2.6 default); doubling that gives half-integer powers (~1.41x
# step), tripling gives third-integer powers, etc.
# Tuple format: (algo_key, human_param_name, [values...]).
DENSITY = 2  # 1 = powers of 2 (default), 2 = ~1.41x step, 4 = ~1.19x step
ALGORITHMS: list[tuple[str, str, list[float]]] = [
    ("egreedy",  "eps",   list(np.logspace(-7, -2, num=5 * DENSITY + 1, base=2))),   # 1/128 .. 1/4
    ("ucb",      "c",     list(np.logspace(-4,  2, num=6 * DENSITY + 1, base=2))),   # 1/16  .. 4
    ("gradient", "alpha", list(np.logspace(-5,  2, num=7 * DENSITY + 1, base=2))),   # 1/32  .. 4
    # m is integer-valued, so dedupe after rounding (avoids duplicate jobs at low m).
    ("etc",      "m",     sorted({int(round(v)) for v in np.logspace(0, 7, num=7 * DENSITY + 1, base=2)})),
]

# Pretty names + colors for plotting (kept here so the data layer stays clean).
PLOT_STYLE: dict[str, tuple[str, str]] = {
    "egreedy":  ("e-greedy",            "tab:blue"),
    "ucb":      ("UCB",                 "tab:orange"),
    "gradient": ("gradient bandit",     "tab:green"),
    "etc":      ("explore-then-commit", "tab:red"),
}

OUT_DIR = Path("plots_parameters")
CSV_PATH = OUT_DIR / "parametric_study.csv"
PLOT_PATH = OUT_DIR / "parametric_study.png"

# Cap workers. On Slurm honour the allocation; locally leave one core free.
if "SLURM_CPUS_PER_TASK" in os.environ:
    N_WORKERS = int(os.environ["SLURM_CPUS_PER_TASK"])
else:
    N_WORKERS = max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------- logging helper

def log(msg: str) -> None:
    """Print with timestamp and flush immediately (so tail -f shows it live)."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------- worker

def _build_learner(algo: str, value: float):
    """Construct a learner instance for the given algorithm key and parameter value."""
    from bandits.egreedy_bandit import EGreedyLearner
    from bandits.explore_then_commit_bandit import ExploreThenCommitLearner
    from bandits.gradient_bandit import GradientLearner
    from bandits.ucb_bandit import UCBLearner

    if algo == "egreedy":
        return EGreedyLearner(eps=value)
    if algo == "ucb":
        return UCBLearner(c=value)
    if algo == "gradient":
        return GradientLearner(alpha=value)
    if algo == "etc":
        return ExploreThenCommitLearner(m=int(value))
    raise ValueError(f"unknown algo key: {algo}")


def _random_potential_hits() -> dict[str, float]:
    """Sample a fresh K-armed Bernoulli bandit from the (already-seeded) RNG."""
    return {f"Arm {i+1}": random.random() for i in range(NUM_ARMS)}


def _run_one(job: tuple[str, str, float, int]) -> tuple[str, str, float, int, float, float]:
    """Run a single (algo, param_name, value, seed) job.

    Returns (algo, param_name, value, seed, mean_reward, seconds).
    """
    algo, param_name, value, seed = job
    pid = os.getpid()
    t0 = time.perf_counter()

    print(
        f"[pid {pid}] START  algo={algo:<9}  {param_name}={value:<8}  seed={seed}",
        flush=True,
    )

    # Per-job determinism. random.seed/np.random.seed cover both the bandit
    # (random_potential_hits + reward draws) and the learner's internal RNG.
    random.seed(seed)
    np.random.seed(seed)

    # Local import: same lazy pattern as lab02 so workers don't pay the cost
    # until they actually start running jobs.
    from environment.environment import KArmedBandit

    class TopHitBandit(KArmedBandit):
        def __init__(self, potential_hits: dict[str, float]):
            self.potential_hits = potential_hits

        def arms(self) -> list[str]:
            return list(self.potential_hits)

        def reward(self, arm: str) -> float:
            return 1.0 if random.random() <= self.potential_hits[arm] else 0.0

    bandit = TopHitBandit(_random_potential_hits())
    learner = _build_learner(algo, value)
    learner.reset(bandit.arms(), TIME_STEPS)

    total_reward = 0.0
    for _ in range(TIME_STEPS):
        arm = learner.pick_arm()
        r = bandit.reward(arm)
        learner.acknowledge_reward(arm, r)
        total_reward += r

    mean_reward = total_reward / TIME_STEPS
    elapsed = time.perf_counter() - t0
    print(
        f"[pid {pid}] DONE   algo={algo:<9}  {param_name}={value:<8}  seed={seed}  "
        f"-> mean_reward={mean_reward:.3f}  ({elapsed:.1f}s)",
        flush=True,
    )
    return algo, param_name, value, seed, mean_reward, elapsed


# ---------------------------------------------------------------- orchestration

def _load_done(csv_path: Path) -> set[tuple[str, float, int]]:
    if not csv_path.exists():
        return set()
    done: set[tuple[str, float, int]] = set()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["algo"], float(row["value"]), int(row["seed"])))
    return done


def _append_row(csv_path: Path, row: tuple[str, str, float, int, float, float]) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["algo", "param_name", "value", "seed", "mean_reward", "seconds"])
        writer.writerow(row)


def _plot(csv_path: Path, plot_path: Path) -> None:
    """Group CSV rows by (algo, value), average over seeds, draw one curve per algo."""
    by_key: dict[tuple[str, str, float], list[float]] = {}
    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["algo"], row["param_name"], float(row["value"]))
            by_key.setdefault(key, []).append(float(row["mean_reward"]))

    plt.figure(figsize=(9, 6))

    algos_in_csv = sorted({a for a, _, _ in by_key})
    for algo in algos_in_csv:
        pretty, color = PLOT_STYLE.get(algo, (algo, None))
        param_name = next(p for a, p, _ in by_key if a == algo)
        values = sorted({v for a, _, v in by_key if a == algo})
        means = [np.mean(by_key[(algo, param_name, v)]) for v in values]
        plt.plot(values, means, marker="o", label=f"{pretty}  ({param_name})", color=color)

    plt.xscale("log", base=2)
    plt.xlabel(r"parameter value (log$_2$ scale)  —  $\varepsilon$ / $c$ / $\alpha$ / $m$")
    plt.ylabel(f"mean reward per step  (avg over {TIME_STEPS} steps, {len(SEEDS)} seeds)")
    plt.title(f"Parametric study — {NUM_ARMS}-armed Bernoulli bandit")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def parametric_study() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"host         : {socket.gethostname()}")
    log(f"working dir  : {os.getcwd()}")
    log(f"python       : {sys.version.split()[0]}")
    log(f"slurm job id : {os.environ.get('SLURM_JOB_ID', '<not in slurm>')}")
    log(f"slurm node   : {os.environ.get('SLURMD_NODENAME', '<not in slurm>')}")
    log(f"workers      : {N_WORKERS}")
    log(f"time steps   : {TIME_STEPS}")
    log(f"num arms     : {NUM_ARMS}")
    log(f"seeds        : {len(SEEDS)}  ({SEEDS[0]}..{SEEDS[-1]})")
    for algo, param_name, values in ALGORITHMS:
        log(f"grid[{algo:<9}] {param_name}: {values}")
    log(f"csv          : {CSV_PATH.resolve()}")
    log("=" * 70)

    all_jobs: list[tuple[str, str, float, int]] = []
    for algo, param_name, values in ALGORITHMS:
        for value, seed in itertools.product(values, SEEDS):
            all_jobs.append((algo, param_name, float(value), seed))

    done = _load_done(CSV_PATH)
    pending = [j for j in all_jobs if (j[0], j[2], j[3]) not in done]

    log(f"jobs: {len(all_jobs)} total, {len(done)} cached, {len(pending)} to run")

    if not pending:
        log("nothing to do — all jobs already in CSV. Re-plotting.")
        _plot(CSV_PATH, PLOT_PATH)
        log(f"wrote {PLOT_PATH}")
        return

    t_start = time.perf_counter()
    completed = 0
    total = len(pending)

    with mp.Pool(processes=N_WORKERS) as pool:
        for result in pool.imap_unordered(_run_one, pending):
            _append_row(CSV_PATH, result)
            completed += 1

            elapsed = time.perf_counter() - t_start
            avg_per_job = elapsed / completed
            remaining = total - completed
            eta = avg_per_job * remaining / max(1, N_WORKERS)

            log(
                f"progress: {completed}/{total} "
                f"({100 * completed / total:5.1f}%)  "
                f"elapsed={elapsed:6.1f}s  ETA~{eta:6.1f}s"
            )

    log(f"all jobs done in {time.perf_counter() - t_start:.1f}s")

    _plot(CSV_PATH, PLOT_PATH)
    log(f"wrote plot: {PLOT_PATH.resolve()}")


if __name__ == "__main__":
    parametric_study()
