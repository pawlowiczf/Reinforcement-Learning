"""
Parametric study of n-step off-policy SARSA, in the style of Sutton & Barto Fig. 7.2.

For each (n, alpha) combination we run several independent seeds, train, then
evaluate with the greedy target policy. The reported metric is the average
penalty per evaluation episode (lower = better), averaged across seeds.

Two things make this fast enough to actually iterate on:

1. The per-episode `_draw_episode` callback in `problem.Experiment` writes a
   300-DPI PNG every 50 episodes. That dominates wall time. We disable it in
   workers by raising `problem.DRAWING_FREQUENCY` and silencing the inner tqdm.
2. (n, alpha, seed) jobs are independent, so we farm them out to a
   `multiprocessing.Pool`. On Windows this uses spawn, hence the worker init.

Raw per-job results are appended to a CSV so a partial run is never lost.
Re-running with the same CSV path skips already-completed (n, alpha, seed)
triples — handy when tweaking the grid.
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
# CORNER_NAME = "corner_b"          # smallest corner -> fastest sweep; swap if desired
# TRAIN_EPISODES = 1500
# EVAL_EPISODES = 50
# EXPERIMENT_RATE = 0.1
# DISCOUNT_FACTOR = 1.0
# STEERING_FAIL_CHANCE = 0.01

CORNER_NAME = "corner_b"          # smallest corner -> fastest sweep; swap if desired
TRAIN_EPISODES = 1500
EVAL_EPISODES = 50
EXPERIMENT_RATE = 0.1
DISCOUNT_FACTOR = 1.0
STEERING_FAIL_CHANCE = 0.01

N_VALUES = [1, 2, 4, 8, 16]
ALPHAS = [round(a, 3) for a in np.linspace(0.1, 1.0, 10)]
SEEDS = [0, 1, 2]

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

def _init_worker() -> None:
    """Runs once per spawned worker. Disables expensive per-episode side effects."""
    import problem

    problem.DRAWING_FREQUENCY = 10**9                # never trigger _draw_episode
    problem.tqdm = lambda x, **kwargs: x             # silence the inner progress bar


def _run_one(job: tuple[int, float, int]) -> tuple[int, float, int, float, float]:
    """Run a single (n, alpha, seed) job. Returns (n, alpha, seed, mean_penalty, seconds)."""
    n, alpha, seed = job
    pid = os.getpid()
    t0 = time.perf_counter()

    print(f"[pid {pid}] START  n={n:>2}  alpha={alpha:<5}  seed={seed}", flush=True)

    random.seed(seed)
    np.random.seed(seed)

    # Imported here so the workers don't pay the import cost until they actually run.
    from problem import Corner, Environment, Experiment
    from solution import OffPolicyNStepSarsaDriver

    corner = Corner(name=CORNER_NAME)

    driver = OffPolicyNStepSarsaDriver(
        step_no=n,
        step_size=alpha,
        experiment_rate=EXPERIMENT_RATE,
        discount_factor=DISCOUNT_FACTOR,
    )

    train = Experiment(
        environment=Environment(corner=corner, steering_fail_chance=STEERING_FAIL_CHANCE),
        driver=driver,
        number_of_episodes=TRAIN_EPISODES,
    )
    train.run()

    driver.evaluation_mode = True
    eval_exp = Experiment(
        environment=Environment(corner=corner, steering_fail_chance=STEERING_FAIL_CHANCE),
        driver=driver,
        number_of_episodes=EVAL_EPISODES,
    )
    eval_exp.run()

    mean_penalty = float(np.mean(eval_exp.penalties))
    elapsed = time.perf_counter() - t0
    print(
        f"[pid {pid}] DONE   n={n:>2}  alpha={alpha:<5}  seed={seed}  "
        f"-> mean_penalty={mean_penalty:>8.2f}  ({elapsed:.1f}s)",
        flush=True,
    )
    return n, alpha, seed, mean_penalty, elapsed


# ---------------------------------------------------------------- orchestration

def _load_done(csv_path: Path) -> set[tuple[int, float, int]]:
    if not csv_path.exists():
        return set()
    done: set[tuple[int, float, int]] = set()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((int(row["n"]), float(row["alpha"]), int(row["seed"])))
    return done


def _append_row(csv_path: Path, row: tuple[int, float, int, float, float]) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["n", "alpha", "seed", "mean_penalty", "seconds"])
        writer.writerow(row)


def _plot(csv_path: Path, plot_path: Path) -> None:
    """Group CSV rows by (n, alpha), average over seeds, draw one curve per n."""
    by_key: dict[tuple[int, float], list[float]] = {}
    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            by_key.setdefault((int(row["n"]), float(row["alpha"])), []).append(
                float(row["mean_penalty"])
            )

    plt.figure(figsize=(8, 5))
    for n in sorted({n for n, _ in by_key}):
        alphas_n = sorted({a for nn, a in by_key if nn == n})
        means = [np.mean(by_key[(n, a)]) for a in alphas_n]
        plt.plot(alphas_n, means, marker="o", label=f"n={n}")

    plt.xlabel(r"$\alpha$ (step size)")
    plt.ylabel(f"avg penalty over {EVAL_EPISODES} eval episodes (lower is better)")
    plt.title(
        f"n-step off-policy SARSA on {CORNER_NAME}  "
        f"({TRAIN_EPISODES} train ep., averaged over {len(SEEDS)} seeds)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def parametric_study() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Environment / SLURM info banner
    log("=" * 70)
    log(f"host         : {socket.gethostname()}")
    log(f"working dir  : {os.getcwd()}")
    log(f"python       : {sys.version.split()[0]}")
    log(f"slurm job id : {os.environ.get('SLURM_JOB_ID', '<not in slurm>')}")
    log(f"slurm node   : {os.environ.get('SLURMD_NODENAME', '<not in slurm>')}")
    log(f"workers      : {N_WORKERS}")
    log(f"corner       : {CORNER_NAME}")
    log(f"train ep.    : {TRAIN_EPISODES}")
    log(f"eval ep.     : {EVAL_EPISODES}")
    log(f"grid         : n={N_VALUES}, alphas={ALPHAS}, seeds={SEEDS}")
    log(f"csv          : {CSV_PATH.resolve()}")
    log("=" * 70)

    all_jobs = list(itertools.product(N_VALUES, ALPHAS, SEEDS))
    done = _load_done(CSV_PATH)
    pending = [j for j in all_jobs if j not in done]

    log(f"jobs: {len(all_jobs)} total, {len(done)} cached, {len(pending)} to run")

    if not pending:
        log("nothing to do — all jobs already in CSV. Re-plotting.")
        _plot(CSV_PATH, PLOT_PATH)
        log(f"wrote {PLOT_PATH}")
        return

    t_start = time.perf_counter()
    completed = 0
    total = len(pending)

    with mp.Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
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