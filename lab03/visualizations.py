"""
Visualizations and experiments for MCTS in the isolation game.

Each `plot_*` function runs an experiment, draws a chart and saves it to a file.
By default images are written to `lab03/images/`.

Example usage:
    from visualizations import plot_win_rate_vs_c
    plot_win_rate_vs_c(
        c_values=[0.0, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0],
        n_games=50,
        time_limit=0.1,
        board_size=(4, 4),
    )
"""

import math
import os
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib

import numpy as np

from isolation import Board, Colour, Game, MCTSNode, MCTSPlayer, Player, RandomPlayer


PlayerFactory = Callable[[], Player]
DEFAULT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")


def _ensure_images_dir(path: str) -> str:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    return path


def _save(fig: plt.Figure, filename: str, images_dir: str = DEFAULT_IMAGES_DIR) -> str:
    out_path = _ensure_images_dir(os.path.join(images_dir, filename))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def run_match(
    red_factory: PlayerFactory,
    blue_factory: PlayerFactory,
    board_size: tuple[int, int],
    n_games: int,
    alternate_starts: bool = True,
) -> dict:
    """Play `n_games` matches. Returns wins and win rate from `red_factory`'s point of view.

    When `alternate_starts=True`, in half of the games the roles are swapped to
    compensate for the first-move advantage. The reported win rate always refers
    to the agent produced by `red_factory`.
    """
    width, height = board_size
    factory_a_wins = 0
    factory_b_wins = 0

    for i in range(n_games):
        board = Board(width, height)
        swap = alternate_starts and (i % 2 == 1)
        if swap:
            red = blue_factory()
            blue = red_factory()
        else:
            red = red_factory()
            blue = blue_factory()

        game = Game(red, blue, board)
        game.run(verbose=False)

        winner_is_a = (
            (not swap and game.winner == Colour.RED)
            or (swap and game.winner == Colour.BLUE)
        )
        if winner_is_a:
            factory_a_wins += 1
        else:
            factory_b_wins += 1

    return {
        "wins_a": factory_a_wins,
        "wins_b": factory_b_wins,
        "n_games": n_games,
        "win_rate_a": factory_a_wins / n_games,
    }


def plot_win_rate_vs_c(
    c_values: list[float],
    n_games: int = 50,
    time_limit: float = 0.1,
    board_size: tuple[int, int] = (4, 4),
    opponent_factory: PlayerFactory | None = None,
    filename: str = "win_rate_vs_c.png",
    images_dir: str = DEFAULT_IMAGES_DIR,
) -> dict:
    """Win rate of MCTS(c) vs `opponent_factory` (defaults to RandomPlayer)."""
    if opponent_factory is None:
        opponent_factory = RandomPlayer

    win_rates, lows, highs = [], [], []
    for c in c_values:
        result = run_match(
            red_factory=lambda c=c: MCTSPlayer(time_limit, c),
            blue_factory=opponent_factory,
            board_size=board_size,
            n_games=n_games,
        )
        wr = result["win_rate_a"]
        lo, hi = _wilson_interval(result["wins_a"], n_games)
        win_rates.append(wr)
        lows.append(wr - lo)
        highs.append(hi - wr)
        print(f"c={c:.2f}  win_rate={wr:.2f}  ({result['wins_a']}/{n_games})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(c_values, win_rates, yerr=[lows, highs], marker="o", capsize=4)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("c (UCB exploration parameter)")
    ax.set_ylabel("MCTS win rate")
    ax.set_title(
        f"Win rate vs. c (board size={board_size}, time limit={time_limit}s, simulations={n_games})"
    )
    ax.set_xticks(c_values, labels=[f"{c:g}" for c in c_values])
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    out = _save(fig, filename, images_dir)
    plt.close(fig)
    print(f"saved: {out}")
    return {"c_values": c_values, "win_rates": win_rates}


def plot_win_rate_vs_time(
    time_limits: list[float],
    n_games: int = 50,
    c_coefficient: float = 0.5,
    board_size: tuple[int, int] = (4, 4),
    opponent_factory: PlayerFactory | None = None,
    filename: str = "win_rate_vs_time.png",
    images_dir: str = DEFAULT_IMAGES_DIR,
) -> dict:
    """Win rate of MCTS(time_limit) vs `opponent_factory` (defaults to Random)."""
    if opponent_factory is None:
        opponent_factory = RandomPlayer

    win_rates, lows, highs = [], [], []
    for t in time_limits:
        result = run_match(
            red_factory=lambda t=t: MCTSPlayer(t, c_coefficient),
            blue_factory=opponent_factory,
            board_size=board_size,
            n_games=n_games,
        )
        wr = result["win_rate_a"]
        lo, hi = _wilson_interval(result["wins_a"], n_games)
        win_rates.append(wr)
        lows.append(wr - lo)
        highs.append(hi - wr)
        print(f"t={t:.3f}s  win_rate={wr:.2f}  ({result['wins_a']}/{n_games})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(time_limits, win_rates, yerr=[lows, highs], marker="o", capsize=4)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("time_limit per move [s] (log scale)")
    ax.set_ylabel("MCTS win rate")
    ax.set_title(
        f"Win rate vs time   (board={board_size}, c={c_coefficient}, n={n_games})"
    )
    ax.set_xticks(time_limits)
    ax.set_xticklabels([f"{t:g}" for t in time_limits], rotation=45, ha="right")
    ax.get_xaxis().set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, which="both")
    out = _save(fig, filename, images_dir)
    plt.close(fig)
    print(f"saved: {out}")
    return {"time_limits": time_limits, "win_rates": win_rates}


def plot_heatmap_c_vs_time(
    c_values: list[float],
    time_limits: list[float],
    n_games: int = 30,
    board_size: tuple[int, int] = (4, 4),
    opponent_factory: PlayerFactory | None = None,
    filename: str = "heatmap_c_vs_time.png",
    images_dir: str = DEFAULT_IMAGES_DIR,
) -> dict:
    """Win rate heatmap over the (c, time_limit) grid."""
    if opponent_factory is None:
        opponent_factory = RandomPlayer

    grid = np.zeros((len(time_limits), len(c_values)))
    for i, t in enumerate(time_limits):
        for j, c in enumerate(c_values):
            result = run_match(
                red_factory=lambda t=t, c=c: MCTSPlayer(t, c),
                blue_factory=opponent_factory,
                board_size=board_size,
                n_games=n_games,
            )
            grid[i, j] = result["win_rate_a"]
            print(f"t={t:.3f}  c={c:.2f}  wr={grid[i, j]:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(c_values)))
    ax.set_xticklabels([f"{c:.2f}" for c in c_values])
    ax.set_yticks(range(len(time_limits)))
    ax.set_yticklabels([f"{t:.3f}" for t in time_limits])
    ax.set_xlabel("c (exploration)")
    ax.set_ylabel("time_limit [s]")
    ax.set_title(f"MCTS win rate vs Random   (board={board_size}, n={n_games})")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if grid[i, j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="win rate")
    out = _save(fig, filename, images_dir)
    plt.close(fig)
    print(f"saved: {out}")
    return {"grid": grid, "c_values": c_values, "time_limits": time_limits}


def plot_tournament_matrix(
    configs: list[dict],
    n_games: int = 30,
    board_size: tuple[int, int] = (4, 4),
    filename: str = "tournament_matrix.png",
    images_dir: str = DEFAULT_IMAGES_DIR,
) -> dict:
    """Round-robin tournament between MCTS configurations.

    `configs` is a list of dicts `{"label": str, "time": float, "c": float}`.
    Cell [i, j] = win rate of configuration i playing against configuration j.
    """
    n = len(configs)
    grid = np.full((n, n), np.nan)

    for i, cfg_a in enumerate(configs):
        for j, cfg_b in enumerate(configs):
            if i == j:
                continue
            result = run_match(
                red_factory=lambda c=cfg_a: MCTSPlayer(c["time"], c["c"]),
                blue_factory=lambda c=cfg_b: MCTSPlayer(c["time"], c["c"]),
                board_size=board_size,
                n_games=n_games,
            )
            grid[i, j] = result["win_rate_a"]
            print(f"{cfg_a['label']} vs {cfg_b['label']}: {grid[i, j]:.2f}")

    labels = [c["label"] for c in configs]
    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.2 * n + 2))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("opponent")
    ax.set_ylabel("agent")
    ax.set_title(f"Round-robin tournament   (board={board_size}, n={n_games})")
    for i in range(n):
        for j in range(n):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="win rate (row vs column)")
    out = _save(fig, filename, images_dir)
    plt.close(fig)
    print(f"saved: {out}")
    return {"grid": grid, "labels": labels}
