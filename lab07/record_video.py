"""
Record an mp4 (or watch in a live window) of a trained agent playing LunarLander.

Renders the environment to rgb frames and writes them with imageio (which ships
its own ffmpeg, so no system install is needed). Works for both the Decision
Transformer and the original PPO expert, so you can watch them side by side.

For the DT you can pin the return-to-go the model is conditioned on with
--target-return; by default it uses the mean return of the dataset's best 10%.

    # one expert-DT episode -> output/videos/expert_dt.mp4
    uv run python record_video.py --level expert

    # the original PPO expert instead
    uv run python record_video.py --level expert --which ppo

    # condition the DT on a specific return-to-go (RTG)
    uv run python record_video.py --level expert --target-return 300

    # open a live window instead of writing a file
    uv run python record_video.py --level expert --target-return 300 --watch

    # stitch 3 episodes into one clip
    uv run python record_video.py --level beginner --episodes 3

    # record DT and PPO for every level in one go
    uv run python record_video.py --all
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import gymnasium as gym
import minari

from dt_eval import (
    ENV_ID,
    LEVELS,
    rollout_dt,
    rollout_ppo,
    target_return_from_dataset,
)
from nanodt.agent import NanoDTAgent
from stable_baselines3 import PPO


def parse_args():
    p = argparse.ArgumentParser(description="Record/watch a DT or PPO agent")
    p.add_argument("--level", default="expert", choices=list(LEVELS))
    p.add_argument("--which", default="dt", choices=["dt", "ppo"])
    p.add_argument("--episodes", type=int, default=1, help="Episodes per clip")
    p.add_argument("--seed", type=int, default=0, help="Base seed for episodes")
    p.add_argument(
        "--target-return",
        type=float,
        default=None,
        help="RTG to condition the DT on (default: mean of dataset's best 10%%)",
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="Open a live window instead of writing an mp4",
    )
    p.add_argument("--out-dir", default="output/videos")
    p.add_argument("--fps", type=int, default=None, help="Override video fps")
    p.add_argument(
        "--all",
        action="store_true",
        help="Record DT and PPO for every level (ignores --level/--which)",
    )
    return p.parse_args()


def run(level, which, episodes, base_seed, render_mode, target_return=None):
    """Play `episodes` and return (returns, frames, target_used)."""
    cfg = LEVELS[level]
    env = gym.make(ENV_ID, render_mode=render_mode)
    collect = render_mode == "rgb_array"  # only the mp4 path needs frames in memory

    target_used = None
    if which == "dt":
        agent = NanoDTAgent.load(cfg["dt"])
        # Use the requested RTG, or fall back to the dataset-derived one.
        target_used = (
            target_return
            if target_return is not None
            else target_return_from_dataset(minari.load_dataset(cfg["dataset"]))
        )
    else:
        model = PPO.load(cfg["ppo"])

    frames, returns = [], []
    for i in range(episodes):
        if which == "dt":
            r, ep_frames = rollout_dt(
                agent, env, target_used, seed=base_seed + i, render=collect
            )
        else:
            r, ep_frames = rollout_ppo(model, env, seed=base_seed + i, render=collect)
        frames.extend(ep_frames)
        returns.append(r)
    env.close()
    return returns, frames, target_used


def label(level, which, target_used):
    tag = f"{level}_{which}"
    if which == "dt" and target_used is not None:
        tag += f"_rtg{round(target_used)}"
    return tag


def record(level, which, episodes, base_seed, out_dir, fps, target_return):
    returns, frames, target_used = run(
        level, which, episodes, base_seed, "rgb_array", target_return
    )
    fps = fps or gym.make(ENV_ID).metadata.get("render_fps", 50)
    out_path = Path(out_dir) / f"{label(level, which, target_used)}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=1)
    report(level, which, episodes, returns, target_used, str(out_path))


def watch(level, which, episodes, base_seed, target_return):
    returns, _, target_used = run(
        level, which, episodes, base_seed, "human", target_return
    )
    report(level, which, episodes, returns, target_used, "live window")


def report(level, which, episodes, returns, target_used, dest):
    avg = sum(returns) / len(returns)
    rtg = f"RTG={target_used:.0f} " if target_used is not None else ""
    print(
        f"[{level:>12} | {which:>3}] {rtg}{episodes} ep, "
        f"mean return {avg:6.1f} -> {dest}"
    )


def main():
    args = parse_args()
    if args.all:
        jobs = [(lv, w) for lv in LEVELS for w in ("dt", "ppo")]
    else:
        jobs = [(args.level, args.which)]

    for level, which in jobs:
        if args.watch:
            watch(level, which, args.episodes, args.seed, args.target_return)
        else:
            record(
                level,
                which,
                args.episodes,
                args.seed,
                args.out_dir,
                args.fps,
                args.target_return,
            )


if __name__ == "__main__":
    main()
