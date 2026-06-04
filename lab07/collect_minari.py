"""
Roll out a trained PPO expert and save its trajectories as a Minari dataset.

This produces the OFFLINE dataset for the Decision Transformer: we load the
expert trained by train_lunarlander.py, play many episodes, and let Minari's
DataCollector record (observation, action, reward, termination) for every step.

    # collect 1000 episodes from models/expert.zip into lunarlander/expert-v0
    uv run python collect_minari.py --run-name expert --num-episodes 1000

    # quick test
    uv run python collect_minari.py --run-name expert --num-episodes 20

Load the result later with:
    import minari
    dataset = minari.load_dataset("lunarlander/expert-v0")
"""

import argparse
import os
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


def parse_args():
    p = argparse.ArgumentParser(description="Collect PPO trajectories into Minari")
    p.add_argument("--env", default="LunarLanderContinuous-v3", help="Gymnasium env id")
    p.add_argument(
        "--run-name", default=None, help="Run name of the model to load (no .zip)"
    )
    p.add_argument("--models-dir", default="models", help="Folder holding the model")
    p.add_argument(
        "--dataset-id",
        default="lunarlander/expert-v0",
        help="Minari dataset id: namespace/name-vX",
    )
    p.add_argument(
        "--datasets-dir",
        default="minari_datasets",
        help="Local folder to store Minari datasets",
    )
    p.add_argument(
        "--num-episodes", type=int, default=1000, help="How many episodes to record"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy (more diverse) instead of the best one",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing dataset with the same id before collecting",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Must be set BEFORE importing minari so datasets land in our local folder
    # instead of the global ~/.minari directory.
    os.environ["MINARI_DATASETS_PATH"] = str(Path(args.datasets_dir).resolve())
    import minari
    from minari import DataCollector

    # Locate and load the expert (same naming scheme as train_lunarlander.py).
    run_name = args.run_name or f"ppo_{args.env}"
    model_path = str(Path(args.models_dir) / run_name)
    model = PPO.load(model_path)
    print(f"[load] expert <- {model_path}.zip")

    # If the dataset already exists, Minari refuses to overwrite it silently.
    if args.dataset_id in minari.list_local_datasets():
        if args.overwrite:
            minari.delete_dataset(args.dataset_id)
            print(f"[minari] deleted existing dataset {args.dataset_id}")
        else:
            raise SystemExit(
                f"dataset '{args.dataset_id}' already exists — pass --overwrite to replace it"
            )

    # DataCollector wraps the env and records every step into an internal buffer.
    # We wrap a plain gym.make (no Monitor) so Minari can store the env spec and
    # later recreate the exact environment for evaluation.
    env = DataCollector(gym.make(args.env), record_infos=False)

    deterministic = not args.stochastic
    returns, lengths = [], []

    for ep in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            # The expert maps observation -> action; DataCollector records it.
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)

        if (ep + 1) % max(args.num_episodes // 10, 1) == 0:
            print(
                f"  {ep + 1:>5}/{args.num_episodes} episodes "
                f"| running mean return={np.mean(returns):.1f}"
            )

    # Flush the recorded buffer into a persisted, named Minari dataset.
    dataset = env.create_dataset(
        dataset_id=args.dataset_id,
        algorithm_name="PPO",
        description=f"PPO {args.run_name} demonstrations on {args.env}",
    )
    env.close()

    returns = np.array(returns)
    print(f"\n[minari] saved dataset '{args.dataset_id}'")
    print(f"  episodes        : {dataset.total_episodes}")
    print(f"  transitions     : {dataset.total_steps}")
    print(f"  mean return     : {returns.mean():.1f} +/- {returns.std():.1f}")
    print(f"  min / max return: {returns.min():.1f} / {returns.max():.1f}")
    print(f"  solved (>=200)  : {(returns >= 200).mean() * 100:.0f}% of episodes")
    print(f"  stored in       : {os.environ['MINARI_DATASETS_PATH']}")


if __name__ == "__main__":
    main()

# uv run python collect_minari.py --run-name ppo_LunarLanderContinuous-v3_100k_s0 --dataset-id lunarlander/ppo_beginner_100k-v0 --num-episodes 1000
# uv run python collect_minari.py --run-name ppo_LunarLanderContinuous-v3_400k_s0 --dataset-id lunarlander/ppo_intermediate_400k-v0 --num-episodes 1000
# uv run python collect_minari.py --run-name ppo_LunarLanderContinuous-v3_1000k_s0 --dataset-id lunarlander/ppo_expert_1000k-v0 --num-episodes 1000
