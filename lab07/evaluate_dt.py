import argparse
import os
from pathlib import Path

# Same local Minari folder as training, so we can read the dataset to derive R̂₁.
os.environ["MINARI_DATASETS_PATH"] = str(
    (Path(__file__).parent / "minari_datasets").resolve()
)

import minari
import numpy as np
import gymnasium as gym

from nanodt.agent import NanoDTAgent

# Defaults (must match what train_dt.py produced). Override with CLI args.
DATASET_ID = "lunarlander/ppo_beginner_100k-v0"
MODEL_PATH = "output/dt/ppo_beginner_100k-v0.pth"
ENV_ID = "LunarLanderContinuous-v3"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a nanoDT Decision Transformer")
    p.add_argument(
        "--dataset-id", default=DATASET_ID, help="Minari dataset used to derive R̂₁"
    )
    p.add_argument("--model-path", default=MODEL_PATH, help="Trained DT to load (.pth)")
    p.add_argument("--env", default=ENV_ID, help="Gymnasium env id to play in")
    return p.parse_args()


def target_return_from_dataset(dataset, top_frac=0.1):
    """
    Algorithm to calculate return-to-go:
    1) take every trajectory in the data
    2) compute each trajectory's return (sum of rewards)
    3) take the top `top_frac` (best 10%) of those returns
    4) average them
    """
    returns = np.array([ep.rewards.sum() for ep in dataset.iterate_episodes()])
    k = max(1, int(len(returns) * top_frac))
    top = np.sort(returns)[-k:]
    return float(top.mean())


def evaluate(dataset_id, model_path, env_id):
    dataset = minari.load_dataset(dataset_id)
    target_return = target_return_from_dataset(dataset)
    print(f"R̂₁ (mean return of top 10% trajectories) = {target_return:.1f}")

    env = gym.make(env_id)
    agent = NanoDTAgent.load(model_path)

    for episode in range(5):
        agent.reset(target_return=target_return)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        rew = None  # no reward before the first action; act() skips the RTG update
        while not done:
            # Pass the previous step's reward so the DT decreases its return-to-go.
            action = agent.act(obs, rew)
            obs, rew, ter, tru, _ = env.step(action)
            done = ter or tru
            total_reward += rew

        print(f"Episode {episode + 1}: total reward = {total_reward:.1f}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.dataset_id, args.model_path, args.env)

# uv run python evaluate_dt.py --dataset-id lunarlander/ppo_beginner_100k-v0 --model-path output/dt/ppo_beginner_100k-v0.pth
