"""
Shared helpers for comparing a trained Decision Transformer against the PPO
expert that generated its training data.

For each skill level we know three things that belong together:
  - the Minari dataset of PPO demonstrations the DT was trained on,
  - the trained DT checkpoint,
  - the original PPO model that produced those demonstrations.

compare_dt.py and record_video.py both build on the rollout helpers here.
"""

import os
from pathlib import Path

# Point Minari at the LOCAL dataset folder BEFORE importing it (same as the
# training/eval scripts), otherwise load_dataset() looks in global ~/.minari.
os.environ.setdefault(
    "MINARI_DATASETS_PATH", str((Path(__file__).parent / "minari_datasets").resolve())
)

import numpy as np
import gymnasium as gym
import minari
from nanodt.agent import NanoDTAgent
from stable_baselines3 import PPO

ENV_ID = "LunarLanderContinuous-v3"

# level -> the matching dataset / DT checkpoint / original PPO expert.
LEVELS = {
    "beginner": {
        "dataset": "lunarlander/ppo_beginner_100k-v0",
        "dt": "output/dt/ppo_beginner_100k-v0.pth",
        "ppo": "models/ppo_LunarLanderContinuous-v3_100k_s0",
    },
    "junior": {
        "dataset": "lunarlander/ppo_junior_200k-v0",
        "dt": "output/dt/ppo_junior_200k-v0.pth",
        "ppo": "models/ppo_LunarLanderContinuous-v3_200k_s0",   
    },
    "intermediate": {
        "dataset": "lunarlander/ppo_intermediate_400k-v0",
        "dt": "output/dt/ppo_intermediate_400k-v0.pth",
        "ppo": "models/ppo_LunarLanderContinuous-v3_400k_s0",
    },
    "expert": {
        "dataset": "lunarlander/ppo_expert_1000k-v0",
        "dt": "output/dt/ppo_expert_1000k-v0.pth",
        "ppo": "models/ppo_LunarLanderContinuous-v3_1000k_s0",
    },
}

SOLVED_THRESHOLD = 200.0  # LunarLander is "solved" at mean reward >= 200


def dataset_returns(dataset):
    """Per-episode return (sum of rewards) of every demonstration trajectory."""
    return np.array([ep.rewards.sum() for ep in dataset.iterate_episodes()])


def target_return_from_dataset(dataset, top_frac=0.1):
    """
    The return-to-go we ask the DT to achieve at eval time: the mean return of
    the best `top_frac` trajectories in the dataset. Same recipe as evaluate_dt.py.
    """
    returns = dataset_returns(dataset)
    k = max(1, int(len(returns) * top_frac))
    top = np.sort(returns)[-k:]
    return float(top.mean())


def rollout_dt(agent, env, target_return, seed=None, render=False):
    """Play one episode with the DT. Returns (total_reward, frames)."""
    agent.reset(target_return=target_return)
    obs, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    rew = None  # no reward before the first action; act() skips the RTG update
    frames = []
    while not done:
        if render:
            frames.append(env.render())
        action = agent.act(obs, rew)
        obs, rew, ter, tru, _ = env.step(action)
        done = ter or tru
        total += rew
    if render:
        frames.append(env.render())
    return total, frames


def rollout_ppo(model, env, seed=None, render=False):
    """Play one episode with the PPO expert. Returns (total_reward, frames)."""
    obs, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    frames = []
    while not done:
        if render:
            frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, ter, tru, _ = env.step(action)
        done = ter or tru
        total += rew
    if render:
        frames.append(env.render())
    return total, frames


def evaluate_dt(level, episodes, base_seed=0, render_mode=None):
    """Run `episodes` DT episodes for a level. Returns (returns_array, frames_of_last)."""
    cfg = LEVELS[level]
    dataset = minari.load_dataset(cfg["dataset"])
    target = target_return_from_dataset(dataset)
    env = gym.make(ENV_ID, render_mode=render_mode)
    agent = NanoDTAgent.load(cfg["dt"])
    returns, frames = [], []
    for i in range(episodes):
        r, frames = rollout_dt(
            agent, env, target, seed=base_seed + i, render=render_mode == "rgb_array"
        )
        returns.append(r)
    env.close()
    return np.array(returns), frames


def evaluate_ppo(level, episodes, base_seed=0, render_mode=None):
    """Run `episodes` PPO episodes for a level. Returns (returns_array, frames_of_last)."""
    cfg = LEVELS[level]
    env = gym.make(ENV_ID, render_mode=render_mode)
    model = PPO.load(cfg["ppo"])
    returns, frames = [], []
    for i in range(episodes):
        r, frames = rollout_ppo(
            model, env, seed=base_seed + i, render=render_mode == "rgb_array"
        )
        returns.append(r)
    env.close()
    return np.array(returns), frames
