# A script to load a trained SAC agent from a checkpoint and visualize it
# acting in the environment (render_mode="human" opens a PyBullet window).
#
# Usage:
#   python play.py                              # PandaReach-v3, checkpoint.pt
#   python play.py --checkpoint runs/best.pt
#   python play.py --env PandaPush-v3 --episodes 10 --sleep 0.02
#
# IMPORTANT: the policy/algo must be built with the SAME parameters as during
# training (hidden_sizes, extractor_type, env id), otherwise loading the
# state_dict will fail with a shape-mismatch error.

import argparse

import gymnasium as gym

# Uncomment the following line to use gymnasium_robotics environments
import gymnasium_robotics
import panda_gym

# Uncomment the following lines to register gymnasium_robotics environments
gym.register_envs(gymnasium_robotics)

from asdf.algos import SAC
from asdf.buffers import HerReplayBuffer
from asdf.extractors import DictExtractor
from asdf.policies import MlpPolicy


def main(env_id: str, checkpoint: str, n_episodes: int, sleep: float) -> None:
    # Visualization runs on CPU; no GPU is needed just to play episodes.
    env = gym.make(env_id)

    policy = MlpPolicy(
        env.observation_space,
        env.action_space,
        hidden_sizes=[64, 64],
        extractor_type=DictExtractor,
    )

    buffer = HerReplayBuffer(
        env=env,
        size=1_000_000,
        n_sampled_goal=3,
        goal_selection_strategy="future",
        device="cpu",
    )

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        alpha="auto",
        max_episode_len=100,
    )

    # Load the trained weights.
    algo.load(checkpoint)
    env.close()

    # Re-create the environment with a window and replay deterministic episodes.
    env = gym.make(env_id, render_mode="human")
    results = algo.test(env, n_episodes=n_episodes, sleep=sleep)
    env.close()

    print(f"Mean episode return: {results['mean_ep_ret']:.3g}")
    print(f"Mean episode length: {results['mean_ep_len']:.3g}")
    if "success_rate" in results:
        print(f"Success rate: {results['success_rate']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="PandaReach-v3", help="Gym environment ID"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint.pt",
        help="Path to the saved checkpoint file",
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of episodes to play"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1 / 30,
        help="Seconds to sleep between steps (slows down the animation)",
    )

    args = parser.parse_args()

    main(args.env, args.checkpoint, args.episodes, args.sleep)
