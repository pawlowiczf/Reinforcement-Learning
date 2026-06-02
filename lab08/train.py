"""
PPO training script for Crafter environment using Stable-Baselines3.

Usage:
    # Baseline run
    python train_ppo_crafter.py --run baseline

    # Experiment run (custom hyperparams)
    python train_ppo_crafter.py --run experiment --lr 5e-4 --ent-coef 0.01

Requirements:
    pip install stable-baselines3 crafter gymnasium opencv-python matplotlib
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium.spaces as spaces
import crafter

# --- Stable-Baselines3 (SB3): library of ready-made, tested RL algorithms ---
# PPO                : the learning algorithm we use (actor-critic, on-policy).
# DummyVecEnv        : runs several env copies in ONE process so the agent
#                      collects data from many games at once.
# VecTransposeImage  : reorders image axes (H,W,C) -> (C,H,W) expected by CNNs.
# Monitor            : env wrapper that records per-episode reward/length and
#                      injects them into the `info` dict under the "episode" key.
# BaseCallback       : base class to hook our own code into the training loop.
# EvalCallback       : built-in callback that periodically evaluates the agent
#                      on a separate env and auto-saves the best model.
# evaluate_policy    : helper that runs a trained model for N episodes and
#                      returns mean/std reward.
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


# ---------------------------------------------------------------------------
# Crafter shim: crafter.Env uses the legacy gym API (4-tuple step, no truncated).
# We bridge it manually to the gymnasium API so nothing touches the old gym package.
# ---------------------------------------------------------------------------


class CrafterGymnasiumEnv(gym.Env):
    """
    Pure gymnasium wrapper around crafter.Env.

    crafter.Env internally uses old gym conventions:
      - reset() returns only obs (no info dict)
      - step() returns (obs, reward, done, info)  -- 4-tuple, no truncated

    We translate both to the gymnasium 5-tuple API here so that SB3
    never has to import the legacy gym package.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 10_000):
        super().__init__()
        self._env = crafter.Env()
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        # Every gym/gymnasium env MUST declare two "spaces" so SB3 knows the
        # shapes of inputs/outputs and can build the right neural network:
        #   observation_space -> what the agent SEES (here: an RGB image)
        #   action_space      -> what the agent can DO  (here: discrete actions)
        obs_space = self._env.observation_space  # already numpy-compatible
        act_space = self._env.action_space

        # Box = a continuous/array space; here a (H, W, 3) image of uint8 pixels.
        # SB3 sees this is an image and picks the CNN feature extractor.
        self.observation_space = spaces.Box(
            low=obs_space.low,
            high=obs_space.high,
            shape=obs_space.shape,
            dtype=np.uint8,
        )
        # Discrete(n) = a finite set of actions {0, 1, ..., n-1}.
        self.action_space = spaces.Discrete(act_space.n)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        # Called at the start of each episode. Gymnasium expects (obs, info).
        if seed is not None:
            self._env._world._seed = seed  # best-effort; crafter has no public seed API
        obs = self._env.reset()  # legacy API: returns obs only
        self._elapsed_steps = 0
        return obs, {}

    def step(self, action):
        # Apply one action. Gymnasium expects a 5-tuple:
        #   (obs, reward, terminated, truncated, info)
        # where terminated = episode ended naturally (death/goal),
        #       truncated  = episode cut off by a time/step limit.
        obs, reward, done, info = self._env.step(action)  # legacy 4-tuple
        self._elapsed_steps += 1

        truncated = self._elapsed_steps >= self._max_episode_steps
        terminated = bool(done) and not truncated

        # Force episode end when time limit is hit
        if truncated:
            done = True

        return obs, float(reward), terminated, truncated, info

    def render(self, size=None):
        # Returns an RGB image of the current game state. Pass size=(W, H) for
        # a higher-resolution frame (default is the 64x64 observation image).
        return self._env.render(size) if size is not None else self._env.render()

    def close(self):
        # crafter.Env may not implement close() (legacy gym), so guard for it.
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Callback: log episode rewards and achievements during training
# ---------------------------------------------------------------------------


class TrainingLoggerCallback(BaseCallback):
    """
    Custom SB3 callback that collects per-episode stats for plotting.

    A "callback" is an object whose methods SB3 calls automatically during
    model.learn(), so we can run our own code without writing the training
    loop ourselves. We subclass BaseCallback and override _on_step().
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Our own buffers — filled as episodes finish during training.
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.timesteps_at_episode_end: list[int] = []
        # Crafter-specific: how many DISTINCT achievements were unlocked in
        # each episode, plus the full per-achievement counts (for the profile).
        self.episode_n_achievements: list[int] = []
        self.episode_achievements: list[dict] = []

    def _on_step(self) -> bool:
        # _on_step() is called by SB3 after EVERY environment step.
        # self.locals gives access to the training loop's local variables;
        # "infos" is the list of info dicts (one per parallel env).
        for info in self.locals.get("infos", []):
            # The Monitor wrapper adds an "episode" entry only when an episode
            # has just ended, containing r=total reward, l=length (steps).
            if "episode" in info:
                ep = info["episode"]
                self.episode_rewards.append(ep["r"])
                self.episode_lengths.append(ep["l"])
                # self.num_timesteps = total steps taken so far (the x-axis).
                self.timesteps_at_episode_end.append(self.num_timesteps)

                # Crafter puts a {name: count} dict of its 22 achievements in
                # info["achievements"]; count an achievement as "unlocked" if
                # it happened at least once this episode.
                ach = info.get("achievements", {}) or {}
                self.episode_achievements.append(dict(ach))
                self.episode_n_achievements.append(
                    sum(1 for v in ach.values() if v > 0)
                )
        # Returning True = keep training; returning False would stop it early.
        return True


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def smooth(values: list[float], window: int = 10) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_results(
    runs: dict[str, TrainingLoggerCallback],
    out_dir: Path,
) -> None:
    # Three learning curves over training time, one line per run so baseline
    # and experiment can be compared on the same axes.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    fig.suptitle("PPO on Crafter -- Training Progress", fontsize=13)

    for label, cb in runs.items():
        ts = np.array(cb.timesteps_at_episode_end)
        rw = smooth(cb.episode_rewards)
        # smooth() shortens the series; align the x-axis to the smoothed length.
        ts_smooth = ts[len(ts) - len(rw) :]

        axes[0].plot(ts_smooth, rw, label=label)
        axes[1].plot(ts_smooth, smooth(cb.episode_lengths), label=label)
        axes[2].plot(ts_smooth, smooth(cb.episode_n_achievements), label=label)

    axes[0].set_title("Episode Reward (smoothed)")
    axes[0].set_ylabel("Reward")

    # Episode length = how long the agent survives before dying / timing out.
    axes[1].set_title("Survival -- Episode Length (smoothed)")
    axes[1].set_ylabel("Steps")

    # Distinct achievements per episode = the clearest "is it learning?" signal.
    axes[2].set_title("Distinct Achievements / Episode (smoothed)")
    axes[2].set_ylabel("# achievements (of 22)")

    for ax in axes:
        ax.set_xlabel("Timesteps")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=120)
    print(f"[plot] saved -> {out_path}")
    plt.show()


def plot_achievements(
    runs: dict[str, TrainingLoggerCallback],
    out_dir: Path,
) -> None:
    """Bar chart: fraction of episodes in which each achievement was unlocked."""
    # Collect the union of achievement names seen across all runs.
    names: list[str] = []
    for cb in runs.values():
        if cb.episode_achievements:
            names = sorted(cb.episode_achievements[-1].keys())
            break
    if not names:
        print("[plot] no achievements recorded, skipping achievement plot")
        return

    # For each run, success rate = in how many episodes each achievement fired.
    rates: dict[str, np.ndarray] = {}
    for label, cb in runs.items():
        eps = cb.episode_achievements
        n = max(len(eps), 1)
        rates[label] = np.array(
            [sum(1 for e in eps if e.get(name, 0) > 0) / n for name in names]
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(names))
    height = 0.8 / len(runs)
    for i, (label, vals) in enumerate(rates.items()):
        ax.barh(y + i * height, vals * 100, height=height, label=label)

    ax.set_yticks(y + height * (len(runs) - 1) / 2)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Episodes unlocked (%)")
    ax.set_title("Achievement profile -- success rate over all training episodes")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    out_path = out_dir / "achievements.png"
    plt.savefig(out_path, dpi=120)
    print(f"[plot] saved -> {out_path}")
    plt.show()


def print_achievement_summary(runs: dict[str, TrainingLoggerCallback]) -> None:
    """Print, per run, which achievements were ever unlocked and how often."""
    for label, cb in runs.items():
        eps = cb.episode_achievements
        if not eps:
            continue
        n = len(eps)
        totals: dict[str, int] = {}
        for e in eps:
            for name, count in e.items():
                if count > 0:
                    totals[name] = totals.get(name, 0) + 1
        unlocked = sorted(totals.items(), key=lambda kv: -kv[1])

        print(f"\n[{label}] achievements over {n} episodes "
              f"({len(unlocked)}/22 ever unlocked):")
        if not unlocked:
            print("    (none — agent never completed any achievement)")
        for name, hits in unlocked:
            print(f"    {name:<22} {hits / n * 100:5.1f}%  ({hits}/{n} episodes)")


# ---------------------------------------------------------------------------
# Factory: build the vectorised env
# ---------------------------------------------------------------------------


def make_env(seed: int = 0):
    # SB3's DummyVecEnv expects a list of FUNCTIONS that each build one env
    # (not the envs themselves), so the vec env can construct them itself.
    # That's why this returns the inner `_init` instead of an env.
    def _init():
        env = CrafterGymnasiumEnv(max_episode_steps=10_000)
        # Monitor wraps the env to record episode reward/length -> "episode" info.
        env = Monitor(env)
        return env

    return _init


def build_vec_env(n_envs: int = 4, seed: int = 0) -> VecTransposeImage:
    # DummyVecEnv stacks n_envs copies into ONE "vectorised" env: each step()
    # advances all of them at once, so the agent gathers data from several
    # games in parallel (more, less-correlated samples per update).
    vec = DummyVecEnv([make_env(seed + i) for i in range(n_envs)])
    # CNNs in PyTorch want channels first; envs give channels last, so transpose.
    vec = VecTransposeImage(vec)  # (H, W, C) -> (C, H, W) for CNN
    return vec


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    run_name: str,
    total_timesteps: int,
    # PPO hyperparameters
    lr: float = 2.5e-4,  # optimizer learning rate — how big a step the network takes per weight update
    n_steps: int = 128,  # steps collected per env before each update (rollout length)
    batch_size: int = 256,  # minibatch size when learning from the collected rollout
    n_epochs: int = 4,  # how many passes PPO makes over the same data per update
    gamma: float = 0.99,  # discount factor — how much future rewards matter (closer to 1 = more far-sighted)
    gae_lambda: float = 0.95,  # GAE parameter — bias/variance trade-off when estimating the advantage
    clip_range: float = 0.2,  # PPO clip — limits how much the policy can change per step, keeps training stable
    ent_coef: float = 0.001,  # entropy coefficient — encourages exploration (higher = more random actions)
    vf_coef: float = 0.5,  # weight of the value-function loss in the total loss
    max_grad_norm: float = 0.5,  # gradient-norm clipping — guards against exploding gradients
    n_envs: int = 4,  # number of parallel environments collecting data
    seed: int = 42,
    log_dir: Path = Path("logs"),
) -> tuple[PPO, TrainingLoggerCallback]:
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Run: {run_name}")
    print(f"  lr={lr}  ent_coef={ent_coef}  clip_range={clip_range}")
    print(f"  n_envs={n_envs}  total_timesteps={total_timesteps}")
    print(f"{'=' * 60}\n")

    # Separate envs for training and evaluation so eval scores aren't biased
    # by the exact episodes the agent trained on (different seed).
    train_env = build_vec_env(n_envs=n_envs, seed=seed)
    eval_env = build_vec_env(n_envs=1, seed=seed + 99)

    # Create the PPO agent. The constructor wires together:
    #   - "CnnPolicy": the network architecture (CNN, because obs are images).
    #     Other options: "MlpPolicy" for vector obs, "MultiInputPolicy" for dicts.
    #   - env: where it collects experience.
    #   - the hyperparameters defined above.
    # SB3 builds the actor + critic networks automatically from policy + spaces.
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,  # 1 = print the training table to the console each update
        seed=seed,
        tensorboard_log=str(log_dir / "tb"),  # also log metrics for TensorBoard
    )

    # Our custom callback that records episode stats for the plots.
    logger_cb = TrainingLoggerCallback()
    # Built-in callback: every eval_freq steps it runs the current policy on
    # eval_env and saves whichever model scores best to best_model_save_path.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / run_name),
        log_path=str(log_dir / run_name),
        # In a vec env one "step" advances all envs, so divide by n_envs to
        # evaluate roughly every 5_000 real environment steps.
        eval_freq=max(5_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,  # use the learned policy without random exploration
        render=False,
    )

    t0 = time.time()
    # learn() runs the whole training loop: collect rollouts -> update -> repeat
    # until total_timesteps env steps have been taken. Callbacks fire inside it.
    model.learn(
        total_timesteps=total_timesteps,
        callback=[logger_cb, eval_cb],
        tb_log_name=run_name,
        reset_num_timesteps=True,  # start the step counter from 0 for this run
    )
    elapsed = time.time() - t0
    print(f"\n[{run_name}] training done in {elapsed / 60:.1f} min")

    # Free the environments (closes underlying game instances).
    train_env.close()
    eval_env.close()

    return model, logger_cb


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------


def final_eval(model: PPO, n_episodes: int = 20, seed: int = 42) -> dict:
    # After training, measure how good the final model is on fresh episodes.
    eval_env = build_vec_env(n_envs=1, seed=seed + 200)
    # evaluate_policy runs the model for n_eval_episodes and returns the
    # mean and std of total reward — the headline performance number.
    mean_r, std_r = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        deterministic=True,
    )
    eval_env.close()
    print(f"  mean_reward={mean_r:.3f} +/- {std_r:.3f}  (over {n_episodes} episodes)")
    return {"mean_reward": mean_r, "std_reward": std_r}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="PPO Crafter training script")
    p.add_argument(
        "--run",
        type=str,
        default="baseline",
        help="Run name tag (e.g. 'baseline', 'experiment')",
    )
    p.add_argument(
        "--timesteps", type=int, default=50000, help="Total environment timesteps"
    )
    p.add_argument(
        "--n-envs", type=int, default=4, help="Number of parallel environments"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=str, default="logs")

    # PPO hyperparams (override for experiments)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    log_dir = Path(args.log_dir)

    model, cb = train(
        run_name=args.run,
        total_timesteps=args.timesteps,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        n_envs=args.n_envs,
        seed=args.seed,
        log_dir=log_dir,
    )

    print("\n--- Final evaluation ---")
    final_eval(model, n_episodes=20, seed=args.seed)

    # Print + save all curves/summaries for this run.
    runs = {args.run: cb}
    print_achievement_summary(runs)        # text summary to the console
    plot_results(runs, out_dir=log_dir)    # reward / survival / achievements curves
    plot_achievements(runs, out_dir=log_dir)  # per-achievement success-rate bars

    # Save model: SB3 writes a single .zip with weights + hyperparameters.
    # Reload later with PPO.load(path) and act via model.predict(obs).
    model_path = log_dir / f"{args.run}_final"
    model.save(str(model_path))
    print(f"[save] model -> {model_path}.zip")
