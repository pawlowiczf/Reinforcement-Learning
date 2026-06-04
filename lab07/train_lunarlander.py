"""
Train / evaluate a PPO expert on LunarLander-v3.

# train from scratch -> models/ppo_LunarLander-v3_500k_s0.zip + tb/...
uv run python train_lunarlander.py

# custom name + shorter run
uv run python train_lunarlander.py --run-name expert --timesteps 200000

# skip training: load a saved run and just watch it play
uv run python train_lunarlander.py --load --run-name expert --watch

# ...and also print the eval score
uv run python train_lunarlander.py --load --run-name expert --watch --eval
"""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def default_run_name(env_id, timesteps, seed):
    # e.g. "ppo_LunarLander-v3_500k_s0" — readable and unique per config.
    return f"ppo_{env_id}_{timesteps // 1000}k_s{seed}"


def make_env(env_id, render_mode=None):
    # Monitor records per-episode reward/length so SB3 can report progress.
    return Monitor(gym.make(env_id, render_mode=render_mode))


def train(env_id, timesteps, model_path, seed, tb_dir, run_name, n_envs):
    # Vectorised env: n_envs copies running in parallel so PPO gathers more,
    # less-correlated data per update (faster + more stable than a single env).
    # make_vec_env wraps each copy in Monitor for us automatically.
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed)

    # PPO agent. "MlpPolicy" = a small fully-connected net, the right choice
    # when observations are vectors (not images). Defaults are well-tuned for
    # LunarLander, so we keep the flow minimal and only set what matters.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,  # print the training table each update
        seed=seed,
        tensorboard_log=tb_dir,  # parent dir; the run name becomes a subfolder
    )

    # The whole training loop: collect rollouts -> update -> repeat.
    # tb_log_name puts this run's logs under <tb_dir>/<run_name>_1.
    model.learn(total_timesteps=timesteps, progress_bar=True, tb_log_name=run_name)

    # Save weights + hyperparameters to a single .zip (path gets a .zip suffix).
    model.save(model_path)
    print(f"[save] model -> {model_path}.zip")
    print(f"[save] tb logs -> {Path(tb_dir) / (run_name + '_1')}")
    env.close()
    return model


def evaluate(model, env_id, n_episodes=20):
    # Measure the trained policy on fresh episodes (no exploration noise).
    eval_env = make_env(env_id)
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=n_episodes, deterministic=True
    )
    eval_env.close()
    print(f"[eval] mean_reward={mean_r:.1f} +/- {std_r:.1f} over {n_episodes} episodes")
    print(
        "[eval] solved!" if mean_r >= 200 else "[eval] not solved yet (target >= 200)"
    )


def watch(model, env_id, n_episodes=3):
    # Open a window and watch the agent play.
    env = make_env(env_id, render_mode="human")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        print(f"[watch] episode {ep + 1}: reward={total:.1f}")
    env.close()


def parse_args():
    p = argparse.ArgumentParser(description="PPO expert for LunarLander")
    p.add_argument("--env", default="LunarLanderContinuous-v3", help="Gymnasium env id")
    p.add_argument(
        "--run-name",
        default=None,
        help="Name for this run (default: auto from env/timesteps/seed). "
        "Used as the model filename and TensorBoard subfolder.",
    )
    p.add_argument(
        "--models-dir", default="models", help="Folder where models are saved/loaded"
    )
    p.add_argument("--tb-dir", default="tb", help="Folder for TensorBoard logs")
    p.add_argument(
        "--timesteps", type=int, default=500_000, help="Total training timesteps"
    )
    p.add_argument(
        "--n-envs", type=int, default=8, help="Parallel environments for training"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--eval-episodes", type=int, default=20, help="Episodes for evaluation"
    )
    p.add_argument(
        "--load",
        action="store_true",
        help="Skip training; load the run from --models-dir and just evaluate/watch",
    )
    p.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation (mean reward over --eval-episodes). "
        "Always on after training; with --load it's opt-in.",
    )
    p.add_argument(
        "--watch", action="store_true", help="Render a few episodes in a window"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve the run name and the on-disk model path: <models-dir>/<run-name>.
    run_name = args.run_name or default_run_name(args.env, args.timesteps, args.seed)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)  # SB3 won't create it for us
    model_path = str(models_dir / run_name)

    if args.load:
        # Reuse a previously trained model instead of training again.
        model = PPO.load(model_path)
        print(f"[load] model <- {model_path}.zip")
    else:
        model = train(
            env_id=args.env,
            timesteps=args.timesteps,
            model_path=model_path,
            seed=args.seed,
            tb_dir=args.tb_dir,
            run_name=run_name,
            n_envs=args.n_envs,
        )

    # Always evaluate after fresh training; when just loading, evaluate only
    # if asked (so --load --watch can open the window without the eval wait).
    if args.eval or not args.load:
        evaluate(model, args.env, n_episodes=args.eval_episodes)

    if args.watch:
        watch(model, args.env)

# uv run python .\train_lunarlander.py --load --run-name "ppo_LunarLanderContinuous-v3_100k_s0" --watch
# uv run python .\train_lunarlander.py --load --run-name "ppo_LunarLanderContinuous-v3_1000k_s0" --watch
