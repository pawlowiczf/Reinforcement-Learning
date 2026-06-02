"""
Visualise a trained PPO agent playing Crafter.

Loads a saved model, runs it in the environment, and either shows the game
live in a window or saves it to an MP4 video.

Usage:
    # Watch the agent live (close the window or press 'q' to stop)
    python play.py --model logs/baseline_final

    # Use the best checkpoint saved by EvalCallback instead
    python play.py --model logs/baseline/best_model

    # Save a video instead of showing a window
    python play.py --model logs/baseline_final --save play.mp4 --episodes 3

    # Watch the untrained / random-ish behaviour for comparison
    python play.py --model logs/baseline_final --stochastic
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import PPO

# Reuse the exact same env definition the model was trained on.
from train import CrafterGymnasiumEnv

# Size (in pixels) of the rendered window/video frame. The agent still "sees"
# the small 64x64 image; this is just an upscaled view for our eyes.
RENDER_SIZE = (512, 512)


def parse_args():
    p = argparse.ArgumentParser(description="Visualise a trained PPO Crafter agent")
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the saved model (without .zip), e.g. logs/baseline_final",
    )
    p.add_argument("--episodes", type=int, default=5, help="How many episodes to play")
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="If given, write an MP4 here instead of opening a live window",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of taking the best one",
    )
    p.add_argument("--fps", type=int, default=15, help="Playback / video frame rate")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()

    # Load the trained agent. PPO.load reads the .zip (weights + hyperparams).
    model = PPO.load(args.model)
    print(f"[play] loaded model from {args.model}")

    # A single, plain env (no vec wrappers) so we control reset/render directly.
    env = CrafterGymnasiumEnv(max_episode_steps=10_000)

    # Set up the video writer up front if we're saving instead of displaying.
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, RENDER_SIZE)

    deterministic = not args.stochastic

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            # The model maps the observation to an action. The CNN inside expects
            # channels-first, but predict() handles a single (H, W, C) image too.
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            ep_reward += reward
            steps += 1

            # Grab a big RGB frame of the current game state for display/saving.
            frame = env.render(size=RENDER_SIZE)  # (H, W, 3) RGB uint8
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR

            if writer is not None:
                writer.write(frame_bgr)
            else:
                cv2.imshow("Crafter - PPO agent", frame_bgr)
                # waitKey controls playback speed; 'q' quits early.
                if cv2.waitKey(int(1000 / args.fps)) & 0xFF == ord("q"):
                    done = True

        print(f"[play] episode {ep + 1}/{args.episodes}: reward={ep_reward:.2f}  steps={steps}")

    env.close()
    if writer is not None:
        writer.release()
        print(f"[play] saved video -> {args.save}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
