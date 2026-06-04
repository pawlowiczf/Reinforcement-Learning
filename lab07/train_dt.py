import argparse
import os
from pathlib import Path

# collect_minari.py stored the datasets in a LOCAL folder, so point Minari there
# BEFORE importing it - otherwise load_dataset() looks in the global ~/.minari.
os.environ["MINARI_DATASETS_PATH"] = str(
    (Path(__file__).parent / "minari_datasets").resolve()
)

import minari

from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries

# Defaults: which expert demonstrations to train on, and where to save the DT.
# Override per run with --dataset-id / --model-path.
DATASET_ID = "lunarlander/ppo_beginner_100k-v0"
MODEL_PATH = "output/dt/ppo_beginner_100k-v0.pth"


def parse_args():
    p = argparse.ArgumentParser(description="Train a Decision Transformer with nanoDT")
    p.add_argument(
        "--dataset-id", default=DATASET_ID, help="Minari dataset to train on"
    )
    p.add_argument(
        "--model-path", default=MODEL_PATH, help="Where to save the trained DT (.pth)"
    )
    return p.parse_args()


def train_dt(dataset_id, model_path):
    seed = 1234
    seed_libraries(seed)
    minari_dataset = minari.load_dataset(dataset_id)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # This nanodt build doesn't resolve "auto"/"mps"; pass a real torch device.
    # No CUDA here, so train on CPU.
    dt_agent = NanoDTAgent(device="cpu")

    # reward_scale divides returns-to-go fed to the model. LunarLander returns
    # are ~200-300, so a smaller scale keeps them sane.
    # max_iters: trainer default is 100k, far too slow on CPU; keep it modest.
    dt_agent.learn(
        minari_dataset,
        reward_scale=100.0,
        max_iters=200_000,
        warmup_iters=1_000,
    )
    dt_agent.save(model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    args = parse_args()
    train_dt(args.dataset_id, args.model_path)

# uv run python train_dt.py --dataset-id lunarlander/ppo_beginner_100k-v0 --model-path output/dt/ppo_beginner_100k-v0.pth
# uv run python train_dt.py --dataset-id lunarlander/ppo_intermediate_400k-v0 --model-path output/dt/ppo_intermediate_400k-v0.pth
# uv run python train_dt.py --dataset-id lunarlander/ppo_expert_1000k-v0 --model-path output/dt/ppo_expert_100k-v0.pth