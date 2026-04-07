from environment import BanditLearner

import numpy as np
import random
import math

class ThompsonSamplingLearner(BanditLearner):
    def __init__(self):
        self.name = "thompson-sampling"
        self.color = "magenta"

        self.arms: list[str] = []
        self.counts: dict[str, int] = {}

        self.alphas: dict[str, float] = {}
        self.betas: dict[str, float] = {}

    #
    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.counts = {arm: 0 for arm in arms}

        self.alphas = {arm: 1 for arm in arms}
        self.betas = {arm: 1 for arm in arms}
    #

    def pick_arm(self) -> str:
        samples = {arm: np.random.beta(self.alphas[arm], self.betas[arm]) for arm in self.arms}
        return max(samples, key=samples.get)
    #

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.counts[arm] = 1 + self.counts[arm]
        if reward == 1.0:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1
    #
# end class