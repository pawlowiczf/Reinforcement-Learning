from environment.environment import BanditLearner

import numpy as np
import random
import math

class EGreedyAlphaLearner(BanditLearner):
    def __init__(self, eps: float = 0.1, alpha: float = None):
        self.name = "e-greedy-alpha"
        self.color = "blue"
        self.eps: float = eps
        self.alpha = alpha

        self.arms: list[str] = []
        self.counts: dict[str, int] = {}
        self.expected_values: dict[str, float] = {}
    #

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.counts = {arm: 0 for arm in arms}
        self.expected_values = {arm: 0.0 for arm in arms}
    #

    def pick_arm(self) -> str:
        choice = random.random()

        if choice < self.eps:
            return random.choice(self.arms)

        max_expected_value = max(self.expected_values.values())
        best_arms = [arm for arm in self.arms if self.expected_values[arm] == max_expected_value]
        return random.choice(best_arms)
    #

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.counts[arm] += 1
        self.expected_values[arm] += self.alpha * (reward - self.expected_values[arm])
    #
#