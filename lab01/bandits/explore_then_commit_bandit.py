from environment.environment import BanditLearner

import numpy as np
import random
import math

class ExploreThenCommitLearner(BanditLearner):
    def __init__(self, m: int = 5):
        self.m = m
        self.name = "explore-then-commit"
        self.color = "red"

        self.arms: list[str] = []
        self.counts: dict[str, int] = {}
        self.expected_values: dict[str, float] = {}
        self.round: int = 0
        self.k: int = 0

    #
    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.counts = {arm: 0 for arm in arms}
        self.expected_values = {arm: 0.0 for arm in arms}
        self.round = 0
        self.k = len(arms)
    #

    def pick_arm(self) -> str:
        if self.round  <= self.m * self.k:
            return self.arms[self.round % self.k]

        max_expected_value = max(self.expected_values.values())
        best_arms = [arm for arm in self.arms if self.expected_values[arm] == max_expected_value]
        return random.choice(best_arms)
    #

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.counts[arm] = 1 + self.counts[arm]
        self.expected_values[arm] = self.expected_values[arm] + (reward - self.expected_values[arm]) / self.counts[arm]
        self.round += 1
    #
# end class