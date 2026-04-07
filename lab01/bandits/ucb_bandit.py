from environment.environment import BanditLearner

import numpy as np
import random
import math

class UCBLearner(BanditLearner):
    def __init__(self, c: int = 0.5):
        self.c = c
        self.name = "upper-confidence-bound"
        self.color = "orange"

        self.arms: list[str] = []
        self.counts: dict[str, int] = {}
        self.expected_values: dict[str, float] = {}
        self.round: int = 0

    #
    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.counts = {arm: 0 for arm in arms}
        self.expected_values = {arm: 0.0 for arm in arms}
        self.round = 0
    #

    def pick_arm(self) -> str:
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm
        #

        ucb = {arm: self.expected_values[arm] + self.c * np.sqrt(np.log(self.round) / self.counts[arm]) for arm in self.arms}
        max_expected_value = max(ucb.values())
        best_arms = [arm for arm, val in ucb.items() if val == max_expected_value]
        return random.choice(best_arms)
    #

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.counts[arm] = 1 + self.counts[arm]
        self.expected_values[arm] = self.expected_values[arm] + (reward - self.expected_values[arm]) / self.counts[arm]
        self.round += 1
    #
# end class