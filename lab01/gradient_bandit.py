from environment import BanditLearner

import numpy as np
import random
import math

class GradientLearner(BanditLearner):
    def __init__(self, alpha: float = 0.1):
        self.name = "gradient"
        self.color = "red"
        self.alpha: float = alpha

        self.arms: list[str] = []
        self.preferences: dict[str, float] = {}
        self.probabilities: dict[str, float] = {}

        self.avg_reward = 0.0
        self.round = 0
    #

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.preferences = {arm: 0.0 for arm in arms} # H
        self.probabilities = {arm: 1.0 / len(arms) for arm in arms}

        self.avg_reward = 0.0
        self.round = 0
    #

    def softmax(self):
        values = np.array([self.preferences[a] for a in self.arms])
        exp_values = np.exp(values - np.max(values))
        probs = exp_values / np.sum(exp_values) # guarantees numerical stability

        for arm, p in zip(self.arms, probs):
            self.probabilities[arm] = p
    #

    def pick_arm(self) -> str:
        self.softmax()

        arms = list(self.probabilities.keys())
        probs = list(self.probabilities.values())

        return np.random.choice(arms, p=probs)
    #

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.round += 1

        self.avg_reward += (reward - self.avg_reward) / self.round

        self.softmax()

        for a in self.arms:
            if a == arm:
                self.preferences[a] += self.alpha * (reward - self.avg_reward) * (1 - self.probabilities[a])
            else:
                self.preferences[a] -= self.alpha * (reward - self.avg_reward) * self.probabilities[a]
    #
# end class