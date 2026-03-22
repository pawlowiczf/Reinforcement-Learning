import matplotlib.pyplot as plt
import numpy as np

from abc import abstractmethod
from itertools import accumulate
from collections import defaultdict
import random
import math
from typing import Protocol

# environment
class KArmedBandit(Protocol):
    @abstractmethod
    def arms(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: str) -> float:
        raise NotImplementedError

# strategy
class BanditLearner(Protocol):
    name: str
    color: str

    @abstractmethod
    def reset(self, arms: list[str], time_steps: int):
        raise NotImplementedError

    @abstractmethod
    def pick_arm(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def acknowledge_reward(self, arm: str, reward: float) -> None:
        pass


class BanditProblem:
    def __init__(self, time_steps: int, bandit: KArmedBandit, learner: BanditLearner):
        self.time_steps: int = time_steps
        self.bandit: KArmedBandit = bandit
        self.learner: BanditLearner = learner
        self.learner.reset(self.bandit.arms(), self.time_steps)

    def run(self) -> list[float]:
        rewards = []
        for _ in range(self.time_steps):
            arm = self.learner.pick_arm()
            reward = self.bandit.reward(arm)
            self.learner.acknowledge_reward(arm, reward)
            rewards.append(reward)
        return rewards


POTENTIAL_HITS = {
    "In Praise of Dreams": 0.8,
    "We Built This City": 0.9,
    "April Showers": 0.5,
    "Twenty Four Hours": 0.3,
    "Dirge for November": 0.1,
}


class TopHitBandit(KArmedBandit):
    def __init__(self, potential_hits: dict[str, float]):
        self.potential_hits: dict[str, float] = potential_hits

    def arms(self) -> list[str]:
        return list(self.potential_hits)

    def reward(self, arm: str) -> float:
        thumb_up_probability = self.potential_hits[arm]
        return 1.0 if random.random() <= thumb_up_probability else 0.0


class RandomLearner(BanditLearner):
    def __init__(self):
        self.name = "random"
        self.color = "black"
        self.arms: list[str] = []

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms

    def pick_arm(self) -> str:
        return random.choice(self.arms)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        pass

class EGreedyLearner(BanditLearner):
    def __init__(self, eps: float = 0.1):
        self.name = "e-greedy"
        self.color = "blue"
        self.eps: float = eps

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

        # return max(self.expected_values, key=self.expected_values.get)
        max_expected_value = max(self.expected_values.values())
        best_arms = [arm for arm in self.arms if self.expected_values[arm] == max_expected_value]
        return random.choice(best_arms)
    #

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.counts[arm] = 1 + self.counts[arm]
        self.expected_values[arm] = self.expected_values[arm] + (reward - self.expected_values[arm]) / self.counts[arm]
    #
# end class

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

class UCB(BanditLearner):
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

        ucb = [
            (
                arm,
                self.expected_values[arm] + self.c * np.sqrt(np.log(self.round) / self.counts[arm])
            )
            for arm in self.arms
        ]

        max_ucb = max(val for _, val in ucb)
        best_arms = [arm for arm, val in ucb if val == max_ucb]
        return random.choice(best_arms)
    #

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.counts[arm] = 1 + self.counts[arm]
        self.expected_values[arm] = self.expected_values[arm] + (reward - self.expected_values[arm]) / self.counts[arm]
        self.round += 1
    #
# end class

class ThompsonSampling(BanditLearner):
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

TIME_STEPS = 1000
TRIALS_PER_LEARNER = 50

def evaluate_learner(learner: BanditLearner) -> None:
    runs_results = []
    for _ in range(TRIALS_PER_LEARNER):
        bandit = TopHitBandit(POTENTIAL_HITS)
        problem = BanditProblem(time_steps=TIME_STEPS, bandit=bandit, learner=learner)
        rewards = problem.run()
        accumulated_rewards = list(accumulate(rewards))
        runs_results.append(accumulated_rewards)

    runs_results = np.array(runs_results)
    mean_accumulated_rewards = np.mean(runs_results, axis=0)
    std_accumulated_rewards = np.std(runs_results, axis=0)
    plt.plot(mean_accumulated_rewards, label=learner.name, color=learner.color)
    plt.fill_between(
        range(len(mean_accumulated_rewards)),
        mean_accumulated_rewards - std_accumulated_rewards,
        mean_accumulated_rewards + std_accumulated_rewards,
        color=learner.color,
        alpha=0.2,
    )
#

def main():
    learners = [
        RandomLearner(),
        EGreedyLearner(),
        ExploreThenCommitLearner(m = 25),
        UCB(c=0.1),
        ThompsonSampling()
    ]
    for learner in learners:
        evaluate_learner(learner)

    plt.xlabel('Czas')
    plt.ylabel('Suma uzyskanych nagród')
    plt.xlim(0, TIME_STEPS)
    plt.ylim(0, TIME_STEPS)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
