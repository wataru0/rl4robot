import statistics as stats
from dataclasses import dataclass
from typing import Final, List, Tuple

import numpy as np
import tqdm

from rl4robot.agents import Agent
from rl4robot.envs import Env


@dataclass(frozen=True)
class EvaluatingResult:
    episode_lengths: List[int]
    episode_returns: List[float]

    def episode_length_mean(self) -> float:
        return stats.mean(self.episode_lengths)

    def episode_length_std(self) -> float:
        return stats.stdev(self.episode_lengths)

    def episode_return_mean(self) -> float:
        return stats.mean(self.episode_returns)

    def episode_return_std(self) -> float:
        return stats.stdev(self.episode_returns)


class EvaluatingLoop:
    env: Final[Env]
    agent: Final[Agent]
    num_episodes: Final[int]

    def __init__(self, env: Env, agent: Agent, num_episodes: int) -> None:
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes

    def run(self) -> EvaluatingResult:
        """評価ループを実行する。"""

        episode_lengths: List[int] = []
        episode_returns: List[float] = []

        for _ in tqdm.tqdm(range(self.num_episodes)):
            episode_length, episode_return = self._collect_episode()

            episode_lengths.append(episode_length)
            episode_returns.append(episode_return)

        return EvaluatingResult(
            episode_lengths=episode_lengths, episode_returns=episode_returns
        )

    def _collect_episode(self) -> Tuple[int, float]:
        """１エピソード評価する。

        Returns
        -------
        Tuple[int, float]
            エピソード長とエピソード収益
        """

        action_low, action_high = self.env.spec.action_range
        episode_done = False
        episode_length = 0
        episode_return = 0.0

        observation = self.env.reset()

        while not episode_done:
            action = self.agent.act(observation)
            action = np.clip(action, action_low, action_high)
            env_step = self.env.step(action)

            observation = env_step.observation
            episode_done = env_step.episode_done
            episode_length += 1
            episode_return += env_step.reward

        return episode_length, episode_return
