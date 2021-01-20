import statistics as stats
import time
from typing import Final, List, Optional, Tuple, Union

import numpy as np

from rl4robot.agents import Trainer
from rl4robot.envs import Env
from rl4robot.types import ObservationArray

from .loggers import Logger

__all__ = [
    "TrainingLoop",
]


def _safety_mean(xs: Union[List[int], List[float]]) -> Optional[float]:
    if len(xs) == 0:
        return None
    return stats.fmean(xs)


class TrainingLoop:
    env: Final[Env]
    trainer: Final[Trainer]
    num_steps: Final[int]
    logger: Final[Optional[Logger]]
    global_step: int
    global_iter: int

    _start_time: float
    _iter_start_time: float
    _iter_start_global_steps: float
    _observation: ObservationArray
    _episode_length = 0
    _episode_return = 0.0

    def __init__(
        self,
        env: Env,
        trainer: Trainer,
        num_steps: int,
        logger: Optional[Logger] = None,
    ) -> None:
        self.env = env
        self.trainer = trainer
        self.num_steps = num_steps
        self.logger = logger

        self.global_step = 0
        self.global_iter = 0

    def run(self) -> None:
        """訓練ループ"""

        self._start_time = time.time()

        self._observation = self.env.reset()
        self._episode_length = 0
        self._episode_return = 0.0

        while self.global_step < self.num_steps:
            self._iter_start_time = time.time()
            self._iter_start_global_steps = self.global_step

            self.global_iter += 1

            episode_length_mean, episode_return_mean = self._collect()

            self.trainer.update()

            if self.logger is not None:
                self._record_time_log()
                self._record_stats_log(
                    episode_length_mean, episode_return_mean
                )
                self.trainer.record_log(self.logger)
                self.logger.dump(self.global_step)

    def _collect(self) -> Tuple[Optional[float], Optional[float]]:
        """サンプルを収集

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            平均エピソード長と平均エピソード収益
        """

        action_low, action_high = self.env.spec.action_range
        episode_lengths: List[int] = []
        episode_returns: List[float] = []

        while not self.trainer.can_update():
            self.global_step += 1

            action = self.trainer.act(self._observation)
            action = np.clip(action, action_low, action_high)

            env_step = self.env.step(action)

            self._episode_length += 1
            self._episode_return += env_step.reward

            self.trainer.observe_result(env_step)

            if env_step.episode_done:
                episode_lengths.append(self._episode_length)
                episode_returns.append(self._episode_return)
                self._observation = self.env.reset()
                self._episode_length = 0
                self._episode_return = 0.0
            else:
                self._observation = env_step.observation

        return _safety_mean(episode_lengths), _safety_mean(episode_returns)

    def _record_time_log(self) -> None:
        """時間に関するログを記録"""

        if self.logger is None:
            return

        progress = self.global_step / self.num_steps
        now = time.time()
        elapsed_secs = now - self._start_time
        iter_elapsed_secs = now - self._iter_start_time
        num_iter_steps = self.global_step - self._iter_start_global_steps
        fps = num_iter_steps / iter_elapsed_secs

        self.logger.record_from_dict(
            {
                "time/global_step": self.global_step,
                "time/global_iter": self.global_iter,
                "time/progress": progress,
                "time/elapsed_secs": elapsed_secs,
                "time/fps": fps,
            }
        )

    def _record_stats_log(
        self,
        episode_length_mean: Optional[float],
        episode_return_mean: Optional[float],
    ) -> None:
        """エピソードの統計に関するログを記録"""

        if self.logger is None:
            return

        if episode_length_mean is None or episode_return_mean is None:
            return

        self.logger.record_from_dict(
            {
                "stats/episode_length": episode_length_mean,
                "stats/episode_return": episode_return_mean,
            }
        )
