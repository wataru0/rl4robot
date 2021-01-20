from typing import Protocol

from rl4robot.types import ActionArray, ObservationArray

__all__ = [
    "Agent",
]


class Agent(Protocol):
    def act(self, observation: ObservationArray) -> ActionArray:
        """行動を決定"""

        ...
