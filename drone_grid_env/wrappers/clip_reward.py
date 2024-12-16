from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gymnasium import RewardWrapper
from gymnasium.core import ActType, ObsType

if TYPE_CHECKING:
    from typing import SupportsFloat

    from gymnasium import Env


class ClipRewardWrapper(RewardWrapper[ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType], low: int = -1, high: int = 1) -> None:
        super().__init__(env)

        self._low = low
        self._high = high

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        return float(np.clip([reward], self._low, self._high)[0])
