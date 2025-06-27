from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from gymnasium import ObservationWrapper, spaces
from gymnasium.core import ActType, Env

if TYPE_CHECKING:
    from typing import Any, SupportsFloat


class FlattenWrapper(ObservationWrapper[NDArray[np.uint8], ActType, dict[str, NDArray[np.uint8]]]):
    def __init__(self, env: Env[dict[str, NDArray[np.uint8]], ActType]):
        super().__init__(env)

        assert isinstance(self.env.observation_space, spaces.Dict), "Wrapper should have DictSpace as input!"

        obs_sample = self.env.observation_space.sample()
        total_size = 0
        for v in obs_sample.values():
            total_size += v.size

        self.observation_space = spaces.Box(0, 255, (total_size,), dtype=np.uint8)

    def observation(self, observation: dict[str, NDArray[np.uint8]]) -> NDArray[np.uint8]:
        array_list = []
        for v in observation.values():
            array_list.append(v.ravel())

        return np.concatenate(array_list)
