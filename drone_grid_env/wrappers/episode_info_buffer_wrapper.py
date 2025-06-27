from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium import Wrapper
from gymnasium.core import ActType, ObsType

if TYPE_CHECKING:
    from typing import Any, SupportsFloat

    from gymnasium import Env


class EpisodeInfoBufferWrapper(Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType]) -> None:
        super().__init__(env)

        self._last_info: dict[str, Any] | None = None

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._last_info = None
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self._last_info = info

        return obs, reward, terminated, truncated, info

    @property
    def buffered_info(self) -> dict[str, Any] | None:
        return self._last_info
