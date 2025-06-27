from __future__ import annotations

from typing import TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env import VecEnv

from drone_grid_env.wrappers import EpisodeInfoBufferWrapper

if TYPE_CHECKING:
    from typing import Any


KEYS_TO_IGNORE = {"episode", "terminal_observation", "TimeLimit.truncated"}


class EnvNotWrappedException(Exception):
    pass


class EpisodeInfoLoggerCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        assert self.training_env is not None

        if isinstance(self.training_env, VecEnv):
            wrapped = all(self.training_env.env_is_wrapped(EpisodeInfoBufferWrapper))
        else:
            wrapped = is_wrapped(self.training_env, EpisodeInfoBufferWrapper)

        if not wrapped:
            raise EnvNotWrappedException(
                "Environment is not wrapped with EpisodeInfoBufferWrapper which is needed for EpisodeInfoLoggerCallback!"
            )

    def _on_step(self) -> bool:
        assert self.training_env is not None

        # Buffered info only returns the info dict when the episode is terminated or truncated, otherwise it will
        # return None.
        if isinstance(self.training_env, VecEnv):
            infos: list[dict[str, Any]] = [_info for _info in self.training_env.get_attr("buffered_info") if _info is not None]
        else:
            infos = [info] if (info := getattr(self.training_env, "buffered_info")) is not None else []

        for _info in infos:
            for k, v in _info.items():
                if k not in KEYS_TO_IGNORE:
                    self.logger.record(f"train/{k}", v)

        return True
