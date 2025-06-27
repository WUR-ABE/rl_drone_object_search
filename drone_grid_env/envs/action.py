from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING
from yaml import YAMLError, safe_load

import numpy as np

from gymnasium.spaces import Discrete

from drone_grid_env import logger

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from gymnasium.spaces import Space


class ActionSpace(IntEnum):
    FLY_WEST = 0
    FLY_EAST = 1
    FLY_NORTH = 2
    FLY_SOUTH = 3
    LAND = 4
    FLY_SOUTH_EAST = 5
    FLY_SOUTH_WEST = 6
    FLY_NORTH_EAST = 7
    FLY_NORTH_WEST = 8

    @staticmethod
    def get_space(config_file: Path) -> Space[int]:
        with config_file.open("r") as cs:
            try:
                data: dict[str, Any] = safe_load(cs)["action_space"]

                num_actions = 4

                if data["allow_diagonals"]:
                    num_actions += 4

                if data["land_action"]:
                    num_actions += 1

                return Discrete(num_actions)  # type: ignore[return-value]

            except (YAMLError, KeyError) as e:
                logger.error(f"Could not load config file '{config_file.name}': {e}")
                raise

    @staticmethod
    def create_keys2action(config_file: Path) -> dict[tuple[int, ...], int | None]:
        """
        Function to return a keymap to work with the OpenAI Gym play function in
        gym.utils.play.

        Source: https://github.com/openai/gym/blob/master/gym/utils/play.py

        :returns: Keymap mapping (multiple) keys to actions.
        """
        with config_file.open("r") as cs:
            try:
                data: dict[str, Any] = safe_load(cs)["action_space"]

                key2action: dict[tuple[int, ...], int | None] = {
                    (): None,
                    (ord("w"),): ActionSpace.FLY_NORTH,
                    (ord("s"),): ActionSpace.FLY_SOUTH,
                    (ord("a"),): ActionSpace.FLY_WEST,
                    (ord("d"),): ActionSpace.FLY_EAST,
                }

                if data["land_action"]:
                    key2action.update(
                        {
                            (ord("l"),): ActionSpace.LAND,
                        }
                    )

                if data["allow_diagonals"]:
                    key2action.update(
                        {
                            (ord("l"),): ActionSpace.LAND,
                            (ord("q"),): ActionSpace.FLY_NORTH_WEST,
                            (ord("e"),): ActionSpace.FLY_NORTH_EAST,
                            (ord("z"),): ActionSpace.FLY_SOUTH_WEST,
                            (ord("x"),): ActionSpace.FLY_SOUTH_EAST,
                        }
                    )

                return key2action

            except (YAMLError, KeyError) as e:  # pragma: no cover
                logger.error(f"Could not load config file '{config_file.name}': {e}")
                raise

    @staticmethod
    def is_movement_action(action: int | np.integer[Any]) -> bool:
        return action in (
            ActionSpace.FLY_WEST,
            ActionSpace.FLY_EAST,
            ActionSpace.FLY_NORTH,
            ActionSpace.FLY_SOUTH,
            ActionSpace.FLY_SOUTH_EAST,
            ActionSpace.FLY_SOUTH_WEST,
            ActionSpace.FLY_NORTH_EAST,
            ActionSpace.FLY_NORTH_WEST,
        )

    @classmethod
    def get_actions_to_flight_map(cls) -> dict[ActionSpace, NDArray[np.int16]]:
        return {
            ActionSpace.FLY_SOUTH: np.array([1, 0], dtype=np.int16),
            ActionSpace.FLY_NORTH: np.array([-1, 0], dtype=np.int16),
            ActionSpace.FLY_EAST: np.array([0, 1], dtype=np.int16),
            ActionSpace.FLY_WEST: np.array([0, -1], dtype=np.int16),
            ActionSpace.FLY_SOUTH_EAST: np.array([1, 1], dtype=np.int16),
            ActionSpace.FLY_SOUTH_WEST: np.array([1, -1], dtype=np.int16),
            ActionSpace.FLY_NORTH_EAST: np.array([-1, 1], dtype=np.int16),
            ActionSpace.FLY_NORTH_WEST: np.array([-1, -1], dtype=np.int16),
        }
