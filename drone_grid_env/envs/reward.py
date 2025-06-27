from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from yaml import YAMLError, safe_load

import numpy as np

from drone_grid_env import logger
from drone_grid_env.envs.action import ActionSpace
from drone_grid_env.envs.utils import count_gt_classified_objects, setdiff2d

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

    from drone_grid_env.envs.drone import Drone
    from drone_grid_env.envs.stopping_criteria import StoppingCriteria
    from drone_grid_env.envs.world import World


@dataclass
class RewardConfig:
    hit_no_fly_zone: float = -1.0
    empty_battery: float = -150.0
    coverage: float = 0.05
    fly_action: float = -0.2
    discovered_objects_tp: float = 1.0
    discovered_objects_fp: float = -1.0
    undiscovered_objects: float = 0.0
    normalize_objects: bool = False


class RewardSpace:
    def __init__(self, config: RewardConfig = RewardConfig()) -> None:
        self.config = config
        self._reward = 0.0

    @contextmanager
    def calculate_reward(self, world: World[Any], drone: Drone, stopping_criteria: StoppingCriteria, action: int) -> Iterator[None]:
        self._reward = 0.0

        discovered_objects_reward_tp = self.config.discovered_objects_tp
        discovered_objects_reward_fp = self.config.discovered_objects_fp
        if self.config.normalize_objects:
            discovered_objects_reward_tp = discovered_objects_reward_tp * (200 / world.number_of_objects)  # 200 for backward compatability
            discovered_objects_reward_fp = discovered_objects_reward_fp * (200 / world.number_of_objects)  # 200 for backward compatability

        try:
            position = drone.position.copy()
            coverage_map = drone.coverage_map.copy()
            classified_object_positions = drone.found_object_positions.copy()
            yield

        finally:
            # Detect whether the action was valid (e.g. no no-fly-zone hit or landed on non-landing zone)
            if (ActionSpace.is_movement_action(action) and not np.array_equal(position, drone.position)) or (
                action == ActionSpace.LAND and stopping_criteria.can_terminate(world, drone, action)
            ):
                if ActionSpace.is_movement_action(action):
                    self._reward += self.config.fly_action

                # Coverage reward
                if not np.array_equal(position, drone.position):
                    discovered_pixels = np.count_nonzero(drone.coverage_map) - np.count_nonzero(coverage_map)
                    self._reward += discovered_pixels * self.config.coverage

                # Classified objects reward
                discovered_objects = setdiff2d(drone.found_object_positions, classified_object_positions)

                for i in range(discovered_objects.shape[0]):
                    if world.object_map[discovered_objects[i, 0], discovered_objects[i, 1]]:  # TP
                        self._reward += discovered_objects_reward_tp
                    else:  # FP
                        self._reward += discovered_objects_reward_fp

            else:
                self._reward = self.config.hit_no_fly_zone

            # Empty battery
            if drone.battery_level <= 0 and not stopping_criteria.can_terminate(world, drone, action):
                self._reward += self.config.empty_battery

            # Final reward for all objects that were not detected
            if stopping_criteria.can_terminate(world, drone, action) and world.number_of_objects > 0:
                remaining_objects = world.number_of_objects - count_gt_classified_objects(world, drone)
                self._reward += self.config.undiscovered_objects * ((remaining_objects) / world.number_of_objects)

    def get_reward(self) -> float:
        return self._reward

    @classmethod
    def from_config_file(cls, config_file: Path) -> RewardSpace:
        with config_file.open("r") as cs:
            try:
                config_dict = safe_load(cs)
                config = RewardConfig(
                    **config_dict["rewards"],
                )
                return RewardSpace(config)

            except (YAMLError, KeyError) as e:  # pragma: no cover
                logger.error(f"Could not load config file '{config_file.name}': {e}")
                raise
