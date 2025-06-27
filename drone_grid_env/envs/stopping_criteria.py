from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from yaml import safe_load

import numpy as np

from .action import ActionSpace
from .utils import count_gt_classified_objects

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Literal

    from drone_grid_env.envs.drone import Drone
    from drone_grid_env.envs.world import World


@dataclass
class StoppingCriteriaConfig:
    method: Literal["objects", "coverage", "land", "no_new_objects_found", "none"] = "objects"
    value: float = 1.0
    only_land_in_zone: bool = False


class StoppingCriteria:
    def __init__(self, config: StoppingCriteriaConfig = StoppingCriteriaConfig()) -> None:
        self.config = config

        self._step_last_found_object = 0
        self._num_found_objects = 0

    def reset(self) -> None:
        if self.config.method == "no_new_objects_found":
            self._step_last_found_object = 0
            self._num_found_objects = 0

    def can_terminate(self, world: World[Any], drone: Drone, action: ActionSpace | int) -> bool:
        if self.config.method == "objects":
            return (
                count_gt_classified_objects(world, drone) / world.number_of_objects >= self.config.value
                if world.number_of_objects > 0
                else True
            )
        elif self.config.method == "coverage":
            return np.count_nonzero(drone.coverage_map) / (world.size[0] * world.size[1]) >= self.config.value
        elif self.config.method == "land":
            return (
                action == ActionSpace.LAND and world.start_landing_map[tuple(drone.position)] == world.start_landing_map.max()
                if self.config.only_land_in_zone
                else action == ActionSpace.LAND
            )
        elif self.config.method == "no_new_objects_found":
            num_found_objects = drone.found_object_positions.shape[0]
            if num_found_objects > self._num_found_objects + 1:  # Detect minimal 2 new objects to avoid problems with FP
                self._num_found_objects = num_found_objects
                self._step_last_found_object = len(drone.flight_path)

            return (len(drone.flight_path) - self._step_last_found_object) > self.config.value
        elif self.config.method == "none":
            return False

        raise NotImplementedError(f"Stopping criteria {self.config.method} is not implemented!")

    @classmethod
    def from_config_file(cls, config_file: Path) -> StoppingCriteria:
        with config_file.open("r") as cs:
            data: dict[str, Any] = safe_load(cs)["stopping_criteria"]

            config = StoppingCriteriaConfig(
                data["method"],
                data["value"],
                data["only_land_in_zone"],
            )
            return cls(config=config)
