from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from yaml import YAMLError, safe_load

import numpy as np
from skimage.measure import block_reduce

from gymnasium import spaces

from drone_grid_env import logger
from drone_grid_env.envs.utils import center_crop, pad_map

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from gymnasium.spaces import Space

    from drone_grid_env.envs.drone import Drone
    from drone_grid_env.envs.world import World


@dataclass
class StateSpaceConfig:
    global_map_reduce: int = 3
    add_start_landing_zone: bool = False


class StateSpace:
    def __init__(
        self,
        world: World[Any],
        drone: Drone,
        config: StateSpaceConfig = StateSpaceConfig(),
    ) -> None:
        self.config = config

        self._world = world
        self._drone = drone

    def get_space(self) -> Space[dict[str, NDArray[np.uint8]]]:
        MC = 2 * np.array(self._world.size) - 1
        GLOBAL_MAP_SIZE = np.ceil(MC.astype(np.float32) / self.config.global_map_reduce).astype(np.uint8)

        if self.config.add_start_landing_zone:
            local_map_space = spaces.Box(0, 255, (4, *self._drone.fov), dtype=np.uint8)
            global_map_space = spaces.Box(0, 255, (4, *GLOBAL_MAP_SIZE), dtype=np.uint8)
        else:
            local_map_space = spaces.Box(0, 255, (3, *self._drone.fov), dtype=np.uint8)
            global_map_space = spaces.Box(0, 255, (3, *GLOBAL_MAP_SIZE), dtype=np.uint8)

        flying_time_space = spaces.Box(0, 255, shape=(1,), dtype=np.uint8)

        return spaces.Dict({"local_map": local_map_space, "global_map": global_map_space, "flying_time": flying_time_space})

    def get_state(self) -> dict[str, NDArray[np.uint8]]:
        # Create a map of the remaining objects
        # remaining_objects_map = np.zeros(self._world.size, dtype=np.uint8)
        # remaining_objects_map[self._world.object_map] = 255
        # if self._drone.classified_object_positions.shape[0] > 0:
        #     remaining_objects_map[self._drone.classified_object_positions[:, 0], self._drone.classified_object_positions[:, 1]] = 0

        classified_objects_map = np.zeros(self._world.size, dtype=np.uint8)
        classified_objects_map[self._world.object_map] = 0
        if self._drone.found_object_positions.shape[0] > 0:
            classified_objects_map[self._drone.found_object_positions[:, 0], self._drone.found_object_positions[:, 1]] = 255

        # Create scalar with the current battery value
        battery_scalar = np.array([(self._drone.battery_level / self._drone.config.initial_battery_level) * 255], dtype=np.uint8)

        prior_knowledge = self._world.prior_knowledge.copy()
        no_fly_zone_map = self._world.no_fly_zone_map.copy().astype(np.uint8) * 255
        start_landing_zone_map = self._world.start_landing_map.copy().astype(np.uint8) * 255

        # Pad start-landing zone map and remaining object map with zero's -> e.g. no objects outside the world
        pad_zero = pad_map(
            np.stack((prior_knowledge, classified_objects_map, start_landing_zone_map), axis=-1),
            self._drone.position,
            pad_value=0,
            first_axis=False,
        )

        # Pad the no-fly-zone map with max values, so that outside the world there is a no-fly-zone
        pad_one = pad_map(np.expand_dims(no_fly_zone_map, axis=-1), self._drone.position, pad_value=255, first_axis=False)

        # Create local and global map by eigher take the center crop
        local_map = center_crop(
            np.stack((pad_one[:, :, 0], pad_zero[:, :, 1], pad_zero[:, :, 2]), axis=0),
            self._drone.config.fov,
            first_axis=True,
        )
        global_map = block_reduce(
            np.stack((pad_zero[:, :, 0], pad_one[:, :, 0], pad_zero[:, :, 1], pad_zero[:, :, 2]), axis=0),
            block_size=(1, self.config.global_map_reduce, self.config.global_map_reduce),
            func=np.mean,
        ).astype(np.uint8)

        # Remove start-landing zone map when needed
        if not self.config.add_start_landing_zone:
            local_map = local_map[:2:, :, :]
            global_map = global_map[:3, :, :]

        # Add observation to local map
        observation = np.expand_dims(self._drone.observation.astype(np.uint8) * 255, axis=0)
        local_map = np.concatenate((observation, local_map), axis=0)

        return {
            "local_map": local_map,
            "global_map": global_map,
            "flying_time": battery_scalar,
        }

    @classmethod
    def from_config_file(
        cls,
        config_file: Path,
        world: World[Any],
        drone: Drone,
    ) -> StateSpace:
        with config_file.open("r") as cs:
            try:
                data: dict[str, Any] = safe_load(cs)["state_space"]

                global_map_reduce = data["global_map_reduce"]
                add_start_landing_zone = data["add_start_landing_zone"]

                config = StateSpaceConfig(
                    global_map_reduce,
                    add_start_landing_zone,
                )

                return cls(world, drone, config)

            except (YAMLError, KeyError) as e:  # pragma: no cover
                logger.error(f"Could not load config file '{config_file.name}': {e}")
                raise
