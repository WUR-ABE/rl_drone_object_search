from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING
from yaml import YAMLError, safe_load

import numpy as np

from drone_grid_env import logger
from drone_grid_env.envs.action import ActionSpace
from drone_grid_env.envs.utils import FlightPath, coordinate_in_list, coordinates_in_size

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from drone_grid_env.envs.world import World


class StartMode(Enum):
    TOP_LEFT = auto()
    RANDOM = auto()
    AT_BORDER = auto()
    IN_START_ZONE = auto()


@dataclass
class DroneConfig:
    fov: tuple[int, int] = (11, 11)
    start_mode: StartMode = StartMode.TOP_LEFT

    # Battery
    initial_battery_level: float = 100.0
    battery_usage_flying: float = 0.2
    battery_usage_landing: float = 0.2

    confidence_threshold: float = 0.5


class Drone:
    def __init__(
        self,
        field: World[Any],
        rng: np.random.Generator = np.random.default_rng(),
        config: DroneConfig = DroneConfig(),
    ) -> None:
        self._rng = rng
        self._world = field
        self._config = config

        self._flight_path = FlightPath()
        self._flight_path.add(self._get_initial_position())

        self._battery_level: float = self._config.initial_battery_level

        self._found_objects: list[NDArray[np.uint16]] = []
        self._coverage_map: NDArray[np.uint8] | None = None
        self._observation: NDArray[np.bool_] | None = None

    @property
    def config(self) -> DroneConfig:
        return self._config

    @property
    def fov(self) -> NDArray[np.uint16]:
        return np.array(self._config.fov, dtype=np.uint16)

    @property
    def flight_path(self) -> FlightPath:
        return self._flight_path

    @property
    def position(self) -> NDArray[np.uint16]:
        return self.flight_path.current_position

    @property
    def found_object_positions(self) -> NDArray[np.uint16]:
        return np.array(self._found_objects).reshape(len(self._found_objects), 2)

    @property
    def battery_level(self) -> float:
        return self._battery_level

    @property
    def path_length(self) -> float:
        return self._flight_path.total_distance

    @property
    def coverage_map(self) -> NDArray[np.uint8]:
        if self._coverage_map is None:
            self._coverage_map = np.zeros(self._world.size, dtype=np.uint8)
            for dp in self.flight_path:
                dp = dp.astype(np.int32)  # Avoid overflow error
                self._coverage_map[
                    np.clip(dp[0] - self.fov[0] // 2, 0, self._coverage_map.shape[0]) : np.clip(
                        dp[0] + self.fov[0] // 2 + 1, 0, self._coverage_map.shape[0]
                    ),
                    np.clip(dp[1] - self.fov[1] // 2, 0, self._coverage_map.shape[1]) : np.clip(
                        dp[1] + self.fov[1] // 2 + 1, 0, self._coverage_map.shape[1]
                    ),
                ] = np.iinfo(np.uint8).max
        return self._coverage_map

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self._rng = rng

        self._flight_path.reset()
        self._flight_path.add(self._get_initial_position())

        self._battery_level = self._config.initial_battery_level
        self._observation = None
        self._found_objects.clear()

        self.discover_objects()

    @property
    def observation(self) -> NDArray[np.bool_]:
        if self._observation is None:
            confidences = self._world.create_observation(self.position, self.fov)
            self._observation = np.zeros(self.fov, dtype=np.bool_)
            self._observation = confidences >= self.config.confidence_threshold
        return self._observation

    def discover_objects(self) -> None:
        discovered_coordinates = np.column_stack(np.nonzero(self.observation)) + self.position - self.fov // 2
        discovered_coordinates = np.clip(discovered_coordinates, [0, 0], np.array(self._world.size) - 1)  # Fix rounding errors in detection
        self._found_objects.extend([p for p in discovered_coordinates if not coordinate_in_list(p, self.found_object_positions)])

    def fly(self, action: int) -> None:
        if not ActionSpace.is_movement_action(action):
            return

        position = self.position.copy().astype(np.int16)

        if self._valid_new_position(position + ActionSpace.get_actions_to_flight_map()[ActionSpace(action)]):
            position += ActionSpace.get_actions_to_flight_map()[ActionSpace(action)]

            # Only subtract battery when action is valid and executed
            self._coverage_map = None
            self._observation = None

            self.flight_path.add(position.astype(np.uint16))

            # Only discover objects when position is valid
            self.discover_objects()

        # Always remove something from the battery, even when flight action is not
        # valid. Otherwise, the drone can fly forever hitting the border of the field
        # and the training will crash
        self._battery_level -= self._config.battery_usage_flying

    def land(self) -> None:
        self._battery_level -= self._config.battery_usage_landing

    def _valid_new_position(self, new_position: NDArray[np.int16]) -> bool:
        if not coordinates_in_size(new_position.reshape((1, 2)), self._world.size)[0]:
            return False

        return True

    def _get_initial_position(self) -> NDArray[np.uint16]:
        if self._config.start_mode == StartMode.TOP_LEFT:
            return np.array((0, 0), dtype=np.uint16)

        if self._config.start_mode == StartMode.RANDOM:
            return self._rng.integers((0, 0), self._world.size, size=2, dtype=np.uint16)

        if self._config.start_mode == StartMode.AT_BORDER:
            border_coordinates = [(0, i) for i in range(self._world.size[1])]  # top border
            border_coordinates.extend((self._world.size[0] - 1, i) for i in range(self._world.size[1]))  # bottom border
            border_coordinates.extend((i, 0) for i in range(1, self._world.size[0] - 1))  # left border
            border_coordinates.extend((i, self._world.size[1] - 1) for i in range(1, self._world.size[0] - 1))  # right border

            return np.array(self._rng.choice(border_coordinates, 1)[0]).astype(np.uint16)

        if self._config.start_mode == StartMode.IN_START_ZONE:
            if len(self._world.start_landing_zones) == 0:
                raise RuntimeError("Cannot use 'IN_START_ZONE' initial position because the 'start_landing_zones' key is not defined!")

            random_zone = self._rng.integers(0, len(self._world.start_landing_zones))
            return np.array(np.mean(self._world.start_landing_zones[random_zone].coordinates, axis=0)).astype(np.uint16)

        raise NotImplementedError(f"Start mode {self._config.start_mode} is not implemented!")

    @classmethod
    def from_config_file(cls, config_file: Path, world: World[Any], rng: np.random.Generator = np.random.default_rng()) -> Drone:
        with config_file.open("r") as cs:
            try:
                data: dict[str, Any] = safe_load(cs)
                drone_config: dict[str, Any] = data["drone"]

                config = DroneConfig(
                    fov=(
                        drone_config["fov"]["height"],
                        drone_config["fov"]["width"],
                    ),
                    start_mode=StartMode[drone_config["start_mode"].upper()],
                    initial_battery_level=drone_config["initial_battery_level"],
                    battery_usage_flying=drone_config["battery_usage_flying"],
                    battery_usage_landing=drone_config["battery_usage_landing"],
                    confidence_threshold=drone_config["confidence_threshold"],
                )
                return cls(world, rng, config)
            except (YAMLError, KeyError) as e:  # pragma: no cover
                logger.error(f"Could not load config file '{config_file.name}': {e}")
                raise
