from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar, Generic

import numpy as np

from drone_grid_env.envs.utils import Zone, coordinates_in_size

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray


WorldType = TypeVar("WorldType", bound="BaseWorld")
WorldConfigType = TypeVar("WorldConfigType", bound="BaseWorldConfig")


@dataclass(kw_only=True)
class BaseWorldConfig:
    size: tuple[int, int] = (63, 63)
    start_landing_zones: list[Zone] = field(default_factory=list)


class BaseWorld(Generic[WorldConfigType], metaclass=ABCMeta):
    config: WorldConfigType

    def __init__(
        self,
        config: WorldConfigType = BaseWorldConfig(),
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        self.config = config
        self.rng = rng

        self._start_landing_map: NDArray[np.bool_] | None = None
        self._additional_visualisation: NDArray[np.uint8] | None = None

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng

    @abstractmethod
    def create_observation(self, coordinate_local: NDArray[np.uint16], fov: tuple[int, int] | NDArray[np.uint16]) -> NDArray[np.float32]:
        pass

    @property
    def size(self) -> tuple[int, int]:
        return self.config.size

    @property
    def start_landing_map(self) -> NDArray[np.bool_]:
        if self._start_landing_map is None:
            self._start_landing_map = np.zeros(self.config.size, dtype=np.bool_)

            for zone in self.config.start_landing_zones:
                coordinates = zone.coordinates[coordinates_in_size(zone.coordinates, self.config.size)]
                self._start_landing_map[coordinates[:, 0], coordinates[:, 1]] = True

        return self._start_landing_map

    @property
    def number_of_objects(self) -> int:
        return np.count_nonzero(self.object_map)

    @property
    def additional_visualisation(self) -> NDArray[np.uint8] | None:
        return self._additional_visualisation

    @property
    @abstractmethod
    def no_fly_zone_map(self) -> NDArray[np.bool_]:
        pass

    @property
    @abstractmethod
    def object_map(self) -> NDArray[np.bool_]:
        pass

    @property
    @abstractmethod
    def prior_knowledge(self) -> NDArray[np.uint8]:
        pass

    @classmethod
    @abstractmethod
    def from_config_file(cls: type[WorldType], config_file: Path, rng: np.random.Generator) -> WorldType:
        pass

    @staticmethod
    def _get_size_and_start_land_map(world_data: dict[str, Any]) -> tuple[tuple[int, int], list[Zone]]:
        size = (world_data["size"]["height"], world_data["size"]["width"])

        def _dict2rect(dict_entry: dict[str, int]) -> Zone:
            x = dict_entry["x"]
            y = dict_entry["y"]

            if x < 0:
                x += size[0]

            if y < 0:
                y += size[1]

            return Zone(x, y, dict_entry["h"], dict_entry["w"])

        def _get(d: dict[str, Any], key: str, replace: Any) -> Any:
            return replace if d.get(key, replace) is None else d.get(key, replace)

        start_landing_zones = [_dict2rect(slz) for slz in _get(world_data, "start_landing_zones", [])]

        return size, start_landing_zones
