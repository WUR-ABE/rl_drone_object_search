from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from yaml import YAMLError, safe_load

import numpy as np

from scipy.stats import multivariate_normal

from drone_grid_env import logger
from drone_grid_env.envs.io.kml import read_kml_coordinate_map
from drone_grid_env.envs.utils import Distribution, Zone, block_mean, coordinates_in_size, get_file
from drone_grid_env.envs.world import World

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


@dataclass
class Uncertainty:
    position: float = 0.0
    percentage_fp: float = 0.0
    percentage_fn: float = 0.0


@dataclass
class PriorKnowledge:
    from_file: Path | None = None
    size: tuple[int, int] = (21, 21)
    uncertainty: Uncertainty = field(default_factory=lambda: Uncertainty())


@dataclass
class SimWorldConfig:
    from_file: Path | None = None
    n_patches: Distribution = field(default_factory=lambda: Distribution("normal", {"loc": 3, "scale": 1}))
    n_objects: Distribution = field(default_factory=lambda: Distribution("normal", {"loc": 20, "scale": 1}))
    objects_in_patches_position: Distribution = field(
        default_factory=lambda: Distribution("multivariate_normal", {"cov": [[5, 8], [8, 20]]})
    )
    prior_knowledge: PriorKnowledge = field(default_factory=lambda: PriorKnowledge())
    observation_uncertainty: Uncertainty = field(default_factory=lambda: Uncertainty())


class SimWorld(World[SimWorldConfig]):
    def __init__(
        self,
        config: SimWorldConfig,
        size: tuple[int, int],
        start_landing_zones: list[Zone] = [],
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(config, size=size, start_landing_zones=start_landing_zones, rng=rng)

        self._prior_knowledge_map: NDArray[np.uint8] | None = None
        self._object_map = self.create_object_map()
        self._no_fly_zone_map = np.zeros(self.size, dtype=np.bool_)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        super().reset(rng)

        self._prior_knowledge_map = None
        self._object_map = self.create_object_map()
        self._no_fly_zone_map = np.zeros(self.size, dtype=np.bool_)

    def create_observation(self, coordinate_local: NDArray[np.uint16], fov: tuple[int, int] | NDArray[np.uint16]) -> NDArray[np.float32]:
        pad_rows, pad_columns = int(fov[0] // 2), int(fov[1] // 2)

        observation_padded = np.pad(
            self.object_map, ((pad_rows, pad_rows), (pad_columns, pad_columns)), mode="constant", constant_values=False
        )
        obs_gt = observation_padded[coordinate_local[0] : coordinate_local[0] + fov[0], coordinate_local[1] : coordinate_local[1] + fov[1]]

        obs_coordinates = self.get_coordinates_with_uncertainty(obs_gt, self.config.observation_uncertainty)
        obs = np.zeros(fov, dtype=np.float32)
        obs[obs_coordinates[:, 0], obs_coordinates[:, 1]] = 1.0

        return obs

    @property
    def object_map(self) -> NDArray[np.bool_]:
        return self._object_map

    @property
    def no_fly_zone_map(self) -> NDArray[np.bool_]:
        # Create a no-fly-zone map with only zero values (e.g. drone can fly everywhere in the world)
        return self._no_fly_zone_map

    @property
    def prior_knowledge(self) -> NDArray[np.uint8]:
        if self._prior_knowledge_map is None:
            self._prior_knowledge_map = np.zeros(self.size, dtype=np.uint8)

            if self.config.prior_knowledge.size == (0, 0):
                return self._prior_knowledge_map

            if self.config.prior_knowledge.from_file is not None:
                prior_knowledge = read_kml_coordinate_map(self.config.prior_knowledge.from_file, self.size, dtype=np.float32)
                prior_knowledge[prior_knowledge < 0.01] = 0.0  # Confidence threshold
                coordinates = np.column_stack(prior_knowledge.nonzero())
            else:
                coordinates = self.get_coordinates_with_uncertainty(self.object_map, self.config.prior_knowledge.uncertainty)

            self._prior_knowledge_map = np.zeros(self.size, dtype=np.uint8)
            self._prior_knowledge_map[coordinates[:, 0], coordinates[:, 1]] = 255

            # Add resolution uncertainty
            block_size = np.array(self._prior_knowledge_map.shape[:2]) // self.config.prior_knowledge.size
            self._prior_knowledge_map = block_mean(self._prior_knowledge_map, block_size)

            # import matplotlib.pyplot as plt

            # plt.imshow(self._prior_knowledge_map, cmap="viridis")
            # plt.show()

        return self._prior_knowledge_map

    def create_object_map(self) -> NDArray[np.bool_]:
        field = np.zeros(self.size_arr, dtype=np.bool_)

        if self.config.from_file is not None:
            object_coordinates = np.loadtxt(self.config.from_file, delimiter=" ", dtype=np.uint16)
            field[object_coordinates[:, 0], object_coordinates[:, 1]] = True
            return field

        n_objects = max(1, int(self.config.n_objects.get_dist_value(self.rng).round(0)))
        n_patches = max(1, int(self.config.n_patches.get_dist_value(self.rng).round(0)))

        ## Step 1: Get all cells (x,y location) in a lit
        xx, yy = np.meshgrid(np.arange(self.size_arr[0]), np.arange(self.size_arr[1]))
        cells = np.array([xx, yy]).T.reshape(self.size_arr.prod(), 2)

        distributions: list[Distribution] = []
        if self.config.objects_in_patches_position.dist_type == "multivariate_normal":
            probs = np.zeros(self.size_arr.prod())

            ## Step 2: Calculate the probability per cell using the distributions
            for i in range(n_patches):
                distribution = deepcopy(self.config.objects_in_patches_position)
                distribution.set_value("mean", self.rng.integers((0, 0), self.size, size=2))

                # If no fixed covariance, create random value
                if not distribution.contains_value("cov"):
                    dist_x = self.rng.integers(3, 20)
                    dist_y = self.rng.integers(3, dist_x * 2)
                    cov = np.array([[dist_x, 8], [8, dist_y]])
                    distribution.set_value("cov", np.dot(cov, cov.transpose()))

                elif isinstance(distribution.get_value("cov"), list):
                    cov_list = distribution.get_value("cov")
                    cov_i = i if len(cov_list) == n_patches else self.rng.integers(0, len(cov_list), 1)[0]
                    distribution.set_value("cov", distribution.get_value("cov")[cov_i])

                # Calculate probs
                probs += multivariate_normal.pdf(cells, mean=distribution.get_value("mean"), cov=distribution.get_value("cov")) / n_patches

                distributions.append(distribution)

            probs /= probs.sum()

        elif self.config.objects_in_patches_position.dist_type == "uniform":
            distribution = deepcopy(self.config.objects_in_patches_position)
            distribution.set_value("low", (0, 0))
            distribution.set_value("high", self.size_arr)

            distributions.append(distribution)
            probs = np.ones(np.prod(self.size_arr)) / np.prod(self.size_arr)
        else:
            dt = self.config.objects_in_patches_position.dist_type
            raise NotImplementedError(f"Distribution of type '{dt}' is not implemented for objects_in_patches_position!")

        ## Step 2b: Show the probabilty distribution as a map as a check
        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.imshow(probs.reshape(self.size[0], self.size[1]), origin="lower")
        # plt.show()

        ## Step 3: Sample N cells using the probability distribution
        max_n_objects = np.prod(self.size_arr) - np.count_nonzero(self.start_landing_map)
        if n_objects > max_n_objects:
            logger.warn(f"Warning: Cannot place {n_objects} objects in map, only place for {max_n_objects}!")
            n_objects = max_n_objects

        object_coordinates = np.empty((n_objects, 2), dtype=np.int16)

        for i in range(n_objects):
            # Make a cumulative sum of the probs
            probs_cum = np.cumsum(probs)

            # Draw a random number from the uniform distribution [0,1)
            r = self.rng.random()

            # Find the element belonging to this random value
            object_i = np.searchsorted(probs_cum, r)

            # Add the corresponding object
            # print("Selected", object_i, "from",probs.shape, probs_cum.shape)
            object_coordinates[i, :] = cells[object_i, :]

            # Remove the object from the list and from the probs
            cells = np.delete(cells, object_i, axis=0)
            probs = np.delete(probs, object_i)

            # Normalize probs again so that it sums to 1
            probs /= probs.sum()

        field[object_coordinates[:, 0], object_coordinates[:, 1]] = True

        return field

    def get_coordinates_with_uncertainty(
        self,
        input_map: NDArray[np.bool_],
        uncertainty: Uncertainty,
    ) -> NDArray[np.uint16]:
        coordinates = np.vstack(input_map.nonzero()).astype(np.uint16).T

        # Position uncertainty
        coordinates += (
            self.rng.normal(
                loc=0.0,
                scale=uncertainty.position,
                size=(np.count_nonzero(input_map), 2),
            )
            .round()
            .astype(np.uint16)
        )

        # False negatives
        n_fn = round(np.count_nonzero(input_map) * uncertainty.percentage_fn)
        fn_indices = self.rng.choice(np.count_nonzero(input_map), n_fn, replace=False)
        coordinates = np.delete(coordinates, fn_indices, axis=0)

        # Add false positives
        false_positives = self.rng.random(input_map.shape[:2]) < uncertainty.percentage_fp
        coordinates = np.unique(np.vstack((coordinates, np.vstack(false_positives.nonzero()).T)), axis=0)

        # Remove coordinates outside the field
        coordinates = coordinates[coordinates_in_size(coordinates, input_map.shape[:2]), :]  # type: ignore[index,arg-type]
        return coordinates

    @classmethod
    def from_config_file(cls: type[SimWorld], config_file: Path, rng: np.random.Generator = np.random.default_rng()) -> SimWorld:
        with config_file.open("r") as cs:
            try:
                data: dict[str, Any] = safe_load(cs)
                world_config: dict[str, Any] = data["world"]

                if from_file := world_config["from_file"]:
                    from_file = get_file(from_file)

                def _dict2distr(dict_entry: dict[str, Any]) -> Distribution:
                    if (dist_kwargs := dict_entry.get("dist_kwargs")) is None:
                        dist_kwargs = {}
                    return Distribution(dict_entry["dist_type"], dist_kwargs)

                size, start_landing_zones = cls._get_size_and_start_land_map(world_config)
                n_objects = _dict2distr(world_config["n_objects"])
                n_patches = _dict2distr(world_config["n_patches"])
                objects_in_patches_position = _dict2distr(world_config["objects_in_patch_position"])

                def _dict2uncertainty(dict_entry: dict[str, float]) -> Uncertainty:
                    return Uncertainty(dict_entry["position"], dict_entry["false_positives"], dict_entry["false_negatives"])

                if prior_knowledge_file := world_config["prior_knowledge"]["from_file"]:
                    prior_knowledge_file = get_file(prior_knowledge_file)

                prior_knowledge_uncertainty = _dict2uncertainty(world_config["prior_knowledge"]["uncertainty"])
                prior_knowledge_size = (world_config["prior_knowledge"]["size"]["height"], world_config["prior_knowledge"]["size"]["width"])
                prior_knowledge = PriorKnowledge(prior_knowledge_file, prior_knowledge_size, prior_knowledge_uncertainty)

                observation_uncertainty = _dict2uncertainty(world_config["observation_uncertainty"])

                config = SimWorldConfig(
                    from_file=from_file,
                    n_patches=n_patches,
                    n_objects=n_objects,
                    objects_in_patches_position=objects_in_patches_position,
                    prior_knowledge=prior_knowledge,
                    observation_uncertainty=observation_uncertainty,
                )

                return cls(config, size=size, start_landing_zones=start_landing_zones, rng=rng)

            except (YAMLError, KeyError) as e:  # pragma: no cover
                logger.error(f"Could not load config file '{config_file.name}': {e}")
                raise
