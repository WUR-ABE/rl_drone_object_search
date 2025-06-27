from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from yaml import YAMLError, safe_load

import numpy as np
from numpy.random._generator import default_rng as default_rng
import torch as th

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from drone_grid_env import logger
from drone_grid_env.envs.io.kml import read_kml_coordinate_map, write_kml_object_locations
from drone_grid_env.envs.io.orthomosaic import DynamicOrthomosaicLoader, rotation_matrix
from drone_grid_env.envs.io.topcon import read_topcon_data
from drone_grid_env.envs.utils import Zone, block_mean, coordinates_in_size, get_file
from drone_grid_env.envs.world import World

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from numpy.typing import NDArray

    from ultralytics.engine.results import Results


class OrthomosaicWorldException(Exception):
    pass


@dataclass
class OrthomosaicWorldConfig:
    size: tuple[int, int] = (63, 63)
    start_landing_zones: list[Zone] = field(default_factory=list)

    scheme_file: Path = Path()
    objects_file: Path = Path()
    name: str = "world"
    yolo_weights_file: Path = Path("best_n.pt")
    origin: NDArray[np.float64] = field(default_factory=lambda: np.array([683162.224, 5763408.719], dtype=np.float64))
    rotation: int = 0
    scale: float = 1.0
    min_conf: float = 0.05
    camera_size: tuple[int, int] = (1280, 1280)
    crs: str = "epsg:32631"

    # Prior knowledge
    simulate_prior_knowledge: bool = False  # Whether to use real prior knowledge or use GT
    prior_knowledge_confidence_threshold: float = 0.01
    coverage_flight_fov: tuple[int, int] = (48, 48)
    coverage_flight_camera_size: tuple[int, int] = (2048, 2048)
    prior_knowledge_size: tuple[int, int] = (48, 48)

    visualise: bool = True


class OrthomosaicWorld(World[OrthomosaicWorldConfig]):
    def __init__(
        self,
        config: OrthomosaicWorldConfig,
        size: tuple[int, int],
        start_landing_zones: list[Zone] = [],
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(config, size=size, start_landing_zones=start_landing_zones, rng=rng)

        logger.info("Using GPU") if th.cuda.is_available() else logger.info("Using CPU")

        # Import YOLO here, since it's slow
        from ultralytics import YOLO

        self.orthomosaic_loader = DynamicOrthomosaicLoader(get_file(self.orthomosaic_scheme_file), crs=self.config.crs)
        self.detection_network = YOLO(self.yolo_weights_file)

    def create_observation(self, coordinate_local: NDArray[np.uint16], fov: tuple[int, int] | NDArray[np.uint16]) -> NDArray[np.float32]:
        utm_coordinate = self.convert_local_to_utm(coordinate_local.reshape(-1, 2))[0, :]
        fov_m = tuple(np.array(fov, dtype=np.float32) * self.config.scale)

        image = self.orthomosaic_loader.get_image_at_coordinate(utm_coordinate, -self.config.rotation, self.config.camera_size, fov_m)

        if image is None:
            return np.zeros(fov, dtype=np.float32)

        image = image[:, :, ::-1]  # RGB -> BGR

        callback = (lambda x: setattr(self, "_additional_visualisation", x)) if self.config.visualise else None
        return self.image_to_observation(image, fov, visualisation_callback=callback)

    def convert_utm_to_local(self, utm_coordinates: NDArray[np.float64]) -> NDArray[np.int16]:
        coordinates_local = utm_coordinates.copy()
        coordinates_local -= self.config.origin
        coordinates_local[:, 1] *= -1  # UTM has north up
        coordinates_local /= self.config.scale
        coordinates_local = coordinates_local[:, [1, 0]]
        coordinates_local @= rotation_matrix(-self.config.rotation).T
        coordinates_local += (self.size_arr - 1) / 2
        return coordinates_local.round().astype(np.int16)

    def convert_local_to_utm(self, local_coordinates: NDArray[np.int16]) -> NDArray[np.float64]:
        utm_coordinates = local_coordinates.copy().astype(np.float64)
        utm_coordinates -= (self.size_arr - 1) / 2
        utm_coordinates @= rotation_matrix(self.config.rotation).T  # Rotate back
        utm_coordinates = utm_coordinates[:, [1, 0]]
        utm_coordinates *= self.config.scale
        utm_coordinates[:, 1] *= -1  # UTM has north up
        utm_coordinates += self.config.origin
        return utm_coordinates

    def image_to_observation(
        self,
        image: NDArray[np.uint8],
        output_size: tuple[int, int] | NDArray[np.uint16],
        visualisation_callback: Callable[[NDArray[np.uint8]], None] | None = None,
    ) -> NDArray[np.float32]:
        result: Results = self.detection_network.predict(image, conf=self.config.min_conf, verbose=False)[0]
        result = result.cpu()

        logger.info(f"Detected {len(result.boxes)} objects")

        ratio: NDArray[np.float32] = np.array(output_size) / result.orig_shape

        detections = result.boxes.xywh[:, :2].numpy()[:, [1, 0]]  # Yolo format swaps x and y
        detections *= ratio
        detections = detections.astype(np.uint16)

        if visualisation_callback:
            visualisation_callback(result.plot()[:, :, ::-1])

        detection_map = np.zeros(output_size, dtype=np.float32)
        detection_map[detections[:, 0], detections[:, 1]] = result.boxes.conf
        return detection_map

    @cached_property
    def world_center_utm(self) -> NDArray[np.float64]:
        center_utm = self.config.origin.copy()
        center_utm[0] += (self.config.size[0] / 2) * self.config.scale
        center_utm[1] -= (self.config.size[1] / 2) * self.config.scale
        return center_utm

    @cached_property
    def object_map(self) -> NDArray[np.bool_]:
        object_map = np.zeros(self.size, dtype=np.bool_)
        object_map[self.gt_coordinates[:, 0], self.gt_coordinates[:, 1]] = True
        return object_map

    @cached_property
    def no_fly_zone_map(self) -> NDArray[np.bool_]:
        return np.zeros(self.size, dtype=np.bool_)

    @cached_property
    def prior_knowledge(self) -> NDArray[np.uint8]:
        if self.config.simulate_prior_knowledge:
            prior_knowledge = np.zeros(self.size, dtype=np.uint8)
            prior_knowledge[self.gt_coordinates[:, 0], self.gt_coordinates[:, 1]] = 255
            return prior_knowledge

        prior_knowledge_folder = Path("generated_prior_knowledge")
        prior_knowledge_folder.mkdir(exist_ok=True)

        prior_knowledge_file = (
            prior_knowledge_folder
            / f"prior_knowledge_{self.config.name}_{self.config.coverage_flight_fov[0]}x{self.config.coverage_flight_fov[1]}.kml"
        )

        if prior_knowledge_file.is_file():
            logger.info(f"Loading prior knowledge from '{prior_knowledge_file.name}'")
            prior_knowledge = read_kml_coordinate_map(prior_knowledge_file, self.config.size, dtype=np.float32)
        else:
            prior_knowledge = self.create_prior_knowledge(np.array(self.config.coverage_flight_fov, dtype=np.uint16))
            self.write_prior_knowledge(prior_knowledge_file, prior_knowledge)

        prior_knowledge_map = np.zeros(self.size, dtype=np.uint8)
        prior_knowledge_map[prior_knowledge >= self.config.prior_knowledge_confidence_threshold] = 255

        block_size = np.array(prior_knowledge_map.shape[:2]) // self.config.prior_knowledge_size
        return block_mean(prior_knowledge_map, block_size)

    @cached_property
    def orthomosaic_scheme_file(self) -> Path:
        return get_file(self.config.scheme_file)

    @cached_property
    def yolo_weights_file(self) -> Path:
        return get_file(self.config.yolo_weights_file)

    @cached_property
    def gt_coordinates(self) -> NDArray[np.uint16]:
        objects_file = get_file(self.config.objects_file)
        coordinates_utm = read_topcon_data(objects_file, output_crs=self.config.crs)
        coordinates_local = self.convert_utm_to_local(coordinates_utm)

        # Filter by the coordinates that are in the field
        return coordinates_local[coordinates_in_size(coordinates_local, self.size), :].astype(np.uint16)

    def create_prior_knowledge(self, fov: NDArray[np.uint16]) -> NDArray[np.float32]:
        x_coords = np.arange(0, self.size[0], fov[0]) + fov[0] // 2
        y_coords = np.arange(0, self.size[1], fov[1]) + fov[1] // 2
        prior_knowledge_coordinates = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

        prior_knowledge = np.zeros(self.size, dtype=np.float32)
        with logging_redirect_tqdm():
            for i in tqdm(range(prior_knowledge_coordinates.shape[0]), desc="Creating prior knowledge"):
                utm_coordinate = self.convert_local_to_utm(prior_knowledge_coordinates[i, :].reshape(-1, 2))[0]
                image = self.orthomosaic_loader.get_image_at_coordinate(
                    utm_coordinate, -self.config.rotation, self.config.coverage_flight_camera_size, (fov * self.config.scale).tolist()
                )
                image = image[:, :, ::-1]  # RGB -> BGR
                observation = self.image_to_observation(image, fov)
                prior_knowledge[
                    prior_knowledge_coordinates[i, 0] - fov[0] // 2 : prior_knowledge_coordinates[i, 0] + fov[0] // 2,
                    prior_knowledge_coordinates[i, 1] - fov[1] // 2 : prior_knowledge_coordinates[i, 1] + fov[1] // 2,
                ] = observation

            return prior_knowledge

    @classmethod
    def from_config_file(
        cls: type[OrthomosaicWorld], config_file: Path, rng: np.random.Generator = np.random.default_rng()
    ) -> OrthomosaicWorld:
        with config_file.open("r") as cs:
            try:
                data: dict[str, Any] = safe_load(cs)
                world_config: dict[str, Any] = data["world"]

                scheme_file = get_file(world_config["scheme_file"])
                objects_file = get_file(world_config["objects_file"])
                yolo_weights_file = get_file(world_config["yolo_weights_file"])

                origin = np.array([world_config["origin"]["easting"], world_config["origin"]["northing"]], dtype=np.float64)

                size, start_landing_zones = cls._get_size_and_start_land_map(world_config)

                crs = world_config["crs"]
                name = world_config["name"]
                rotation = world_config["rotation"]
                scale = world_config["scale"]
                min_conf = world_config["min_conf"]
                camera_size = (world_config["camera"]["width"], world_config["camera"]["height"])
                coverage_flight_fov = (
                    world_config["prior_knowledge"]["coverage_fov"]["width"],
                    world_config["prior_knowledge"]["coverage_fov"]["height"],
                )
                coverage_flight_camera_size = (
                    world_config["prior_knowledge"]["camera"]["width"],
                    world_config["prior_knowledge"]["camera"]["height"],
                )
                prior_knowledge_size = (
                    world_config["prior_knowledge"]["size"]["width"],
                    world_config["prior_knowledge"]["size"]["height"],
                )
                prior_knowledge_confidence_threshold = world_config["prior_knowledge"]["confidence_threshold"]

                config = OrthomosaicWorldConfig(
                    scheme_file=scheme_file,
                    objects_file=objects_file,
                    name=name,
                    size=size,
                    yolo_weights_file=yolo_weights_file,
                    origin=origin,
                    rotation=rotation,
                    scale=scale,
                    min_conf=min_conf,
                    camera_size=camera_size,
                    crs=crs,
                    coverage_flight_fov=coverage_flight_fov,
                    coverage_flight_camera_size=coverage_flight_camera_size,
                    prior_knowledge_size=prior_knowledge_size,
                    prior_knowledge_confidence_threshold=prior_knowledge_confidence_threshold,
                )

                return cls(config, size=size, start_landing_zones=start_landing_zones, rng=rng)

            except (YAMLError, KeyError) as e:  # pragma: no cover
                logger.error(f"Could not load config file '{config_file.name}': {e}")
                raise

    @staticmethod
    def write_prior_knowledge(output_file: Path, prior_knowledge: NDArray[np.float32]) -> None:
        coordinates = np.column_stack(np.nonzero(prior_knowledge))
        values = prior_knowledge[coordinates[:, 0], coordinates[:, 1]]

        ext_data = []
        for i in range(coordinates.shape[0]):
            ext_data.append({"confidence": values[i]})

        write_kml_object_locations(output_file, coordinates, point_extended_data=ext_data)


if __name__ == "__main__":
    from drone_grid_env.envs.drone_grid_env import DroneGridEnv
    from drone_grid_env.utils.play import play

    env = DroneGridEnv(config_file="experiments/showcase/clustered_1.yaml", verbose=True, render_mode="rgb_array")
    play(env)
