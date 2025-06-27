from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import WARNING, getLogger
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from affine import Affine
from lxml import etree as ET
from rasterio import open as rasterio_open
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.plot import reshape_as_image
from rasterio.transform import rowcol
from rasterio.warp import Resampling, reproject, transform
from shapely import Polygon
from shapely.ops import unary_union

from drone_grid_env import logger
from drone_grid_env.envs.io.kml import get_element_value, get_namespace
from drone_grid_env.envs.utils import LogProcessingTime

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from rasterio import DatasetReader

getLogger("rasterio").setLevel(WARNING)  # Ignore rasterio logging


@dataclass
class AdditionalObject:
    location: NDArray[np.float64]
    rotation: float
    image: NDArray[np.uint8]


class DynamicOrthomosaicLoader:
    def __init__(self, orthomosaic_scheme_file: Path, crs: str = "epsg:28992") -> None:
        self.tiles = get_tile_polygons(orthomosaic_scheme_file, crs=crs)

        self._loaded_raster: DatasetReader | None = None
        self._loaded_tiles: set[Path] | None = None
        self._loaded_polygon: Polygon | None = None

        self._data_transform: Affine | None = None
        self._data_buffer: NDArray[np.uint8] | None = None

        self._crs = crs

    def get_image_at_coordinate(
        self, coordinate: NDArray[np.float64], heading_deg: int, fov_px: tuple[int, int], fov_m: tuple[float, float]
    ) -> NDArray[np.uint8]:
        tiles_to_load = get_tiles_to_load(self.tiles, coordinate, heading_deg, fov_m)
        if len(tiles_to_load) == 0:
            logger.warn(f"UTM coordinate {coordinate} is not in any tile!")
            return np.zeros((*fov_px, 3), dtype=np.uint8)

        if self._loaded_tiles is None or not tiles_to_load.issubset(self._loaded_tiles):
            self._loaded_raster = load_tiles(list(tiles_to_load), parallel=True)
            self._loaded_tiles = tiles_to_load
            self._loaded_polygon = cast(Polygon, unary_union([self.tiles[t] for t in tiles_to_load]))

            self._data_heading = None
            self._data_buffer = None

            if self._loaded_raster.crs.linear_units != "metre":
                raise RuntimeError(
                    f"The loaded raster linear units '{self._loaded_raster.crs.linear_units}' are not in meters! Use an"
                    " orthomosaic with UTM coordinates."
                )

        assert self._loaded_raster is not None
        assert self._loaded_tiles is not None
        assert self._loaded_polygon is not None

        transform, array_size = self.calculate_raster_transform_fast(coordinate, heading_deg)

        if self._data_transform is None or not self._data_transform.almost_equals(transform):
            self._data_buffer = None
            self._data_transform = None

        if self._data_buffer is None or self._data_transform is None:
            with LogProcessingTime(logger.info, "Reprojected orthomosaic"):
                self._data_transform = transform
                data = np.zeros(array_size, dtype=self._loaded_raster.dtypes[0])

                reproject(
                    self._loaded_raster.read(),
                    data,
                    src_transform=self._loaded_raster.transform,
                    src_crs=self._loaded_raster.crs,
                    dst_transform=self._data_transform,
                    dst_crs=self._loaded_raster.crs,
                    dst_nodata=255,
                    resampling=Resampling.nearest,
                )

                self._data_buffer = np.ascontiguousarray(reshape_as_image(data))

        coordinate = self.transform_to_raster_coordinates(coordinate)
        drone_pixel_coordinates = rowcol(self._data_transform, *coordinate[:2])
        fov_px = (fov_m / get_orthomosaic_resolution(self._data_transform)).astype(np.uint16)

        img_data = extract_roi_from_orthomosaic(self._data_buffer, drone_pixel_coordinates, np.flip(fov_px), pad_value=255)

        return cv2.resize(img_data, fov_px, interpolation=cv2.INTER_NEAREST)  # type: ignore[return-value,no-any-return]

    def calculate_raster_transform_fast(self, coordinate: NDArray[np.float64], heading_deg: int) -> tuple[Affine, tuple[int, int, int]]:
        assert self._loaded_raster is not None

        coordinate = self.transform_to_raster_coordinates(coordinate)
        drone_pixel_coordinates = rowcol(self._loaded_raster.transform, *coordinate[:2])
        rotated_transform = self._loaded_raster.transform * Affine.rotation(heading_deg, pivot=np.flip(drone_pixel_coordinates))

        return rotated_transform, (self._loaded_raster.count, self._loaded_raster.height, self._loaded_raster.width)

    def transform_to_raster_coordinates(self, coordinate: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._loaded_raster is not None

        coordinate = coordinate.copy()
        if self._loaded_raster.crs.to_string() != self._crs:
            coordinate = np.column_stack(transform(self._crs, self._loaded_raster.crs.to_string(), [coordinate[0]], [coordinate[1]]))[0, :]

        return coordinate


def get_orthomosaic_resolution(transform: Affine) -> NDArray[np.float64]:
    resolution_x = (transform[0] ** 2 + transform[1] ** 2) ** 0.5
    resolution_y = (transform[3] ** 2 + transform[4] ** 2) ** 0.5
    return np.array([resolution_x, resolution_y], dtype=np.float64)


def get_tiles_to_load(
    tile_scheme: dict[Path, Polygon], coordinate: NDArray[np.float64], heading_deg: int, fov_m: tuple[float, float]
) -> set[Path]:
    tiles_to_load = []
    fov_polygon = get_fov_polygon(coordinate, heading_deg, fov_m)

    for tile_file, polygon in tile_scheme.items():
        if polygon.intersects(fov_polygon):
            tiles_to_load.append(tile_file)

    return set(sorted(tiles_to_load))


def get_fov_polygon(coordinates: NDArray[np.float64], heading_deg: int, fov_m: tuple[float, float]) -> Polygon:
    corners_relative = np.array(
        [
            [-fov_m[0] / 2, fov_m[1] / 2],
            [fov_m[0] / 2, fov_m[1] / 2],
            [fov_m[0] / 2, -fov_m[1] / 2],
            [-fov_m[0] / 2, -fov_m[1] / 2],
        ]
    )

    rotated_corners = corners_relative @ rotation_matrix(-heading_deg).T
    corners_utm = rotated_corners + coordinates

    return Polygon([corners_utm[i, :] for i in range(corners_utm.shape[0])])


def get_tile_polygons(kml_file: Path, crs: str = "EPSG:32631") -> dict[Path, Polygon]:
    with kml_file.open("rb") as kml_file_buffer:
        kml_root = ET.fromstring(kml_file_buffer.read())

    namespace = {"kml": get_namespace(kml_root)}

    tile_boundaries: dict[Path, Polygon] = {}

    for placemark in kml_root.findall(".//kml:Placemark", namespaces=namespace):
        name = get_element_value(placemark, "kml:name", namespaces=namespace)
        coordinates = get_element_value(placemark, "kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", namespaces=namespace)

        coordinates_array = np.array([np.fromstring(c, dtype=np.float64, sep=",") for c in coordinates.split()], dtype=np.float64)
        utm_eastings, utm_northings = transform("EPSG:4326", crs, coordinates_array[:, 0], coordinates_array[:, 1])
        polygon = Polygon([(e, n) for e, n in zip(utm_eastings, utm_northings)])

        tile_path = kml_file.parent / name
        if not tile_path.is_file():
            raise FileNotFoundError(f"Error: orthomosaic tile {tile_path.name} is missing!")

        tile_boundaries[tile_path] = polygon

    return tile_boundaries


def load_tiles(tiles_to_load: list[Path], parallel: bool = True) -> DatasetReader:
    logger.info(f"Loading tiles {[t.name for t in tiles_to_load]}")

    with LogProcessingTime(logger.info, "Loaded tiles"):
        if len(tiles_to_load) == 1:
            return rasterio_open(tiles_to_load[0])

        if parallel:
            with ThreadPoolExecutor() as executor:
                loaded_datasets = list(executor.map(load_tile, tiles_to_load))
        else:
            loaded_datasets = [load_tile(tile) for tile in tiles_to_load]

        mosaic, output_trans = merge(loaded_datasets)
        output_meta = loaded_datasets[0].meta.copy()
        output_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": output_trans,
            }
        )

        with MemoryFile() as memfile:
            with memfile.open(**output_meta) as writer:
                writer.write(mosaic)

            return rasterio_open(memfile)


def load_tile(tile_path: Path) -> DatasetReader:
    with rasterio_open(tile_path) as src:
        with MemoryFile() as memfile:
            with memfile.open(
                driver=src.driver,
                width=src.width,
                height=src.height,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=src.transform,
            ) as dataset:
                dataset.write(src.read())

            return rasterio_open(memfile)


def extract_roi_from_orthomosaic(
    orthomosaic_data: NDArray[np.uint8], roi_center: NDArray[np.uint16], size: NDArray[np.uint16], pad_value: int = 0
) -> NDArray[np.uint8]:
    roi = np.full((size[0], size[1], orthomosaic_data.shape[2]), pad_value, dtype=np.uint8)

    pad_size = np.maximum(size // 2 - roi_center, 0)
    roi_coordinates = np.concatenate(
        (
            np.clip(roi_center - size // 2, 0, orthomosaic_data.shape[:2]),
            np.clip(roi_center + size // 2, 0, orthomosaic_data.shape[:2]),
        )
    )

    orthomosaic_part = orthomosaic_data[roi_coordinates[0] : roi_coordinates[2], roi_coordinates[1] : roi_coordinates[3], :]
    if not np.any(np.array(orthomosaic_part.shape[:2]) == 0):
        roi[
            pad_size[0] : pad_size[0] + roi_coordinates[2] - roi_coordinates[0],
            pad_size[1] : pad_size[1] + roi_coordinates[3] - roi_coordinates[1],
            :,
        ] = orthomosaic_data[roi_coordinates[0] : roi_coordinates[2], roi_coordinates[1] : roi_coordinates[3], :]

    return roi


def rotation_matrix(heading_deg: float) -> NDArray[np.float64]:
    rad = np.deg2rad(heading_deg)
    return np.array(
        [
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)],
        ],
        dtype=np.float64,
    )
