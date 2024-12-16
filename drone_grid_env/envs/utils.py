from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any, Callable, Generator

    from numpy.typing import NDArray

    from drone_grid_env.envs.drone import Drone
    from drone_grid_env.envs.world.base import BaseWorld


@dataclass
class Distribution:
    dist_type: str
    dist_kwargs: dict[str, Any] = field(default_factory=dict)

    def get_dist_value(self, rng: np.random.Generator, **kwargs: Any) -> NDArray[np.float32]:
        return np.array(getattr(rng, self.dist_type)(**self.dist_kwargs, **kwargs))

    def set_value(self, name: str, value: Any) -> None:
        self.dist_kwargs[name] = value

    def get_value(self, name: str) -> Any:
        return self.dist_kwargs[name]

    def contains_value(self, name: str) -> bool:
        return name in self.dist_kwargs

    def get_class(self, rng: np.random.Generator) -> type[Any]:
        return getattr(rng, self.dist_type)  # type: ignore[no-any-return]


@dataclass
class Zone:
    x: int
    y: int
    h: int
    w: int

    @property
    def coordinates(self) -> NDArray[np.uint16]:
        return np.mgrid[self.x : self.x + self.h, self.y : self.y + self.w].reshape(2, -1).T

    @property
    def shape(self) -> tuple[int, int]:
        return (self.w, self.h)

    def contains(self, coordinate: NDArray[np.uint16]) -> bool:
        return coordinate_in_list(coordinate, self.coordinates)


@dataclass
class FlightPath:
    positions: list[NDArray[np.uint16]] = field(default_factory=list)

    @property
    def current_position(self) -> NDArray[np.uint16]:
        return self.positions[-1]

    def add(self, position: NDArray[np.uint16]) -> None:
        self.positions.append(position)

    def reset(self) -> None:
        self.positions.clear()

    def get_length(self) -> float:
        length = 0.0
        for i in range(0, len(self.positions) - 1):
            length += float(np.linalg.norm(self.positions[i].astype(np.int16) - self.positions[i + 1]))
        return length

    def to_array(self) -> NDArray[np.uint16]:
        return np.array(self.positions, dtype=np.uint16)

    def __iter__(self) -> Generator[NDArray[np.uint16], None, None]:
        return (self.positions[i] for i in range(len(self.positions)))

    def __len__(self) -> int:
        return len(self.positions)


class LogProcessingTime:
    def __init__(self, log_fn: Callable[[str], None], msg: str) -> None:
        self.log_fn = log_fn
        self.msg = msg
        self.start_time: float | None = None

    def __enter__(self) -> LogProcessingTime:
        self.start_time = time()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        assert self.start_time is not None

        self.log_fn(f"{self.msg} in {time()-self.start_time:.2f} seconds")


def coordinates_in_size(coordinates: NDArray[np.integer[Any]], size: tuple[int, int]) -> NDArray[np.bool_]:
    return (coordinates[:, 0] >= 0) & (coordinates[:, 0] < size[0]) & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < size[1])


def coordinate_in_list(coordinate: NDArray[np.integer[Any]], coordinates: NDArray[np.integer[Any]]) -> bool:
    if coordinates.shape == (2,):
        return False
    return any(np.equal(coordinates, coordinate).all(1))


def pad_map(
    input_map: NDArray[np.uint8],
    position: NDArray[np.uint16],
    pad_value: int | tuple[int, ...] = 0,
    first_axis: bool = True,
) -> NDArray[np.uint8]:
    """
    Pads the map till size 2M-1.

    :param input_map: The map to pad.
    :param position: The position around which the map needs to be center padded.
    :param pad_value: The value which is used to pad the map.
    :param first_axis: True if the channels are on the first axis, False if the channels are on the last axis.
    """
    # Pad in a way that the position is in the center of the map
    padding: NDArray[np.uint32] = np.array(input_map.shape[1:] if first_axis else input_map.shape[:2], dtype=np.uint32)
    position_offset = padding - position

    if first_axis:
        pad_width = [
            [0, 0],
            [int(position_offset[0] - 1), int(padding[0] - position_offset[0])],
            [int(position_offset[1] - 1), int(padding[1] - position_offset[1])],
        ]
    else:
        pad_width = [
            [int(position_offset[0] - 1), int(padding[0] - position_offset[0])],
            [int(position_offset[1] - 1), int(padding[1] - position_offset[1])],
            [0, 0],
        ]

    centered_map: NDArray[np.uint8] = np.pad(
        input_map,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value,
    )

    # Check size
    if first_axis:
        np.testing.assert_equal(centered_map.shape[1:], 2 * np.array(input_map.shape[1:]) - 1)
    else:
        np.testing.assert_equal(centered_map.shape[:2], 2 * np.array(input_map.shape[:2]) - 1)
    return centered_map


def center_crop(img: NDArray[np.uint8], dim: tuple[int, int], first_axis: bool = True) -> NDArray[np.uint8]:
    """
    Crops the image around the center.

    Source: https://gist.github.com/Nannigalaxy/35dd1d0722f29672e68b700bc5d44767

    :param img: Input image.
    :param dim: The dimensions to crop.
    :param first_axis: True if the channels are on the first axis, False if the channels are on the last axis.
    """
    if first_axis:
        height, width = img.shape[1:]
    else:
        height, width = img.shape[:2]

    startx = height // 2 - dim[0] // 2
    starty = width // 2 - dim[1] // 2

    if first_axis:
        return img[:, startx : startx + dim[0], starty : starty + dim[1]]

    return img[startx : startx + dim[0], starty : starty + dim[1], :]


def block_mean(img: NDArray[np.uint8], block_size: tuple[int, int] | NDArray[np.signedinteger[Any]]) -> NDArray[np.uint8]:
    assert img.shape[0] % block_size[0] == 0 and img.shape[1] % block_size[1] == 0, "Block size should exactly match input size"

    n_blocks = np.array(img.shape[:2]) // block_size
    blocks = img[: n_blocks[0] * block_size[0], : n_blocks[1] * block_size[1]].reshape(n_blocks[0], block_size[0], n_blocks[1], block_size[1])
    values = np.mean(blocks, axis=(1, 3), dtype=np.double).astype(np.uint8)
    img_mean = np.repeat(np.repeat(values, block_size[0], axis=0), block_size[1], axis=1)

    assert np.unique(img_mean.reshape(-1)).shape[0] <= n_blocks[0] * n_blocks[1]
    return img_mean


def setdiff2d(arr1: NDArray[Any], arr2: NDArray[Any]) -> NDArray[Any]:
    """
    Compares two lists of coordinates and returns the coordinates that are present in
    arr1 but not in arr2.
    """
    set1 = set(map(tuple, arr1))
    set2 = set(map(tuple, arr2))
    return np.array(list(set1 - set2), dtype=arr1.dtype)


def count_gt_classified_objects(world: BaseWorld, drone: Drone) -> int:
    remaining_object_map = np.zeros(world.object_map.shape[:2], dtype=np.uint8)
    remaining_object_map[world.object_map] = 255
    if drone.found_object_positions.shape[0] > 0:
        remaining_object_map[drone.found_object_positions[:, 0], drone.found_object_positions[:, 1]] = 0
    return world.number_of_objects - np.count_nonzero(remaining_object_map)


def parse_input(input: str) -> str | int | float | bool:
    try:
        return int(input)
    except ValueError:
        try:
            return float(input)
        except ValueError:
            if input.lower() == "false" or input.lower() == "true":
                return bool(input)
            return input
