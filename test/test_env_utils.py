from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from drone_grid_env.envs.utils import center_crop, coordinate_in_list, coordinates_in_size, pad_map, setdiff2d

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.mark.parametrize(
    "coordinates,size,result",
    [
        (np.array([[12, 9], [1, 1], [4, 76]]), (10, 10), np.array([False, True, False])),
        (np.array([[0, 0], [1, 1], [9, 9]]), (10, 10), np.array([True, True, True])),
        (np.array([[10, 10], [2, 7], [2, 10]]), (10, 10), np.array([False, True, False])),
    ],
)
def test_coordinates_in_size(coordinates: NDArray[np.int16], size: tuple[int, int], result: NDArray[np.int16]) -> None:
    np.testing.assert_array_equal(coordinates_in_size(coordinates, size), result)


@pytest.mark.parametrize(
    "coordinate,coordinates,result",
    [
        (np.array([10, 10]), np.array([[10, 10], [20, 23]]), True),
        (np.array([10, 10]), np.empty((0, 2)), False),
        (np.array([10, 10]), np.array([[11, 10], [20, 23]]), False),
    ],
)
def test_coordinates_in_list(coordinate: NDArray[np.int16], coordinates: NDArray[np.int16], result: bool) -> None:
    assert coordinate_in_list(coordinate, coordinates) == result


@pytest.mark.parametrize(
    "input_map,position,result",
    [
        (
            np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
            np.array([2, 1]),
            np.array([[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
            np.array([0, 0]),
            np.array([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 2, 3], [0, 0, 4, 5, 6], [0, 0, 7, 8, 9]]]),
        ),
    ],
)
def test_pad_map(input_map: NDArray[np.uint8], position: NDArray[np.uint16], result: NDArray[np.uint8]) -> None:
    np.testing.assert_array_equal(pad_map(input_map, position), result)


@pytest.mark.parametrize(
    "img,dim,result",
    [
        (
            np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]),
            (3, 3),
            np.array([[[7, 8, 9], [12, 13, 14], [17, 18, 19]]]),
        ),
        (
            np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]),
            (5, 5),
            np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]),
        ),
    ],
)
def test_center_crop(img: NDArray[np.uint8], dim: tuple[int, int], result: NDArray[np.uint8]) -> None:
    np.testing.assert_array_equal(center_crop(img, dim), result)


@pytest.mark.parametrize(
    "arr1,arr2,result",
    [
        (
            np.array([[10, 10], [20, 20], [30, 30]]),
            np.array([[10, 10], [30, 30]]),
            np.array([[20, 20]]),
        ),
        (
            np.array([[10, 10], [20, 20], [30, 30]]),
            np.array([[10, 10], [20, 20], [30, 30]]),
            np.array([]),
        ),
        (
            np.array([[10, 10], [30, 30]]),
            np.array([[10, 10], [20, 20], [30, 30]]),
            np.array([]),
        ),
    ],
)
def test_setdiff2d(arr1: NDArray[np.int16], arr2: NDArray[np.int16], result: NDArray[np.int16]) -> None:
    np.testing.assert_array_equal(setdiff2d(arr1, arr2), result)
