from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from csv import DictReader, writer
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

EVALUATION_FILE_HEADER = [
    "episode",
    "step",
    "path_length",
    "counted_objects",
    "counted_objects_percentage",
    "coverage_percentage",
    "battery_level",
    "reward",
]


def read_csv_by_column_name(
    data_file: Path,
    column_name: str = "counted_objects",
    n_steps: int = 300,
    expected_n_episodes: int = 1000,
    value_min: float | None = None,
    value_max: float | None = None,
    dtype: type[np.floating] = np.float32,
) -> NDArray[np.floating]:
    """
    :param data_file: CSV file path.
    :param column_name: Name of the column to read from.
    :param n_steps: Number of steps to read. If higher than number of stored steps, last value is repeated.
    :param expected_n_episodes: Expected number of episodes.
    :returns: Numpy array of the requested data.
    """
    # n_steps +=1  # Reset is already step 0, so we have to add 1

    data = defaultdict(list)
    with data_file.open("r") as data_file_handler:
        reader = DictReader(data_file_handler)

        for row in reader:
            data[row["episode"]].append(row[column_name])

    assert len(data) == expected_n_episodes, f"Number of episodes '{len(data)}' does not match expected number '{expected_n_episodes}'!"

    for episode, episode_data in data.items():
        if len(episode_data) > n_steps:
            data[episode] = episode_data[:n_steps]
        elif len(episode_data) < n_steps:
            while len(episode_data) < n_steps:
                episode_data.append(episode_data[-1])

    arr = np.array([data[episode] for episode in sorted(data.keys())], dtype=dtype)

    if value_min is not None or value_max is not None:
        arr = np.clip(arr, value_min, value_max)

    return arr


def write_csv(output_file: Path, data: list[list[int | float]]) -> None:
    header = deepcopy(EVALUATION_FILE_HEADER)
    header.extend([f"action_value_{i}" for i in range(len(data[0]) - len(EVALUATION_FILE_HEADER))])

    with output_file.open("w") as evaluation_file_writer:
        csv_writer = writer(evaluation_file_writer)
        csv_writer.writerow(header)
        csv_writer.writerows(data)
