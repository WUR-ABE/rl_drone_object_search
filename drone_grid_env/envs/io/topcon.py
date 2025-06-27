from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rasterio.warp import transform

if TYPE_CHECKING:
    from numpy.typing import NDArray


def read_topcon_data(topcon_file: Path, file_crs: str = "epsg:28992", output_crs: str = "EPSG:32631") -> NDArray[np.float64]:
    with topcon_file.open("r") as filereader:
        header = next(filereader)
        is_latlon = header.index("Lon(East)") > header.index("Lat(North)")

        coordinates = []
        for line in filereader.readlines():
            line_content = line.split(",")

            rd_x = line_content[2] if is_latlon else line_content[1]
            rd_y = line_content[1] if is_latlon else line_content[2]

            coordinates.append([rd_x, rd_y])

        coordinates_array = np.array(coordinates, dtype=np.float64)
        utm_eastings, utm_northings = transform(file_crs, output_crs, coordinates_array[:, 0], coordinates_array[:, 1])

    return np.column_stack([utm_eastings, utm_northings])
