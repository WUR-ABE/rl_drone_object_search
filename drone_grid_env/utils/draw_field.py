#!/usr/bin/env python3
"""
Tool to rasterize a field and draw the field borders to create a world configuration file.
"""

from __future__ import annotations

from enum import Enum, auto
import os
from pathlib import Path
from tkinter import Tk, filedialog
from typing import TYPE_CHECKING, cast
from yaml import safe_dump

import numpy as np

from rasterio.warp import transform
from shapely import Point, Polygon, coverage_union_all
from tkintermapview import TkinterMapView
from tkintermapview.canvas_button import CanvasButton

from drone_grid_env import logger
from drone_grid_env.envs.io.orthomosaic import get_tile_polygons, rotation_matrix
from drone_grid_env.envs.io.topcon import read_topcon_data

if TYPE_CHECKING:
    from tkinter import Event

    from numpy.typing import NDArray

    from tkintermapview.canvas_position_marker import CanvasPositionMarker


class Stage(Enum):
    SET_GRID = "Set grid"
    SELECT_START_POINT = "Set start point"


class TileServers(Enum):
    OSM = auto()
    GOOGLE = auto()
    GOOGLE_SATELLITE = auto()
    ARCGIS = auto()


TILE_SERVER_URL = {
    TileServers.OSM: "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
    TileServers.GOOGLE: "https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga",
    TileServers.GOOGLE_SATELLITE: "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga",
    TileServers.ARCGIS: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}


class DrawFieldGUI(Tk):
    """
    GUI to select the field border and initial position of the drone. Internally, this class uses lat-lon notation since
    tkintermapview uses this format.
    """

    WIDTH = 1000
    HEIGHT = 700

    def __init__(
        self,
        scheme_file: Path | None = None,
        objects_file: Path | None = None,
        tile_server: TileServers = TileServers.ARCGIS,
        raster_cells: tuple[int, int] = (48, 48),
        initial_zoom: int = 20,
        initial_field_scale: float = 1.0,
        initial_heading: int = 0,
        crs: str = "epsg:32631",
    ) -> None:
        super().__init__()

        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")

        self._scheme_file = scheme_file
        self._objects_file = objects_file
        self._tile_server = tile_server
        self._raster_cells = np.array(raster_cells, dtype=np.uint16)
        self.zoom = initial_zoom
        self.field_scale = initial_field_scale
        self.raster_rotation = initial_heading
        self._crs = crs

        self.map_widget = TkinterMapView(self, width=1000, height=700, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)
        self.map_widget.set_zoom(self.zoom)
        self.map_widget.set_tile_server(TILE_SERVER_URL[self._tile_server])

        initial_map_center = np.array((51.99140, 5.66819), dtype=np.float64)

        self.scheme_file_polygon: NDArray[np.float64] | None = None
        if self._scheme_file is not None:
            polygon = cast(Polygon, coverage_union_all([p for p in get_tile_polygons(self._scheme_file, crs="EPSG:4326").values()]))
            self.scheme_file_polygon = np.flip(np.array(polygon.exterior.coords, dtype=np.float64), axis=1)
            initial_map_center = self.scheme_file_polygon.mean(axis=0)

        self.object_coordinates: NDArray[np.float64] | None = None
        if self._objects_file is not None:
            self.object_coordinates = np.flip(read_topcon_data(self._objects_file, output_crs="EPSG:4326"), axis=1)
            initial_map_center = self.object_coordinates.mean(axis=0)

        self.map_widget.set_position(*initial_map_center)

        self.continue_button = CanvasButton(self.map_widget, (self.WIDTH - 80, 20), text="→", command=self.next)
        self.decrease_scale_button = CanvasButton(self.map_widget, (self.WIDTH - 80, 60), text="s–", command=self._decrease_field_scale)
        self.increase_scale_button = CanvasButton(self.map_widget, (self.WIDTH - 80, 100), text="s+", command=self._increase_field_scale)
        self.rotate_counterclockwise_button = CanvasButton(
            self.map_widget, (self.WIDTH - 80, 140), text="↺", command=self._rotate_field_counterclockwise
        )
        self.rotate_clockwise_button = CanvasButton(self.map_widget, (self.WIDTH - 80, 180), text="↻", command=self._rotate_field_clockwise)

        self.raster_center_coordinate: NDArray[np.float64] = initial_map_center
        self.raster_polygon_coordinates: NDArray[np.float64] | None = None
        self.raster_cell_polygon_coordinates: NDArray[np.float64] | None = None
        self.start_position: NDArray[np.float64] | None = None

        self._move_marker = self.map_widget.set_marker(*initial_map_center, command=self._start_move_field)
        self._move_bindings: dict[str, str] = {}
        self._visible_markers: list[CanvasPositionMarker] = []

        self._current_stage = Stage.SET_GRID

        self._do_not_process_click = False

        self.title(self._current_stage.value)
        self.draw_grid_field()

    def next(self) -> None:
        self._do_not_process_click = True

        if self._current_stage == Stage.SET_GRID:
            self._current_stage = Stage.SELECT_START_POINT

            self._move_marker.delete()
            self._delete_button_from_map(self.decrease_scale_button)
            self._delete_button_from_map(self.increase_scale_button)
            self._delete_button_from_map(self.rotate_counterclockwise_button)
            self._delete_button_from_map(self.rotate_clockwise_button)

            self.title(self._current_stage.value)

            self.map_widget.add_left_click_map_command(self._set_start_point)

            self.draw_grid_field()

        elif self._current_stage == Stage.SELECT_START_POINT and self.start_position is not None:
            self.map_widget.map_click_callback = None
            self.draw_grid_field()

            self.save()
            self.quit()

    def draw_grid_field(self) -> None:
        field_size = self._raster_cells.astype(np.float32) * self.field_scale

        utm_raster_center = self.convert_to_crs(self.raster_center_coordinate.reshape(-1, 2))[0]
        utm_edges = np.array(
            [
                [utm_raster_center[0] - field_size[0] / 2, utm_raster_center[1] + field_size[1] / 2],
                [utm_raster_center[0] + field_size[0] / 2, utm_raster_center[1] + field_size[1] / 2],
                [utm_raster_center[0] + field_size[0] / 2, utm_raster_center[1] - field_size[1] / 2],
                [utm_raster_center[0] - field_size[0] / 2, utm_raster_center[1] - field_size[1] / 2],
            ],
            dtype=np.float64,
        )

        local_points = utm_edges - utm_raster_center
        local_points @= rotation_matrix(self.raster_rotation).T
        utm_edges = local_points + utm_raster_center

        utm_cell_edges = self.generate_grid_coordinates(utm_edges, self._raster_cells[0], self._raster_cells[1])

        self.raster_polygon_coordinates = self.convert_from_crs(utm_edges)
        self.raster_cell_polygon_coordinates = self.convert_from_crs(utm_cell_edges)

        self.clear_field()

        if self.scheme_file_polygon is not None:
            self.map_widget.set_polygon(self.scheme_file_polygon, outline_color="yellow", fill_color=None)

        if self.object_coordinates is not None:
            for c in self.object_coordinates:
                self._add_text(*c, "x", color="#FFA500")  # type: ignore[call-arg,misc]

        if self.start_position is not None:
            self._visible_markers.append(
                self.map_widget.set_marker(*self.start_position, marker_color_circle="#627ba4", marker_color_outside="#0099ff")
            )

        # Show raster
        self.map_widget.set_polygon(self.raster_polygon_coordinates, outline_color="red", fill_color=None)

        for i in range(0, self.raster_cell_polygon_coordinates.shape[0], 4):
            fill_color = None
            outline_color = "red"

            self.map_widget.set_polygon(
                self.raster_cell_polygon_coordinates[i : i + 4, :],
                outline_color=outline_color,
                fill_color=fill_color,
                border_width=1,
            )

    def clear_field(self) -> None:
        self.map_widget.delete_all_polygon()

        for marker in self._visible_markers:
            marker.delete()

        self._visible_markers

    def _add_text(self, deg_x: float, deg_y: float, text: str, color: str = "#ffffff") -> None:
        marker = self.map_widget.set_marker(
            deg_x,
            deg_y,
            marker_color_circle="",
            marker_color_outside="",
            text=text,
            text_color=color,
            icon_anchor="s",
        )

        marker.text_y_offset = 0
        marker.draw()
        self._visible_markers.append(marker)

    def _set_start_point(self, coordinates: tuple[float, float]) -> None:
        if self._do_not_process_click:
            self._do_not_process_click = False
            return

        if not Polygon(self.raster_polygon_coordinates).contains(Point(coordinates)):
            logger.info("Cannot place start point outside grid!")
            return

        self.map_widget.map_click_callback = None
        self.start_position = np.array(coordinates, dtype=np.float64)

        self._move_marker = self.map_widget.set_marker(
            *coordinates, command=self._start_move_start_point, marker_color_circle="#627ba4", marker_color_outside="#0099ff"
        )

    def _start_move_start_point(self, marker: CanvasPositionMarker) -> None:
        if len(self._move_bindings) == 0:
            self._move_bindings["<B1-Motion>"] = self.map_widget.canvas.bind("<B1-Motion>", self._move_start_point)
            self._move_bindings["<ButtonRelease-1>"] = self.map_widget.canvas.bind("<ButtonRelease-1>", self._stop_move)

    def _move_start_point(self, event: Event) -> None:
        self.start_position = np.array(self.map_widget.convert_canvas_coords_to_decimal_coords(event.x, event.y), dtype=np.float64)
        self._move_marker.set_position(*self.start_position)

    def _start_move_field(self, marker: CanvasPositionMarker) -> None:
        if len(self._move_bindings) == 0:
            self._move_bindings["<B1-Motion>"] = self.map_widget.canvas.bind("<B1-Motion>", self._move_field)
            self._move_bindings["<ButtonRelease-1>"] = self.map_widget.canvas.bind("<ButtonRelease-1>", self._stop_move)

    def _move_field(self, event: Event) -> None:
        self.raster_center_coordinate = np.array(
            self.map_widget.convert_canvas_coords_to_decimal_coords(event.x, event.y), dtype=np.float64
        )
        self._move_marker.set_position(*self.raster_center_coordinate)
        self.draw_grid_field()

    def _stop_move(self, event: Event) -> None:
        for k, v in self._move_bindings.items():
            self.map_widget.canvas.unbind(k, v)

        # Bind events to map again
        self.map_widget.canvas.bind("<B1-Motion>", self.map_widget.mouse_move)
        self.map_widget.canvas.bind("<ButtonRelease-1>", self.map_widget.mouse_release)

        self._move_bindings.clear()

    def _increase_field_scale(self) -> None:
        self.field_scale += 0.1
        logger.info(f"Set grid-cell size to {self.field_scale:.1f}x{self.field_scale:.1f}m")

        self.draw_grid_field()

    def _decrease_field_scale(self) -> None:
        self.field_scale -= 0.1
        logger.info(f"Set grid-cell size to {self.field_scale:.1f}x{self.field_scale:.1f}m")

        self.draw_grid_field()

    def _rotate_field_clockwise(self) -> None:
        self.raster_rotation -= 5
        logger.info(f"Set grid-cell rotation to {self.raster_rotation} deg")

        self.draw_grid_field()

    def _rotate_field_counterclockwise(self) -> None:
        self.raster_rotation += 5
        logger.info(f"Set grid-cell rotation to {self.raster_rotation} deg")

        self.draw_grid_field()

    def _delete_button_from_map(self, button: CanvasButton) -> None:
        button.command = None
        self.map_widget.canvas.delete(button.canvas_rect)
        self.map_widget.canvas.delete(button.canvas_text)

    def convert_to_local(self, utm_coordinates: NDArray[np.float64], heading_deg: float) -> NDArray[np.int16]:
        utm_raster_center = self.convert_to_crs(self.raster_center_coordinate.reshape(-1, 2))[0, :]

        coordinates_local = utm_coordinates.copy()
        coordinates_local -= utm_raster_center
        coordinates_local[:, 1] *= -1  # UTM has north up
        coordinates_local /= self.field_scale
        coordinates_local = coordinates_local[:, [1, 0]]
        coordinates_local @= rotation_matrix(-heading_deg).T
        coordinates_local += (self._raster_cells - 1) / 2
        return coordinates_local.round().astype(np.int16)

    def convert_from_local(self, local_coordinates: NDArray[np.int16], heading_deg: float) -> NDArray[np.float64]:
        utm_raster_center = self.convert_to_crs(self.raster_center_coordinate.reshape(-1, 2))[0, :]

        utm_coordinates = local_coordinates.copy().astype(np.float64)
        utm_coordinates -= (self._raster_cells - 1) / 2
        utm_coordinates @= rotation_matrix(heading_deg).T  # Rotate back
        utm_coordinates = utm_coordinates[:, [1, 0]]
        utm_coordinates *= self.field_scale
        utm_coordinates[:, 1] *= -1  # UTM has north up
        utm_coordinates += utm_raster_center
        return utm_coordinates

    def convert_to_crs(self, gps_coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack(transform("EPSG:4326", self._crs, gps_coordinates[:, 1], gps_coordinates[:, 0]))

    def convert_from_crs(self, crs_coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.flip(np.column_stack(transform(self._crs, "EPSG:4326", crs_coordinates[:, 0], crs_coordinates[:, 1])), axis=1)

    def save(self) -> None:
        assert self.start_position is not None
        assert self.raster_polygon_coordinates is not None

        utm_raster_center = self.convert_to_crs(self.raster_center_coordinate.reshape(-1, 2))[0]

        save_path = Path(filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]))
        with save_path.open("w") as file_writer:
            config = {
                "world": {
                    "type": "OrthomosaicWorld",
                    "crs": self._crs,
                    "name": save_path.stem,
                    "objects_file": None if self._objects_file is None else self.create_path(str(self._objects_file.resolve())),
                    "scheme_file": None if self._objects_file is None else self.create_path(str(self._scheme_file.resolve())),
                    "size": {
                        "width": int(self._raster_cells[0]),
                        "height": int(self._raster_cells[1]),
                    },
                    "scale": self.field_scale,
                    "rotation": self.raster_rotation,
                    "origin": {
                        "easting": float(utm_raster_center[0]),
                        "northing": float(utm_raster_center[1]),
                    },
                    "start_position": {
                        "easting": float(self.convert_to_crs(self.start_position.reshape(1, -1))[0, 0]),
                        "northing": float(self.convert_to_crs(self.start_position.reshape(1, -1))[0, 1]),
                    },
                    "start_landing_zones": [],
                    "camera": {
                        "width": 2048,
                        "height": 2048,
                    },
                    "yolo_weights_file": "best_n.pt",
                    "min_conf": 0.05,
                    "prior_knowledge": {
                        "camera": {
                            "width": 2048,
                            "height": 2048,
                        },
                        "coverage_fov": {
                            "width": 48,
                            "height": 48,
                        },
                        "size": {
                            "width": 12,
                            "height": 12,
                        },
                        "confidence_threshold": 0.05,
                    },
                },
                "stopping_criteria": {"method": "land", "value": 1.0, "only_land_in_zone": False},
                "action_space": {"allow_diagonals": False, "land_action": True},
                "drone": {
                    "battery_usage_flying": 0.2,
                    "battery_usage_landing": 0.2,
                    "confidence_threshold": 0.5,
                    "fov": {"height": 11, "width": 11},
                    "initial_battery_level": 75,
                    "start_mode": "in_start_zone",
                },
                "rewards": {
                    "coverage": 0.0,
                    "discovered_objects_tp": 1.0,
                    "discovered_objects_fp": 0.0,
                    "empty_battery": -150.0,
                    "fly_action": -0.5,
                    "hit_no_fly_zone": -1.0,
                    "undiscovered_objects": 0.0,
                    "normalize_objects": True,
                },
                "state_space": {"global_map_reduce": 3, "add_start_landing_zone": False},
            }
            safe_dump(config, file_writer, indent=2)

    @staticmethod
    def create_path(path: str) -> str:
        if data_home := os.environ.get("DATA_HOME"):
            if path.startswith(data_home):
                return path.replace(data_home, "$DATA_HOME")
        return path

    @staticmethod
    def generate_grid_coordinates(points: NDArray[np.float64], ncols: int, nrows: int) -> NDArray[np.float64]:
        left_edge = np.linspace(points[0], points[3], nrows + 1)
        right_edge = np.linspace(points[1], points[2], nrows + 1)

        cols_fraction = np.linspace(0, 1, ncols + 1).reshape(1, -1, 1)
        top_edges = left_edge[:-1].reshape(-1, 1, 2) * (1 - cols_fraction) + right_edge[:-1].reshape(-1, 1, 2) * cols_fraction
        bottom_edges = left_edge[1:].reshape(-1, 1, 2) * (1 - cols_fraction) + right_edge[1:].reshape(-1, 1, 2) * cols_fraction

        top_left = top_edges[:, :-1]
        top_right = top_edges[:, 1:]
        bottom_right = bottom_edges[:, 1:]
        bottom_left = bottom_edges[:, :-1]

        return np.stack([top_left, top_right, bottom_right, bottom_left], axis=2).reshape(-1, 2)


def main() -> None:
    from tap import ArgumentError, Tap

    from rasterio.crs import CRS, CRSError

    class ArgumentParser(Tap):
        scheme_file: Path | None = None  # Path to the orthomosaic scheme file
        objects_file: Path | None = None  # Path to the objects file
        tile_server: TileServers = TileServers.ARCGIS  # Tile server
        raster_cells: tuple[int, int] = (48, 48)  # Size of the discrete raster (width, height)
        initial_zoom: int = 20  # Initial zoom level for the map
        initial_field_scale: float = 1.0  # Initial field scale (m per grid cell)
        initial_heading: int = 0  # Initial heading (degree)
        crs: str = "epsg:32631"  # CRS of the output file (should be metric)

        def configure(self) -> None:
            self.add_argument("--tile_server", type=str, choices=[e.name.lower() for e in TileServers])

        def process_args(self) -> None:
            if isinstance(self.tile_server, str):
                self.tile_server = TileServers[self.tile_server.upper()]

            try:
                CRS.from_string(self.crs)
            except CRSError:
                raise ArgumentError(self._get_optional_actions()[-2], f"CRS '{self.crs}' is not valid!")

    args = ArgumentParser().parse_args()

    gui = DrawFieldGUI(
        scheme_file=args.scheme_file,
        objects_file=args.objects_file,
        tile_server=args.tile_server,
        raster_cells=args.raster_cells,
        initial_zoom=args.initial_zoom,
        initial_field_scale=args.initial_field_scale,
        initial_heading=args.initial_heading,
        crs=args.crs,
    )
    gui.mainloop()


if __name__ == "__main__":
    main()
