from __future__ import annotations

from pathlib import Path
from re import search
from typing import TYPE_CHECKING, Literal

import matplotlib
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import numpy as np

from pygame import Surface
from pygame.surfarray import array3d
from rasterio.warp import transform
from tilemapbase import Extent, Plotter, init, project, to_lonlat
from tilemapbase.tiles import Tiles, build_OSM

from drone_grid_env.envs.drone_grid_env import DroneGridEnv
from drone_grid_env.envs.utils import FlightPath, get_file, intersect2d, setdiff2d, transform_polygon
from drone_grid_env.envs.world.orthomosaic_world import OrthomosaicWorld
from drone_grid_env.envs.world.sim_world import SimWorld

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from shapely import Polygon

matplotlib.rcParams["text.usetex"] = True

init(create=True)

LINE_STYLES = ["-", "--", "-.", ":"]

# fmt: off
COLORS = {
    1:  ["#800000"],
    2:  ["#800000", "#3cb44b"],
    3:  ["#800000", "#3cb44b", "#4363d8"],
    4:  ["#800000", "#3cb44b", "#4363d8", "#a9a9a9"],
    5:  ["#800000", "#3cb44b", "#4363d8", "#a9a9a9", "#42d4f4"],
    6:  ["#800000", "#3cb44b", "#4363d8", "#a9a9a9", "#42d4f4", "#f58231"],
    7:  ["#800000", "#3cb44b", "#4363d8", "#a9a9a9", "#42d4f4", "#f58231", "#f032e6"],
    8:  ["#800000", "#3cb44b", "#4363d8", "#a9a9a9", "#42d4f4", "#f58231", "#f032e6", "#000075"],
    9:  ["#800000", "#3cb44b", "#4363d8", "#a9a9a9", "#42d4f4", "#f58231", "#f032e6", "#000075", "#fffac8"],
    10: ["#800000", "#3cb44b", "#4363d8", "#a9a9a9", "#42d4f4", "#f58231", "#f032e6", "#000075", "#fffac8", "#aaffc3"],
    11: ["#800000", "#3cb44b", "#4363d8", "#a9a9a9", "#42d4f4", "#f58231", "#f032e6", "#000075", "#fffac8", "#aaffc3", "#808000"],
    12: ["#800000", "#3cb44b", "#4363d8", "#a9a9a9", "#42d4f4", "#f58231", "#f032e6", "#000075", "#fffac8", "#aaffc3", "#808000", "#469990"],
}
# fmt: on


def get_tiles(tile_type: Literal["OSM", "ArcGis"]) -> Tiles:
    if tile_type == "OSM":
        return build_OSM()
    elif tile_type == "ArcGis":
        ARCGIS_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
        return Tiles(ARCGIS_URL, "satellite")
    else:
        raise NotImplementedError(f"Tile type {tile_type} is not implemented!")


def calculate_mean_std(data: NDArray[np.float32]) -> NDArray[np.float32]:
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return np.column_stack((means, stds))


def mean_and_std(data: NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.empty((data.shape[1], 2), dtype=np.float32)
    arr[:, 0] = np.mean(data, axis=0)
    arr[:, 1] = np.std(data, axis=0)
    return arr


def create_line_plot(
    data: list[NDArray[np.float32]],
    ax: Axes,
    labels: list[str] | None = None,
    groups: list[list[str]] | None = None,
    unique_color: list[str] | None = None,
    conf_interval: bool = False,
    custom_colors: list[str] | None = None,
    xlim: tuple[float, float] = (0, 300),
    ylim: tuple[float, float] = (0, 200),
    xlabel: str = "Flight path length",
    ylabel: str = "Number of found objects",
    legend_loc: str | None = "lower left",
    legend_kwargs: dict[str, Any] = {},
) -> None:
    if labels is None:
        assert groups is None, "Labels needs to be defined to use groups!"

        labels = [f"Line {i}" for i in range(len(data))]

    assert len(data) == len(labels), f"Number of data arrays {len(data)} and labels {len(labels)} should match!"

    linestyle = {member: LINE_STYLES[0] for member in labels}
    colors = {member: COLORS[len(labels)][i] for i, member in enumerate(labels)}

    if custom_colors is not None:
        assert len(custom_colors) == len(labels), f"Number of custom colors {len(custom_colors)} and labels {len(labels)} should match!"
        colors = {member: custom_colors[i] for i, member in enumerate(labels)}

    if groups is not None:
        assert all(all(m in labels for m in g) for g in groups), "Not all group members are in labels!"
        assert all(m in [mg for mgs in groups for mg in mgs] for m in labels), "All labels should be in groups!"

        if unique_color is None:
            unique_color = []

        assert all(u in labels for u in unique_color), "Not all unique color members are in labels!"

        # Overwrite linestyle by group
        linestyle = {member: LINE_STYLES[i] for i, group in enumerate(groups) for member in group}

        if custom_colors is None:
            colors = {
                member: COLORS[len(group) + len(unique_color)][i + len(unique_color)] for group in groups for i, member in enumerate(group)
            }

            n = len(set(list(colors.values())))
            for m, c in colors.items():
                if m in unique_color:
                    colors[m] = COLORS[n + len(unique_color)][unique_color.index(m)]

    for i, d in enumerate(data):
        mean_std = calculate_mean_std(d)

        ax.plot(
            np.arange(*xlim),
            mean_std[:, 0],
            label=labels[i],
            color=colors[labels[i]],
            linestyle=linestyle[labels[i]],
        )

        if conf_interval:
            ax.fill_between(np.arange(*xlim), mean_std[:, 0] - mean_std[:, 1], mean_std[:, 0] + mean_std[:, 1], alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True)

    if legend_loc:
        ax.legend(loc=legend_loc, **legend_kwargs)


def create_bar_plot(
    data: list[NDArray[np.float32]],
    ax: Axes,
    labels: list[str] | None = None,
    groups: list[list[str]] | None = None,
    group_labels: list[str] | None = None,
    category_labels: list[str] | None = None,
    custom_colors: list[str] | None = None,
    std: bool = True,
    timestep: int = 400,
    ylim: tuple[float, float] = (0, 200),
    ylabel: str = "value",
    legend_loc: str = "lower left",
    secondary_axis_groups: list[str] = [],
    secondary_axis_label: str = "",
    secondary_axis_ylim: tuple[float, float] = (0.0, 1.0),
) -> None:
    if labels is None:
        assert groups is None and group_labels is None and category_labels is None, "Labels needs to be defined to use groups!"

        labels = [f"Line {i}" for i in range(len(data))]

    if groups is not None:
        assert all(all(m in labels for m in g) for g in groups), "Not all group members are in labels!"
        assert all(m in [mg for mgs in groups for mg in mgs] for m in labels), "All labels should be in groups!"
        assert len({len(sublist) for sublist in groups}) == 1, "All groups should have equal length!"

    if groups is None:
        groups = [labels]

    if group_labels is None:
        group_labels = [f"Group {i + 1}" for i in range(len(groups))]

    if category_labels is None:
        category_labels = [f"Category {i + 1}" for i in range(len(groups[0]))]

    assert len(group_labels) == len(groups), "Length of group labels should match the number of groups!"
    assert len(category_labels) == len(groups[0]), "Length of category labels should match the length of the groups!"

    value_data = []
    error_data = []

    for group in groups:
        group_values = []
        group_errors = []

        for member in group:
            mean_std = mean_and_std(data[labels.index(member)])[timestep - 1]
            group_values.append(mean_std[0])
            group_errors.append(mean_std[1])

        value_data.append(group_values)
        error_data.append(group_errors)

    if len(secondary_axis_groups) > 0:
        secondary_axis = ax.twinx()

    x = np.arange(len(groups[0]))
    bar_width = 1 / (len(groups) + 1)

    for i in range(len(groups)):
        color = custom_colors[i] if custom_colors else None
        _ax = secondary_axis if group_labels[i] in secondary_axis_groups else ax

        if std:
            _ax.bar(
                x + (i - (len(groups) - 1) / 2) * bar_width,
                value_data[i],
                bar_width,
                yerr=error_data[i],
                label=group_labels[i],
                capsize=2,
                color=color,
            )
        else:
            _ax.bar(x + (i - (len(groups) - 1) / 2) * bar_width, value_data[i], bar_width, label=group_labels[i], capsize=2, color=color)

    ax.set_xticks(x, category_labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True)

    if len(secondary_axis_groups) > 0:
        secondary_axis.set_ylabel(secondary_axis_label)
        secondary_axis.set_ylim(*secondary_axis_ylim)

    # Legend should be drawn on the last initialised axis: https://github.com/matplotlib/matplotlib/issues/3706
    if len(group_labels) > 1:
        _ax = secondary_axis if len(secondary_axis_groups) > 0 else ax
        _ax.legend(group_labels, loc=legend_loc)


def create_box_plot(
    data: list[NDArray[np.float32]],
    ax: Axes,
    labels: list[str] | None = None,
    groups: list[list[str]] | None = None,
    group_labels: list[str] | None = None,
    category_labels: list[str] | None = None,
    custom_colors: list[str] | None = None,
    showoutliers: bool = True,
    ylim: tuple[float, float] = (0, 200),
    ylabel: str = "value",
    legend_loc: str = "lower left",
    secondary_axis_groups: list[str] = [],
    secondary_axis_label: str = "",
    secondary_axis_ylim: tuple[float, float] = (0.0, 1.0),
) -> None:
    if labels is None:
        assert groups is None and group_labels is None and category_labels is None, "Labels needs to be defined to use groups!"

        labels = [f"Line {i}" for i in range(len(data))]

    if groups is not None:
        assert all(all(m in labels for m in g) for g in groups), "Not all group members are in labels!"
        assert all(m in [mg for mgs in groups for mg in mgs] for m in labels), "All labels should be in groups!"
        assert len({len(sublist) for sublist in groups}) == 1, "All groups should have equal length!"

    if groups is None:
        groups = [labels]

    if group_labels is None:
        group_labels = [f"Group {i + 1}" for i in range(len(groups))]

    if category_labels is None and len(groups) == 1:
        category_labels = labels
    elif category_labels is None:
        category_labels = [f"Category {i + 1}" for i in range(len(groups[0]))]

    assert len(group_labels) == len(groups), "Length of group labels should match the number of groups!"
    assert len(category_labels) == len(groups[0]), "Length of category labels should match the length of the groups!"

    box_data = []

    for group in groups:
        group_values = []

        for member in group:
            group_values.append(data[labels.index(member)])

        box_data.append(group_values)

    x = np.arange(len(groups[0]))
    bar_width = 1 / (len(groups) + 1)
    bps = []

    if len(secondary_axis_groups) > 0:
        secondary_axis = ax.twinx()

    for i in range(len(groups)):
        color = custom_colors[i] if custom_colors else None
        _ax = secondary_axis if group_labels[i] in secondary_axis_groups else ax

        bp = _ax.boxplot(
            box_data[i],
            positions=x + (i - (len(groups) - 1) / 2) * bar_width,
            widths=bar_width,
            patch_artist=True,
            showfliers=showoutliers,
            boxprops=dict(color="black", facecolor=color),
            medianprops=dict(color="black"),
        )
        bps.append(bp)

    ax.set_xticks(x, category_labels, rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True)

    if len(secondary_axis_groups) > 0:
        secondary_axis.set_ylabel(secondary_axis_label)
        secondary_axis.set_ylim(*secondary_axis_ylim)

    # Legend should be drawn on the last initialised axis: https://github.com/matplotlib/matplotlib/issues/3706
    if len(group_labels) > 1:
        _ax = secondary_axis if len(secondary_axis_groups) > 0 else ax
        _ax.legend([bp["boxes"][0] for bp in bps], group_labels, loc=legend_loc)


def create_flight_map(
    flight_path: list[NDArray[np.uint16]],
    config_file: str | Path,
    tile_type: Literal["OSM", "ArcGis"] = "ArcGis",
    figsize: tuple[int, int] = (4, 4),
    save_path: str | Path | None = None,
    seed: int | None = None,
) -> None:
    env = DroneGridEnv(config_file=config_file, render_mode="rgb_array_headless")
    env.reset(seed=seed)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    plt.axis("off")

    if isinstance(env.world, OrthomosaicWorld):
        orthomosaic2matplotlib(flight_path, env, ax, tile_type=tile_type)
    elif isinstance(env.world, SimWorld):
        env.drone._flight_path = FlightPath(initial_path=flight_path)
        sim2matplotlib(env, ax, plot_drone=True)
    else:
        raise NotImplementedError

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, format=str(save_path).split(".")[-1])

    plt.show()


def orthomosaic2matplotlib(
    flight_path: list[NDArray[np.uint16]],
    env: DroneGridEnv,
    ax: Axes,
    tile_type: Literal["OSM", "ArcGis"] = "ArcGis",
    linewidth: float = 0.5,
    fontsize: int = 10,
    scalebar_length: float | None = None,
    scalebar_args: dict[str, Any] = {},
    name: str | None = None,
    start_annotation_offset: tuple[int, int] | None = None,
    start_annotation_alignment: str = "left",
    end_annotation_offset: tuple[int, int] | None = None,
    end_annotation_alignment: str = "left",
    custom_extent: Extent | None = None,
    expand_lon_lat: tuple[float, float] = (0.0005, 0.0005),
) -> None:
    gt_objects = np.array(np.column_stack(env.world.object_map.nonzero()), dtype=np.int16)
    discovered_objects = env.drone.found_object_positions
    true_positives = intersect2d(gt_objects, discovered_objects)
    false_negatives = setdiff2d(gt_objects, discovered_objects)
    false_positives = setdiff2d(discovered_objects, gt_objects)

    def _local_to_gps_and_plt(local_coordinates: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert isinstance(env.world, OrthomosaicWorld)

        utm_coordinates = env.world.convert_local_to_utm(local_coordinates)
        gps_coordinates = np.array(
            np.column_stack(transform(env.world.config.crs, "EPSG:4326", utm_coordinates[:, 0], utm_coordinates[:, 1])), dtype=np.float64
        )
        plt_coordinates = np.array([project(lon, lat) for lon, lat in zip(gps_coordinates[:, 0], gps_coordinates[:, 1])], dtype=np.float64)
        return gps_coordinates, plt_coordinates

    flight_path_gps, flight_path_proj = _local_to_gps_and_plt(np.stack(flight_path).astype(np.int16))
    _, true_positives_proj = _local_to_gps_and_plt(true_positives)
    _, false_negatives_proj = _local_to_gps_and_plt(false_negatives)
    _, false_positives_proj = _local_to_gps_and_plt(false_positives)

    if custom_extent is None:
        extent = Extent.from_lonlat(
            flight_path_gps[:, 0].min() - expand_lon_lat[0],
            flight_path_gps[:, 0].max() + expand_lon_lat[0],
            flight_path_gps[:, 1].min() - expand_lon_lat[1],
            flight_path_gps[:, 1].max() + expand_lon_lat[1],
        )
    else:
        extent = custom_extent

    tiles = get_tiles(tile_type)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plotter = Plotter(extent, tiles, height=600)
    plotter.plot(ax, tiles, alpha=0.8)

    icon_path = get_file("marker.png")
    for i in range(false_negatives_proj.shape[0]):
        imagebox = _load_icon(icon_path, color=(200, 88, 108))
        imagebox.image.axes = ax

        ab = AnnotationBbox(
            imagebox,
            (false_negatives_proj[i, 0], false_negatives_proj[i, 1]),
            xybox=(0.0, 4.5),
            xycoords="data",
            boxcoords="offset points",
            bboxprops={"edgecolor": "#FF000000", "facecolor": "#FF000000"},
            zorder=2,
        )
        ax.add_artist(ab)

    for i in range(false_positives_proj.shape[0]):
        imagebox = _load_icon(icon_path, color=(240, 144, 60))
        imagebox.image.axes = ax

        ab = AnnotationBbox(
            imagebox,
            (false_positives_proj[i, 0], false_positives_proj[i, 1]),
            xybox=(0.0, 4.5),
            xycoords="data",
            boxcoords="offset points",
            bboxprops={"edgecolor": "#FF000000", "facecolor": "#FF000000"},
            zorder=2,
        )
        ax.add_artist(ab)

    for i in range(true_positives_proj.shape[0]):
        imagebox = _load_icon(icon_path, color=(102, 153, 204))
        imagebox.image.axes = ax

        ab = AnnotationBbox(
            imagebox,
            (true_positives_proj[i, 0], true_positives_proj[i, 1]),
            xybox=(0.0, 4.5),
            xycoords="data",
            boxcoords="offset points",
            bboxprops={"edgecolor": "#FF000000", "facecolor": "#FF000000"},
            zorder=2,
        )
        ax.add_artist(ab)

    ax.plot(flight_path_proj[:, 0], flight_path_proj[:, 1], color="blue", linewidth=linewidth)

    if start_annotation_offset is not None:
        ax.annotate(
            "Start",
            xy=tuple(flight_path_proj[0, :2]),
            xytext=start_annotation_offset,
            textcoords="offset points",
            horizontalalignment=start_annotation_alignment,
            arrowprops=dict(arrowstyle="->", linewidth=linewidth * 0.5),
            fontsize=fontsize,
        )

    if end_annotation_offset is not None:
        ax.annotate(
            "End",
            xy=tuple(flight_path_proj[-1, :2]),
            xytext=end_annotation_offset,
            textcoords="offset points",
            horizontalalignment=end_annotation_alignment,
            arrowprops=dict(arrowstyle="->", linewidth=linewidth * 0.5),
            fontsize=fontsize,
        )

    if scalebar_length is not None:
        _add_scalebar(scalebar_length, extent, ax, utm_crs=env.world.config.crs, linewidth=linewidth, fontsize=fontsize, **scalebar_args)

    if name is not None:
        ax.set_title(name)

    _remove_all_axes(ax)


def sim2matplotlib(env: DroneGridEnv, ax: Axes, plot_drone: bool = True, custom_rendering: Callable[[Surface], None] | None = None) -> None:
    env.rendering.config.world_image_padding = 0
    env.render()

    surface = Surface((1024, 1024))
    env.rendering.world_to_surface(surface)

    if plot_drone:
        env.rendering.classified_pixels_to_surface(surface)
        env.rendering.flight_path_to_surface(surface, color=(255, 0, 0))

        env.rendering.drone_to_surface(surface)
        env.rendering.fov_to_surface(surface, color=(255, 0, 0))

    if custom_rendering:
        custom_rendering(surface)

    ax.axis("off")
    ax.imshow(array3d(surface).transpose(1, 0, 2)[..., ::-1])


def plot_tiles_and_fov(ax: Axes, tile_scheme: dict[Path, Polygon], fov: Polygon, polygon_crs: str = "epsg:32631") -> None:
    tile_scheme_gps = tile_scheme.copy()
    for k, v in tile_scheme_gps.items():
        tile_scheme_gps[k] = transform_polygon(v, from_crs=polygon_crs, to_crs="EPSG:4326")

    fov_gps = transform_polygon(fov, from_crs="epsg:32631", to_crs="EPSG:4326")

    tiles = get_tiles("ArcGis")

    all_polygons = [*tile_scheme_gps.values(), fov_gps]
    min_lon = min(poly.bounds[0] for poly in all_polygons)
    min_lat = min(poly.bounds[1] for poly in all_polygons)
    max_lon = max(poly.bounds[2] for poly in all_polygons)
    max_lat = max(poly.bounds[3] for poly in all_polygons)

    extent = Extent.from_lonlat(min_lon, max_lon, min_lat, max_lat)

    plotter = Plotter(extent, tiles, zoom=20)
    plotter.plot(ax)

    def plot_polygon(polygon: Polygon, color: str = "blue") -> None:
        projected_coordinates = np.array([project(*coordinate) for coordinate in polygon.exterior.coords], dtype=np.float64)
        ax.plot(projected_coordinates[:, 0], projected_coordinates[:, 1], color=color)

    plot_polygon(fov_gps)

    for k, v in tile_scheme_gps.items():
        plot_polygon(v, color="red")
        if match := search(r"(\d+-\d+)", k.stem):
            text_coordinates = project(v.centroid.xy[0][0], v.centroid.xy[1][0])
            plt.text(text_coordinates[0], text_coordinates[1], match.group(0), ha="center", va="center")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def _get_frequent_color(img: NDArray[np.uint8], frequency_index: int = 0) -> NDArray[np.uint8]:
    reshaped_img = img.reshape(-1, img.shape[-1])
    unique_colors, counts = np.unique(reshaped_img, axis=0, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    return unique_colors[sorted_indices[frequency_index]]  # type: ignore[no-any-return]


def _load_icon(icon_file: Path, color: tuple[int, ...] | NDArray[np.float32] | None = None, zoom: float = 0.02) -> OffsetImage:
    arr_img = plt.imread(icon_file)

    if color is not None:
        mask = np.all(arr_img == _get_frequent_color(arr_img, frequency_index=1), axis=-1)
        color = np.array([*color, 255], dtype=np.float32) if len(color) == 3 else np.array(color, dtype=np.float32)
        arr_img[mask] = np.array(color, dtype=np.float32) / 255

    return OffsetImage(arr_img, zoom=zoom)


def _add_scalebar(
    scalebar_length: float,
    extent: Extent,
    ax: Axes,
    lw: float = 0.75,
    fontsize: float = 6.0,
    utm_crs: str = "epsg:32631",
    **kwargs: Any,
) -> None:
    def _utm_to_plt(utm_coordinate: NDArray[np.float64]) -> tuple[float, float]:
        lons, lats = transform(utm_crs, "EPSG:4326", [utm_coordinate[0]], [utm_coordinate[1]])
        return project(lons[0], lats[0])

    lon, lat = to_lonlat(extent.xmax, extent.ymax)
    bottom_left_utm = np.array(transform("EPSG:4326", utm_crs, [lon], [lat]), dtype=np.float64).flatten()
    left_utm = bottom_left_utm - (1.10 * scalebar_length, -0.10 * scalebar_length)
    right_utm = bottom_left_utm - (0.10 * scalebar_length, -0.10 * scalebar_length)
    text_utm = left_utm.copy() + (0.5 * scalebar_length, 1.5)

    ax.annotate(
        "",
        xy=(_utm_to_plt(right_utm)[0], _utm_to_plt(left_utm)[1]),  # Rounding error (?)
        xytext=_utm_to_plt(left_utm),
        arrowprops=dict(arrowstyle="<->", lw=lw, color="black", **kwargs),
    )
    ax.text(*_utm_to_plt(text_utm), f"{scalebar_length:.0f}m", ha="center", va="bottom", fontsize=fontsize)


def _remove_all_axes(ax: Axes) -> None:
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
