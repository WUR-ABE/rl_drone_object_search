from __future__ import annotations

from pathlib import Path
from re import search
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pygame import Surface
from pygame.surfarray import array3d

from drone_grid_env.envs.drone_grid_env import DroneGridEnv
from drone_grid_env.envs.utils import FlightPath
from drone_grid_env.envs.world.sim_world import SimWorld

if TYPE_CHECKING:
    from typing import Callable, Literal

    from numpy.typing import NDArray

matplotlib.rcParams["text.usetex"] = True

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
    ax: plt.Axes,
    labels: list[str] | None = None,
    groups: list[list[str]] | None = None,
    unique_color: list[str] | None = None,
    conf_interval: bool = False,
    custom_colors: list[str] | None = None,
    xlim: tuple[float, float] = (0, 300),
    ylim: tuple[float, float] = (0, 200),
    xlabel: str = "Flight path length",
    ylabel: str = "Number of found objects",
    legend_loc: str = "lower left",
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
            colors = {member: COLORS[len(group) + len(unique_color)][i + len(unique_color)] for group in groups for i, member in enumerate(group)}

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
    ax.legend(loc=legend_loc)


def create_bar_plot(
    data: list[NDArray[np.float32]],
    ax: plt.Axes,
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
        assert len(set(len(sublist) for sublist in groups)) == 1, "All groups should have equal length!"

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
    data: list[NDArray],
    ax: plt.Axes,
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
        assert len(set(len(sublist) for sublist in groups)) == 1, "All groups should have equal length!"

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

    if isinstance(env.world, SimWorld):
        env.drone._flight_path = FlightPath(flight_path)
        sim2matplotlib(env, ax, plot_drone=True)
    else:
        raise NotImplementedError

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, format=str(save_path).split(".")[-1])

    plt.show()


def sim2matplotlib(env: DroneGridEnv, ax: plt.Axes, plot_drone: bool = True, custom_rendering: Callable[[Surface], None] | None = None) -> None:
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