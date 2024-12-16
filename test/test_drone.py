from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from drone_grid_env import DroneGridEnv
from drone_grid_env.envs.action import ActionSpace

if TYPE_CHECKING:
    from drone_grid_env.envs.drone import Drone


def test_reset(drone_1: Drone) -> None:
    drone_1.fly(ActionSpace.FLY_SOUTH)
    drone_1.reset()

    np.testing.assert_array_equal(drone_1.position, [0, 0])
    np.testing.assert_array_equal(drone_1.flight_path.current_position, [0, 0])
    assert drone_1.battery_level == pytest.approx(150.0)


def test_land_outside_area(drone_2: Drone) -> None:
    drone_2.fly(ActionSpace.FLY_EAST)
    drone_2.fly(ActionSpace.FLY_EAST)
    drone_2.land()


def test_start_landing_zone_invalid() -> None:
    env_config = Path(__file__).parent / "test_config_invalid_start.yaml2"

    with pytest.raises(RuntimeError) as exc_info:
        DroneGridEnv(config_file=env_config)

    assert "'start_landing_zones' key is not defined" in str(exc_info.value)


def test_no_fly_action(drone_1: Drone) -> None:
    drone_1.fly(ActionSpace.LAND)

    # Nothing should happen
    np.testing.assert_array_equal(drone_1.position, [0, 0])
    np.testing.assert_array_equal(drone_1.flight_path.current_position, [0, 0])
    assert drone_1.battery_level == pytest.approx(150.0)


@pytest.mark.parametrize(
    "actions,expected_position,expected_path_length",
    [
        ([ActionSpace.FLY_NORTH, ActionSpace.FLY_EAST], (0, 1), 1.0),
        # ([ActionSpace.FLY_SOUTH, ActionSpace.FLY_NORTHEAST], (0, 1)),
        ([ActionSpace.FLY_WEST, ActionSpace.FLY_SOUTH], (1, 0), 1.0),
        # ([ActionSpace.FLY_NORTHWEST, ActionSpace.FLY_NORTHEAST], (0, 0)),
        # ([ActionSpace.FLY_SOUTHWEST, ActionSpace.FLY_WEST], (0, 0)),
        # ([ActionSpace.FLY_SOUTHEAST, ActionSpace.FLY_WEST], (1, 0)),
        ([ActionSpace.FLY_SOUTH, ActionSpace.FLY_NORTH], (0, 0), 2.0),
        ([ActionSpace.FLY_SOUTH, ActionSpace.FLY_SOUTH], (2, 0), 2.0),
        # ([ActionSpace.FLY_SOUTHEAST, ActionSpace.FLY_NORTHWEST], (0, 0)),
        # ([ActionSpace.FLY_SOUTHEAST, ActionSpace.FLY_SOUTHWEST], (2, 0)),
    ],
)
def test_fly(
    drone_1: Drone,
    actions: list[ActionSpace],
    expected_position: tuple[int, int],
    expected_path_length: float,
) -> None:
    for action in actions:
        drone_1.fly(action)

    np.testing.assert_array_equal(drone_1.position, expected_position)
    assert drone_1.battery_level == pytest.approx(150 - 0.5 * len(actions))
    assert drone_1.path_length == expected_path_length
