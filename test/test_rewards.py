from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from drone_grid_env.envs.action import ActionSpace

if TYPE_CHECKING:
    from drone_grid_env import DroneGridEnv


@pytest.mark.parametrize(
    "action,expected_reward",
    [
        (ActionSpace.FLY_EAST, 6.05),
        (ActionSpace.FLY_WEST, -1.0),
        (ActionSpace.FLY_SOUTH, 6.05),
        (ActionSpace.FLY_NORTH, -1.0),
        (ActionSpace.LAND, -1.0),
        # (ActionSpace.FLY_NORTHEAST, -1.0 - 0.1),
        # (ActionSpace.FLY_NORTHWEST, -1.0 - 0.1),
        # (ActionSpace.FLY_SOUTHEAST, 0.250 * 7 + 6 * 0.250 - 0.1),
        # (ActionSpace.FLY_SOUTHWEST, -1.0 - 0.1),
    ],
)
def test_rewards_env_1(drone_grid_env_1: DroneGridEnv, action: int, expected_reward: float) -> None:
    _, reward, _, _, _ = drone_grid_env_1.step(action)
    assert reward == expected_reward
