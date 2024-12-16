from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from drone_grid_env import DroneGridEnv
from drone_grid_env.envs.world import SimWorld

if TYPE_CHECKING:
    from drone_grid_env.envs.drone import Drone


@pytest.fixture
def drone_grid_env_1() -> DroneGridEnv:
    env_config = Path(__file__).parent / "test_config_1.yaml"

    env = DroneGridEnv(config_file=env_config)
    env.reset(seed=0)
    return env


@pytest.fixture
def drone_grid_env_2() -> DroneGridEnv:
    env_config = Path(__file__).parent / "test_config_2.yaml"

    env = DroneGridEnv(config_file=env_config)
    env.reset(seed=0)
    return env


@pytest.fixture
def drone_grid_env_3() -> DroneGridEnv:
    env_config = Path(__file__).parent / "test_config_3.yaml"

    env = DroneGridEnv(config_file=env_config)
    env.reset(seed=0)
    return env


@pytest.fixture
def drone_grid_env_4() -> DroneGridEnv:
    env_config = Path(__file__).parent / "test_config_4.yaml"

    env = DroneGridEnv(config_file=env_config)
    env.reset(seed=0)
    return env


@pytest.fixture
def drone_1(drone_grid_env_1: DroneGridEnv) -> Drone:
    return drone_grid_env_1.drone


@pytest.fixture
def drone_2(drone_grid_env_2: DroneGridEnv) -> Drone:
    return drone_grid_env_2.drone


@pytest.fixture
def world_1(drone_grid_env_1: DroneGridEnv) -> SimWorld:
    return cast(SimWorld, drone_grid_env_1.world)


@pytest.fixture
def world_2(drone_grid_env_2: DroneGridEnv) -> SimWorld:
    return cast(SimWorld, drone_grid_env_2.world)
