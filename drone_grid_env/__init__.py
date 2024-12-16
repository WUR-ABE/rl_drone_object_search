from __future__ import annotations

import os

from gymnasium import register

# Disable Pygame welcome message before we import Pygame somewhere
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from drone_grid_env.envs.drone_grid_env import DroneGridEnv

__version__ = "1.0.0"
__all__ = ["DroneGridEnv"]


register(
    id="DroneGridEnv-v0",
    entry_point="drone_grid_env.envs.drone_grid_env:DroneGridEnv",
)
