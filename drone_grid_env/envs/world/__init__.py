from __future__ import annotations

from typing import TYPE_CHECKING

from drone_grid_env.envs.world.sim_world import SimWorld

if TYPE_CHECKING:
    from drone_grid_env.envs.world.base import BaseWorld


__all__ = ["SimWorld", "get_world_by_name"]


def get_world_by_name(name: str) -> type[BaseWorld]:
    return {"SimWorld": SimWorld}[name]
