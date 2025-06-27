from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from yaml import YAMLError, safe_load

from gymnasium import Env
import numpy as np

from drone_grid_env import logger
from drone_grid_env.envs.action import ActionSpace
from drone_grid_env.envs.drone import Drone
from drone_grid_env.envs.rendering import Rendering
from drone_grid_env.envs.reward import RewardSpace
from drone_grid_env.envs.state import StateSpace
from drone_grid_env.envs.stopping_criteria import StoppingCriteria
from drone_grid_env.envs.utils import count_gt_classified_objects, get_file
from drone_grid_env.envs.world.orthomosaic_world import OrthomosaicWorld
from drone_grid_env.envs.world.sim_world import SimWorld

if TYPE_CHECKING:
    from typing import Any, Literal

    from drone_grid_env.envs.world import World

from numpy.typing import NDArray

ObsType = dict[str, NDArray[np.uint8]]
ActType = int

VIEWPORT_W = 600
VIEWPORT_H = 400

WORLDS: dict[str, type[World[Any]]] = {
    "SimWorld": SimWorld,
    "OrthomosaicWorld": OrthomosaicWorld,
}


class DroneGridEnv(Env[ObsType, ActType]):
    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array", "rgb_array_headless", "rgb_array_list", "rgb_array_list_headless"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config_file: Path | str = "random_world.yaml",
        render_mode: Literal["human", "rgb_array", "rgb_array_headless", "rgb_array_list", "rgb_array_list_headless"] | None = None,
        verbose: bool = False,
    ) -> None:
        assert render_mode in self.metadata["render_modes"] or render_mode is None, "Render mode is not valid!"

        super().__init__()

        logger.set_level(logger.DEBUG if verbose else logger.WARN)

        self.config_file = config_file
        self._check_config_file(self.config_file_path)

        self.world = WORLDS[self.config["world"]["type"]].from_config_file(self.config_file_path, rng=self.np_random)
        self.drone = Drone.from_config_file(self.config_file_path, self.world, rng=self.np_random)
        self.state_space = StateSpace.from_config_file(self.config_file_path, self.world, self.drone)
        self.reward_space = RewardSpace.from_config_file(self.config_file_path)
        self.stopping_criteria = StoppingCriteria.from_config_file(self.config_file_path)

        # Define state and action space
        self.observation_space = self.state_space.get_space()
        self.action_space = ActionSpace.get_space(self.config_file_path)

        self.action_values = None

        self._steps = 0
        self._total_reward = 0.0
        self._last_state: ObsType | None = None
        self._last_action: int | ActType | None = None
        self._last_info: dict[str, Any] | None = None

        self.render_mode = render_mode
        self.rendering = Rendering(self.world, self.drone)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self._steps = 0
        self._total_reward = 0.0
        self._last_action = None

        self.action_values = None

        self.world.reset(rng=self.np_random)
        self.drone.reset(rng=self.np_random)
        self.stopping_criteria.reset()
        self.rendering.reset()

        logger.info(f"Total of {self.world.number_of_objects} objects in world")

        self._last_state = self.state_space.get_state()
        self._last_info = self._get_info()

        return self._last_state, self._last_info

    def step(self, action: int) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action) or action is None, f"{action!r} ({type(action)}) invalid!"

        terminated = False
        truncated = False

        with self.reward_space.calculate_reward(self.world, self.drone, self.stopping_criteria, action):
            if ActionSpace.is_movement_action(action):
                self.drone.fly(action)

            if action == ActionSpace.LAND:
                self.drone.land()

            if self.drone.battery_level <= 0:
                terminated = True

            if self.stopping_criteria.can_terminate(self.world, self.drone, action):  # type: ignore[arg-type]
                terminated = True

        self._steps += 1
        self._total_reward += self.reward_space.get_reward()

        self._last_action = action
        self._last_state = self.state_space.get_state()
        self._last_info = self._get_info()

        return self._last_state, self.reward_space.get_reward(), terminated, truncated, self._last_info

    def render(self) -> NDArray[np.uint8] | list[NDArray[np.uint8]] | None:  # type:ignore[override]
        if self.render_mode is None:
            return None

        try:
            return self.rendering.render(self._last_state, self._last_action, self._last_info, self.render_mode)  # type: ignore[arg-type]
        except Exception as e:
            logger.warn(f"Something went wrong: {e}")

        return None

    def close(self) -> None:
        self.rendering.close()

    @cached_property
    def config_file_path(self) -> Path:
        return get_file(self.config_file)

    @cached_property
    def config(self) -> dict[str, Any]:
        with self.config_file_path.open("r") as cs:
            try:
                return safe_load(cs)

            except YAMLError as e:  # pragma: no cover
                logger.error(f"Could not load config file '{self.config_file}': {e}")
                raise

    def set_action_values(self, action_values: NDArray[np.float32]) -> None:
        self.action_values = np.ravel(action_values)

    def get_keys_to_action(self) -> dict[tuple[int, ...], int | None]:
        return ActionSpace.create_keys2action(self.config_file_path)

    def _get_info(self) -> dict[str, Any]:
        return {
            "rew": self._total_reward,
            "bat": self.drone.battery_level,
            "cw": count_gt_classified_objects(self.world, self.drone),
            "cwp": (
                count_gt_classified_objects(self.world, self.drone) / self.world.number_of_objects
                if self.world.number_of_objects != 0
                else 0
            ),
            "cov": np.count_nonzero(self.drone.coverage_map) / (self.world.size[0] * self.world.size[1]),
            "stp": self._steps,
            "pl": self.drone.flight_path.total_distance,
            "action_values": self.action_values,
        }

    @staticmethod
    def _check_config_file(config_file: Path) -> None:
        if not config_file.is_file():
            logger.error("Config file does not exist!")

        with config_file.open("r") as cs:
            try:
                data: dict[str, Any] = safe_load(cs)

                if data["stopping_criteria"]["method"] == "land" and not data["action_space"]["land_action"]:
                    logger.warn("Stopping criteria is set to landing, however, landing action is not enabled!")

                if data["stopping_criteria"]["only_land_in_zone"] and not data["state_space"]["add_start_landing_zone"]:
                    logger.warn("Only allowing landing in start-landing zone, however, start-landing zone is not added to the state-space")

            except (YAMLError, KeyError) as e:  # pragma: no cover
                logger.error(f"Could not load config file '{config_file.name}': {e}")
                raise
