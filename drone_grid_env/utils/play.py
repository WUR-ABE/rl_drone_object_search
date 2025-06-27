from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pygame
from tabulate import tabulate
from tap import Tap

from gymnasium import envs, make
from gymnasium.utils.play import MissingKeysToAction, PlayableGame, display_arr

import drone_grid_env as _  # Register environment
from drone_grid_env import logger
from drone_grid_env.envs.utils import parse_input

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from numpy.typing import NDArray

    from gymnasium import Env
    from gymnasium.core import ActType, ObsType


def play(
    env: Env[ObsType, int],
    transpose: bool = True,
    fps: int | None = None,
    zoom: float | None = None,
    callback: Callable[[ObsType | None, ObsType, ActType | None, float | None, bool, bool, dict[str, Any]], None] | None = None,
    keys_to_action: dict[tuple[str | int, str], int] | None = None,
    seed: int | None = None,
) -> None:
    """
    Overrides 'gymnasium.utils.play.play' to not do an action when no input key is pressed and adds
    the possiblity to reset an environment.
    """
    env.reset(seed=seed)

    if keys_to_action is None:
        if hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert env.spec is not None
            raise MissingKeysToAction(f"{env.spec.id} does not have explicit key to action mapping, please specify one manually")
    assert keys_to_action is not None

    key_code_to_action = {}
    for key_combination, action in keys_to_action.items():
        key_code = tuple(sorted(ord(key) if isinstance(key, str) else key for key in key_combination))
        key_code_to_action[key_code] = action

    game = PlayableGame(env, keys_to_action=key_code_to_action, zoom=zoom)

    if fps is None:
        fps = cast(int, env.metadata.get("render_fps", 30))

    info: dict[str, Any] | None = None
    i = 0
    done = True
    obs = None
    new_keypress_available = False
    clock = pygame.time.Clock()

    while game.running:
        if done:
            if info is not None:
                print(tabulate([(k, v) for k, v in info.items()]))

            done = False
            obs, info = env.reset(seed=seed if i == 0 else None)

            i += 1
        else:
            action = key_code_to_action.get(tuple(sorted(game.pressed_keys)), None)
            prev_obs = obs

            if action is not None and new_keypress_available:
                obs, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                new_keypress_available = False

                if callback is not None:
                    callback(prev_obs, obs, action, rew, terminated, truncated, info)  # type: ignore[arg-type]

        if obs is not None:
            rendered: NDArray[np.uint8] | list[NDArray[np.uint8]] | None = env.render()
            if isinstance(rendered, list):
                rendered = rendered[-1]
            assert rendered is not None and isinstance(rendered, np.ndarray)
            display_arr(game.screen, rendered, transpose=transpose, video_size=game.video_size)

        # process pygame events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == ord("r"):
                done = False

                env.reset()

                i += 1

            if event.type == pygame.KEYUP:
                new_keypress_available = True

            game.process_event(event)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


class ArgumentParser(Tap):
    env: str = "DroneGridEnv-v0"  # Environment to use
    seed: int | None = None  # Seed to use
    verbose: bool = False  # Verbose logging
    env_args: dict[str, str | int | float | bool] = {}  # Environment args to pass through the environment

    def configure(self) -> None:
        env_ids = [env_spec.id for env_spec in envs.registry.values()]  # type: ignore[attr-defined]
        env_ids.sort()

        self.add_argument("--env", type=str, choices=env_ids)
        self.add_argument("--env_args", type=str, nargs="*", metavar="NAME=VAR")

    def process_args(self) -> None:
        self.env_args = {k: v if v.startswith("/") else parse_input(v) for k, v in (arg.split("=") for arg in self.env_args)}


if __name__ == "__main__":
    args = ArgumentParser().parse_args()

    if args.verbose:
        logger.set_level(logger.INFO)

    env = make(args.env, **args.env_args, render_mode="rgb_array")  # type: ignore[arg-type]
    play(env, zoom=1, fps=60, seed=args.seed)
