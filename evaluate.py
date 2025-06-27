from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from tap import Tap
from typing import TYPE_CHECKING
from warnings import filterwarnings

import cv2
import gymnasium as gym
import numpy as np

from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.rich import tqdm

from drone_grid_env import logger
from drone_grid_env.envs.drone_grid_env import DroneGridEnv
from drone_grid_env.envs.io.csv import write_csv
from drone_grid_env.envs.io.kml import write_kml_flightpath, write_kml_object_locations
from drone_grid_env.envs.utils import parse_input
from sb3_custom.custom_dqn_algorithm import CustomDQN
from sb3_custom.policies.local_global_softmax_scaling_policy import LocalGlobalSoftmaxScalingPolicy

if TYPE_CHECKING:
    from pathlib import Path

    from stable_baselines3.common.base_class import BaseAlgorithm


os.environ["SDL_VIDEODRIVER"] = "dummy"

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "rl": CustomDQN,
}

try:
    from sb3_custom.fields2cover_algorithm import Fields2CoverAlgorithm

    ALGORITHMS["fields2cover"] = Fields2CoverAlgorithm

except ImportError:
    pass

filterwarnings("ignore", category=UserWarning)  # Ignore Gymnasium UserWarning


def evaluate(
    algorithm: BaseAlgorithm,
    env: DroneGridEnv,
    eval_folder: Path,
    name_prefix: str,
    n_episodes: int,
    max_length: int | None,
    seed: int | None,
    save_object_map: bool = False,
    save_gt_map: bool = False,
    deterministic: bool = True,
    render: bool = True,
    video: bool = False,
) -> None:
    if max_length is not None:
        env = gym.wrappers.TimeLimit(env, max_length)  # type: ignore[assignment]

    env.reset(seed=seed)

    if video:
        env = gym.wrappers.RecordVideo(env, str(eval_folder), name_prefix=name_prefix, episode_trigger=lambda x: x == n_episodes - 1)  # type: ignore[assignment]

    if render:
        cv2.imshow("render", env.render()[..., ::-1])  # type: ignore[call-overload, index]
        cv2.waitKey(50)

    _num_actions = env.action_space.n  # type:ignore[attr-defined]

    flight_paths = []
    random_states = []
    evaluation_results = []

    for i in tqdm(range(n_episodes)):
        with logging_redirect_tqdm():
            terminated = False
            truncated = False
            steps = 0

            random_states.append(env.unwrapped.np_random.bit_generator.state)  # Save random state for each episode

            obs, info = env.reset(seed=seed if i == 0 else None)

            action_values = [0 for _ in range(_num_actions)]
            if info["action_values"] is not None:
                action_values = info["action_values"].tolist()

            evaluation_results.append(
                [
                    i,
                    steps,
                    info["pl"],
                    info["cw"],
                    info["cwp"],
                    info["cov"],
                    info["bat"],
                    info["rew"],
                    *action_values,
                ]
            )

            while not (terminated or truncated):
                obs_unsqueezed = (
                    {k: np.expand_dims(v, axis=0) for k, v in obs.items()} if isinstance(obs, dict) else np.expand_dims(obs, axis=0)
                )  # type: ignore[call-overload]
                prediction = algorithm.predict(obs_unsqueezed, deterministic=deterministic)

                actions = prediction[0]
                if len(prediction) == 3:
                    env.unwrapped.set_action_values(prediction[1])

                obs, _, terminated, truncated, info = env.step(actions[0])

                steps += 1

                if render:
                    cv2.imshow("render", env.render()[..., ::-1])  # type: ignore[call-overload, index]
                    cv2.waitKey(50)

                action_values = [0 for _ in range(_num_actions)]
                if info["action_values"] is not None:
                    action_values = info["action_values"].tolist()

                evaluation_results.append(
                    [
                        i,
                        steps,
                        info["pl"],
                        info["cw"],
                        info["cwp"],
                        info["cov"],
                        info["bat"],
                        info["rew"],
                        *action_values,
                    ]
                )

            # Copy to avoid changing values episode
            flight_paths.append(deepcopy(env.unwrapped.drone.flight_path.array))  # type: ignore[attr-defined]

            if save_object_map:
                save_folder = eval_folder / "detected_object_maps"
                save_folder.mkdir(exist_ok=True)
                discovered_objects = env.unwrapped.drone.found_object_positions.copy()  # type: ignore[attr-defined]
                write_kml_object_locations(save_folder / f"{name_prefix}_episode_{i}.kml", discovered_objects)

            if save_gt_map:
                save_folder = eval_folder / "gt_object_maps"
                save_folder.mkdir(exist_ok=True)
                gt_coordinates = np.column_stack(env.unwrapped.world.object_map.nonzero())
                np.savetxt(save_folder / f"{name_prefix}_episode_{i}.txt", gt_coordinates, delimiter=" ")

    env.close()

    extended_kml_data = {"seed": seed, "algorithm": type(algorithm).__name__, "env_config_file": env.unwrapped.config_file_path}  # type: ignore[attr-defined]

    write_csv(eval_folder / (name_prefix + ".csv"), evaluation_results)
    write_kml_flightpath(
        eval_folder / (name_prefix + ".kml"),
        flight_paths,
        extended_data_per_item=random_states,
        extended_data=extended_kml_data,
    )


class ArgumentParser(Tap):
    algorithm: str  # Algorithms to evaluate
    weight_file: Path | None = None  # Path to RL weight file
    env: str = "DroneGridEnv-v0"  # Environment to evaluate
    env_args: dict[str, str | int | float | bool] = {}  # Environment args to pass through the environment
    algorithm_args: dict[str, str | int | float | bool] = {}  # Environment args to pass through the algorithm
    output_folder: Path = Path("evaluations")  # Path to the output folder
    seed: int | None = None  # Seed to evaluate
    n_episodes: int = 150  # Number of episodes to evaluate on
    max_length: int | None = None  # Maximum number of steps for evaluation
    save_object_map: bool = False  # Save discovered objects map
    save_gt_map: bool = False  # Save gt object map
    prefix: str = ""  # Name prefix for output
    render: bool = False  # Render output
    video: bool = False  # Save video from last episode
    deterministic: bool = True  # Deterministic evaluation
    verbose: bool = False  # Verbose logging

    def configure(self) -> None:
        env_ids = [env_spec.id for env_spec in gym.envs.registry.values()]  # type: ignore[attr-defined]
        env_ids.sort()

        self.add_argument("algorithm", type=str, choices=list(ALGORITHMS.keys()))
        self.add_argument("--env", type=str, choices=env_ids)
        self.add_argument("--env_args", type=str, nargs="*", metavar="NAME=VAR")
        self.add_argument("--algorithm_args", type=str, nargs="*", metavar="NAME=VAR")

    def process_args(self) -> None:
        self.env_args = {k: v if v.startswith("/") else parse_input(v) for k, v in (arg.split("=") for arg in self.env_args)}
        self.algorithm_args = {k: v if v.startswith("/") else parse_input(v) for k, v in (arg.split("=") for arg in self.algorithm_args)}


if __name__ == "__main__":
    args = ArgumentParser().parse_args()

    args.env_args["render_mode"] = "rgb_array"

    print(f"Running evaluation on {args.env} ({args.env_args})")

    env = gym.make(args.env, **args.env_args)  # type: ignore[arg-type]

    args.output_folder.mkdir(exist_ok=True, parents=True)

    prefix = args.prefix
    if prefix != "":
        prefix += "-"

    custom_objects = {
        "policy_class": LocalGlobalSoftmaxScalingPolicy,
    }

    # Set logging
    logger.set_level(logger.DEBUG if args.verbose else logger.WARN)

    method = ALGORITHMS[args.algorithm].load(args.weight_file, env=env, custom_objects=custom_objects, **args.algorithm_args)  # type: ignore[arg-type]
    name_prefix = f"{prefix}{args.algorithm}"

    assert isinstance(env.unwrapped, DroneGridEnv), f"Can only evaluate DroneGridEnv ({type(env.unwrapped)}) for now..."

    evaluate(
        method,
        env,
        args.output_folder,
        name_prefix,
        args.n_episodes,
        args.max_length,
        args.seed,
        save_object_map=args.save_object_map,
        save_gt_map=args.save_gt_map,
        deterministic=args.deterministic,
        render=args.render,
        video=args.video,
    )

    print(f"Saved evaluation results in {args.output_folder} with prefix {name_prefix}")
