from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import filterwarnings

import numpy as np
from codecarbon import OfflineEmissionsTracker
from cv2 import imshow, waitKey
from tap import Tap
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.rich import tqdm

from gymnasium import envs, make
from gymnasium.wrappers.common import TimeLimit
from gymnasium.wrappers.rendering import RecordVideo

from drone_grid_env import logger
from drone_grid_env.envs.drone_grid_env import DroneGridEnv
from drone_grid_env.envs.io.csv import write_csv
from drone_grid_env.envs.io.kml import write_kml_flightpath
from drone_grid_env.envs.utils import parse_input
from sb3_custom.custom_dqn_algorithm import CustomDQN

from sb3_custom.fields2cover_algorithm import Fields2CoverAlgorithm
from sb3_custom.policies.local_global_softmax_scaling_policy import LocalGlobalSoftmaxScalingPolicy

if TYPE_CHECKING:
    from pathlib import Path

    from stable_baselines3.common.base_class import BaseAlgorithm


os.environ["SDL_VIDEODRIVER"] = "dummy"

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "rl": CustomDQN,
    "fields2cover": Fields2CoverAlgorithm,
}


filterwarnings("ignore", category=UserWarning)  # Ignore Gymnasium UserWarning


def evaluate(
    algorithm: BaseAlgorithm,
    env: DroneGridEnv,
    eval_folder: Path,
    name_prefix: str,
    n_episodes: int,
    max_length: int,
    seed: int | None,
    render: bool = True,
    video: bool = False,
) -> None:
    env = TimeLimit(env, max_length)  # type: ignore[assignment]
    env.reset(seed=seed)

    if video:
        env = RecordVideo(env, str(eval_folder), name_prefix=name_prefix, episode_trigger=lambda x: x == n_episodes - 1)  # type: ignore[assignment]

    if render:
        imshow("render", env.render()[..., ::-1])  # type: ignore[call-overload, index]
        waitKey(50)

    _num_actions = env.action_space.n  # type:ignore[attr-defined]

    flight_paths = []
    evaluation_results = []

    for i in tqdm(range(n_episodes)):
        with logging_redirect_tqdm():
            terminated = False
            truncated = False
            steps = 0

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
                obs_unsqueezed = {k: np.expand_dims(v, axis=0) for k, v in obs.items()} if isinstance(obs, dict) else np.expand_dims(obs, axis=0)  # type: ignore[call-overload]
                prediction = algorithm.predict(obs_unsqueezed)

                actions = prediction[0]
                if len(prediction) == 3:
                    env.unwrapped.set_action_values(prediction[1])

                obs, _, terminated, truncated, info = env.step(actions[0])

                steps += 1

                if render:
                    imshow("render", env.render()[..., ::-1])  # type: ignore[call-overload, index]
                    waitKey(50)

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
            flight_paths.append(deepcopy(env.unwrapped.drone.flight_path.to_array()))  # type: ignore[attr-defined]

    env.close()

    extended_kml_data = {"seed": seed, "algorithm": type(algorithm).__name__, "env_config_file": env.unwrapped.config_file_path}  # type: ignore[attr-defined]

    write_csv(eval_folder / (name_prefix + ".csv"), evaluation_results)
    write_kml_flightpath(eval_folder / (name_prefix + ".kml"), flight_paths, extended_data=extended_kml_data)


class ArgumentParser(Tap):
    algorithm: str  # Algorithms to evaluate
    weight_file: Path | None = None  # Path to RL weight file
    env: str = "DroneGridEnv-v0"  # Environment to evaluate
    env_args: dict[str, str | int | float | bool] = {}  # Environment args to pass through the environment
    output_folder: Path = Path("evaluations")  # Path to the output folder
    seed: int | None = None  # Seed to evaluate
    n_episodes: int = 150  # Number of episodes to evaluate on
    max_length: int = 400  # Maximum number of steps for evaluation
    prefix: str = ""  # Name prefix for output
    render: bool = False  # Render output
    video: bool = False  # Save video from last episode
    verbose: bool = False  # Verbose logging

    def configure(self) -> None:
        env_ids = [env_spec.id for env_spec in envs.registry.values()]  # type: ignore[attr-defined]
        env_ids.sort()

        self.add_argument("algorithm", type=str, choices=list(ALGORITHMS.keys()))
        self.add_argument("--env", type=str, choices=env_ids)
        self.add_argument("--env_args", type=str, nargs="*", metavar="NAME=VAR")

    def process_args(self) -> None:
        self.env_args = {k: v if v.startswith("/") else parse_input(v) for k, v in (arg.split("=") for arg in self.env_args)}


if __name__ == "__main__":
    args = ArgumentParser().parse_args()

    args.env_args["render_mode"] = "rgb_array"

    print(f"Running evaluation on {args.env} ({args.env_args})")

    env = make(args.env, **args.env_args)  # type: ignore[arg-type]

    args.output_folder.mkdir(exist_ok=True, parents=True)

    prefix = args.prefix
    if prefix != "":
        prefix += "-"

    custom_objects = {
        "policy_class": LocalGlobalSoftmaxScalingPolicy,
    }

    # Set logging
    logger.set_level(logger.DEBUG if args.verbose else logger.WARN)

    method = ALGORITHMS[args.algorithm].load(args.weight_file, env=env, custom_objects=custom_objects)  # type: ignore[arg-type]
    name_prefix = f"{prefix}{args.algorithm}"

    tracker = None
    if args.algorithm == "rl":
        tracker = OfflineEmissionsTracker(
            country_iso_code="NLD",
            project_name=f"inference_{name_prefix}",
            experiment_name=f"inference_{name_prefix}",
            log_level="warning",
        )
        tracker.start()

    assert isinstance(env.unwrapped, DroneGridEnv), f"Can only evaluate DroneGridEnv for now, but type is {type(env)}..."

    evaluate(
        method,
        env,
        args.output_folder,
        name_prefix,
        args.n_episodes,
        args.max_length,
        args.seed,
        render=args.render,
        video=args.video,
    )

    print(f"Saved evaluation results in {args.output_folder} with prefix {name_prefix}")

    if tracker is not None:
        tracker.stop()
