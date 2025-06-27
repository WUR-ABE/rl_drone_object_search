from __future__ import annotations

from copy import deepcopy
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import wandb
from moviepy.editor import ImageSequenceClip

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .exceptions import NoSuitableRendermodeException
from .utils import apply_wrappers, vec_env_to_gym_env

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from gymnasium import Env
    from gymnasium.core import ActType


class RecordVideoCallback(BaseCallback):
    def __init__(
        self,
        record_freq: int = 10000,
        video_path: str = "videos",
        deterministic: bool = True,
        n_video_episodes: int = 1,
        use_gif: bool = True,
        save_individual_episodes: bool = False,
        fallback_env: Env[dict[str, NDArray[np.uint8]] | NDArray[np.uint8], ActType] | str | None = None,
        fallback_env_kwargs: dict[str, Any] | None = None,
        fallback_env_wrapper: list[str] | None = None,
        seed: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)

        self._record_freq = record_freq
        self._video_folder = Path(video_path)
        self._deterministic = deterministic
        self._n_video_episodes = n_video_episodes
        self._use_gif = use_gif
        self._save_individual_episodes = save_individual_episodes
        self._seed = seed
        self._fallback_env = fallback_env
        self._fallback_env_kwargs = fallback_env_kwargs
        self._fallback_env_wrapper = fallback_env_wrapper

        self.video_folder: Path | None = None
        self.video_env: DummyVecEnv | None = None
        self.env_name = "Unknown"

        self._render_mode = "rgb_array"

    def _on_step(self) -> bool:
        if self.num_timesteps % self._record_freq == 0:
            self.record_video()

        return True

    def _on_training_start(self) -> None:
        # Also record at start to get a video that is completely random
        self.record_video()

    def record_video(self) -> None:
        assert self.training_env is not None

        frame_buffer: list[NDArray[np.uint8]] = []

        if self.video_env is None:
            _render_modes = self.training_env.get_attr("metadata")[0]["render_modes"]

            if "rgb_array_headless" in _render_modes:
                _render_mode = "rgb_array_headless"
            elif "rgb_array" in _render_modes:
                _render_mode = "rgb_array"
            else:
                raise NoSuitableRendermodeException(
                    f"Render mode 'rgb_array_headless' or 'rgb_array' not available for environment! Available modes are: {_render_modes}"
                )

            training_env = vec_env_to_gym_env(
                self.training_env.unwrapped,  # type: ignore[attr-defined]
                fallback_env=self._fallback_env,
                fallback_env_kwargs=self._fallback_env_kwargs,
                fallback_env_wrapper=self._fallback_env_wrapper,
                render_mode=_render_mode,
            )

            self.env_name = type(training_env.unwrapped).__name__

            self.video_env = DummyVecEnv([lambda: deepcopy(training_env)])
            self.video_env = apply_wrappers(self.training_env, self.video_env)  # type: ignore[type-var,assignment]

        if self.video_folder is None:
            self.video_folder = self._get_video_folder()

        if self.verbose > 0:
            print("Start recording episode...")

        if self._seed is not None:
            self.video_env.seed(self._seed)

        for i in range(self._n_video_episodes):
            frames = self._play_episode()

            if self._save_individual_episodes:
                self._save_video(frames, f"{self.env_name}_step_{self.num_timesteps}_eps_{i}")

            frame_buffer.extend(frames)

        if self.verbose > 0:
            print("Finished recording episode...")
            print(f"Start creating {'gif' if self._use_gif else 'mp4'} from {len(frame_buffer)} frames...")

        video_path = self._save_video(frame_buffer, f"{self.env_name}_step_{self.num_timesteps}")

        if self.verbose > 0:
            print(f"Finished creating {'gif' if self._use_gif else 'mp4'}...")
            print("Start uploading recoding episode to WandB...")

        # TODO: fix timesteps in wandb
        wandb.log({"render": wandb.Image(str(video_path))})

        # # Dump log so the evaluation results are printed with the correct timestep
        # self.logger.record(
        #     "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        # )
        # self.logger.dump(self.num_timesteps)

        if self.verbose > 0:
            print("Finished uploading recording...")

    def _get_video_folder(self) -> Path:
        assert self.video_env is not None

        if run_name := environ.get("WANDB_NAME", False):
            video_path_full = self._video_folder / str(run_name)
        else:
            video_path_full = self._video_folder / self.env_name

        # Add number if folder exists
        video_path_full = self._increase_path(video_path_full)
        video_path_full.mkdir(parents=True)

        return video_path_full

    def _increase_path(self, path: Path) -> Path:
        if path.is_dir():
            stem_parts = path.stem.split("_")
            maybe_number = stem_parts[-1]
            if maybe_number.isnumeric():
                stem_parts[-1] = str(int(maybe_number) + 1)
            else:
                stem_parts.append("1")
            return self._increase_path(path.parent / "_".join(stem_parts))
        return path

    def _play_episode(self) -> list[NDArray[np.uint8]]:
        assert self.video_env is not None

        frames = []

        terminated = False
        truncated = False

        obs = self.video_env.reset()

        while not (terminated or truncated):
            # VecEnvs reset automatically, render before taking an action
            frame: NDArray[np.uint8] | list[NDArray[np.uint8]] | None = self.video_env.envs[0].render()

            if isinstance(frame, np.ndarray):
                frames.append(frame)

            actions, action_values, _ = self.model.predict(obs, deterministic=self._deterministic)  # type: ignore[arg-type,misc]

            for i in range(self.video_env.num_envs):
                self.video_env.env_method("set_action_values", action_values[i, :], indices=i)

            obs, _, terminations, infos = self.video_env.step(actions)

            terminated = terminations[0]
            truncated = infos[0].get("TimeLimit.truncated", False)

        return frames

    def _save_video(self, frames: list[NDArray[np.uint8]], filename: str) -> Path:
        assert self.video_folder is not None
        assert self.video_env is not None

        frames_per_sec = self.video_env.metadata.get("render_fps", 30)
        clip = ImageSequenceClip(frames, fps=frames_per_sec)

        if self._use_gif:
            full_path = self.video_folder / (filename + ".gif")
            clip.write_gif(
                full_path,
                fps=frames_per_sec,
                verbose=False,
                logger=None,
            )

            print(f"Saved GIF to {full_path}")
        else:
            full_path = self._video_folder / (filename + ".mp4")
            clip.write_videofile(
                full_path,
                fps=frames_per_sec,
                verbose=False,
                logger=None,
            )

            print(f"Saved MP4 to {full_path}")

        clip.close()
        return full_path
