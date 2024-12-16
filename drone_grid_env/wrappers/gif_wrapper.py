from __future__ import annotations

from pathlib import Path
from re import search
from typing import TYPE_CHECKING, cast
from warnings import warn

from gymnasium import Wrapper
from gymnasium.core import ActType, ObsType

try:
    from moviepy.editor import ImageSequenceClip

    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

if TYPE_CHECKING:
    from typing import Any, SupportsFloat

    import numpy as np
    from numpy.typing import NDArray

    from gymnasium import Env


class GIFWrapper(Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType], save_path: Path = Path("episode.gif")) -> None:
        super().__init__(env)

        self.save_path = save_path
        self.image_buffer: list[NDArray[np.uint8]] = []
        self._clip = None

        if self.env.render_mode not in ("rgb_array", "rgb_array_list", "rgb_array_headless", "rgb_array_list_headless"):
            warn("Cannot create gif because render mode is not 'rgb_array' or 'rgb_array_list'!")

        if not HAS_MOVIEPY:
            warn("Cannot create gif because moviepy is not installed. Install moviepy by `pip install moviepy`.")

        self.frames_per_sec = cast(int, env.metadata.get("video.frames_per_second", 30))

    @property
    def clip(self) -> ImageSequenceClip:
        if not HAS_MOVIEPY:
            raise ImportError("To return the clip, moviepy is needed.")

        if self._clip is None:
            self.create_clip()
        return self._clip

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        state = super().reset(seed=seed, options=options)
        self.image_buffer.clear()
        return state

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        step = self.env.step(action)

        if HAS_MOVIEPY:
            # Collect frames for this step
            if self.env.render_mode == "rgb_array":
                frame: NDArray[np.uint8] | list[NDArray[np.uint8]] | None = self.env.render()

                if isinstance(frame, np.ndarray):
                    self.image_buffer.append(frame)

            # When terminated or truncated
            if step[2] or step[3]:
                # Collect all frames for the whole episode
                if self.env.render_mode == "rgb_array_list":
                    frames: NDArray[np.uint8] | list[NDArray[np.uint8]] | None = self.env.render()

                    if isinstance(frames, list):
                        self.image_buffer.extend(frames)

                self.create_clip()

                if self._clip is not None:
                    self._clip.write_gif(self.create_unique_path(self.save_path), fps=self.frames_per_sec)

        return step

    def create_clip(self) -> None:
        self._clip = ImageSequenceClip(self.image_buffer, fps=self.frames_per_sec)

    @staticmethod
    def create_unique_path(save_path: Path) -> Path:
        if not save_path.is_file():
            return save_path

        match = search(r"(.+)_([0-9]+)", save_path.stem)
        if match is not None:
            return GIFWrapper.create_unique_path(save_path.parent / f"{match.group(1)}_{int(match.group(2)) + 1}.gif")

        return GIFWrapper.create_unique_path(save_path.parent / f"{save_path.stem}_1.gif")
