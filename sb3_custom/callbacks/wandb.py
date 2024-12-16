from __future__ import annotations

from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING

import wandb
from wandb.integration.sb3 import WandbCallback

from gymnasium import make

if TYPE_CHECKING:
    from typing import Literal


class CustomWandbCallback(WandbCallback):
    def __init__(
        self,
        verbose: int = 0,
        model_save_path: str | None = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Literal["gradients", "parameters", "all"] | None = "all",
        log_graph: bool = False,
    ) -> None:
        super().__init__(verbose, model_save_path, model_save_freq, gradient_save_freq, log)

        self.log_graph = log_graph

    def _init_callback(self) -> None:
        """
        Adaptation of default WandB callback to include:
        - uploading of environment config file
        - uploading of rlzoo config file
        - adapting run name based on environment variable
        - adding tags from environment variable (comma seperated)
        """
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__

        for key in self.model.__dict__:
            if key in wandb.config:
                continue

            if isinstance(self.model.__dict__[key], (float, int, str)):
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])

        if self.gradient_save_freq > 0 and self.model is not None:
            wandb.watch(
                self.model.policy,
                log_freq=self.gradient_save_freq,
                log=self.log,
                log_graph=self.log_graph,
            )

        wandb.config.setdefaults(d)

        # Hack to upload DroneGridEnv stuff
        self.custom_env_uploads()

        # Upload training yaml file
        if "conf_file" in wandb.config:
            train_config_file: Path = Path.cwd() / wandb.config["conf_file"]
            self.upload_file(train_config_file, "train_config" + train_config_file.suffix)

        assert wandb.run is not None

        # Change run name from env
        if getenv("WANDB_NAME") is not None:
            wandb.run.name = getenv("WANDB_NAME")

        # Add algorithm name as tag
        algo_name = wandb.config.get("algo", None)
        if algo_name is not None:
            wandb.run.tags = wandb.run.tags + tuple((algo_name,))

        # Add tags from env
        wandb_tags = getenv("WANDB_TAGS")
        if wandb_tags is not None and getenv("WANDB_TAGS") != "":
            wandb.run.tags = wandb.run.tags + tuple(wandb_tags.split(","))

    @staticmethod
    def upload_file(file: Path, filename: str, tmp_path: Path = Path.cwd() / "wandb/tmp") -> None:
        # if not tmp_path.is_dir():
        #     tmp_path.mkdir(parents=True)

        # if (tmp_path / filename).is_file():
        #     (tmp_path / filename).unlink()

        # tmp_file = Path(copy2(file, tmp_path))
        # tmp_file.rename(tmp_path / filename)

        wandb.save(str(file), str(file.parent))

    @staticmethod
    def custom_env_uploads() -> None:
        # Hack to upload config file for DroneGridEnv
        if all(k in wandb.config for k in ("env", "env_kwargs")) and "DroneGridEnv" in wandb.config["env"]:
            _env = make(wandb.config["env"], **wandb.config["env_kwargs"])
            if hasattr(_env, "config_file_path"):
                CustomWandbCallback.upload_file(_env.config_file_path, "env_config_file.yaml")

            _env.close()  # type: ignore[no-untyped-call]
