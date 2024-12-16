from __future__ import annotations

from importlib import resources
from pathlib import Path
from yaml import safe_load

import pytest

from gymnasium import make
from gymnasium.utils.env_checker import check_env

from drone_grid_env import assets

# Gather config files
config_files: list[Path] = []
for source in resources.files(assets).iterdir():
    with resources.as_file(source) as config_file:
        if not config_file.suffix == ".yaml":
            continue

        config_files.append(config_file)

for config_file in (Path(__file__).parent).glob("*.yaml"):
    config_files.append(config_file)

for config_file in Path("experiments").glob("**/env_config_*.yaml"):
    config_files.append(config_file)


@pytest.mark.parametrize("config_file", config_files)
def test_gym_valid_config_files(config_file: Path) -> None:
    if safe_load(config_file.read_text())["world"]["type"] != "SimWorld":
        pytest.skip()

    env = make("DroneGridEnv-v0", config_file=config_file)
    check_env(env.unwrapped)
    env.close()  # type: ignore[no-untyped-call]
