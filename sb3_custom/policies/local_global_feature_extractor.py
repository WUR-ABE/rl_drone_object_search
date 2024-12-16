from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict, cast

import torch as th
import torch.nn as nn

from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.type_aliases import TensorDict

    class _T_EXTRACTOR(TypedDict):
        conv_layer_num: int
        conv_kernel_size: int
        conv_kernels_num: int
        activation_fn: type[nn.Module]
        pooling_layer: type[nn.Module] | None
        pooling_layer_kwargs: dict[str, Any]


MULTI_INPUT_DEFAULT_CONFIG: dict[str, _T_EXTRACTOR] = {
    "local_map": {
        "conv_layer_num": 2,
        "conv_kernel_size": 5,
        "conv_kernels_num": 16,
        "activation_fn": nn.ReLU,
        "pooling_layer": None,
        "pooling_layer_kwargs": {},
    },
    "global_map": {
        "conv_layer_num": 2,
        "conv_kernel_size": 5,
        "conv_kernels_num": 16,
        "activation_fn": nn.ReLU,
        "pooling_layer": None,
        "pooling_layer_kwargs": {},
    },
}


def update(d: dict[str, Any], u: dict[str, Any]) -> dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update(d.get(k, {}), v)  # type: ignore[arg-type]
        else:
            d[k] = v
    return d


class LocalGlobalFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        config: dict[str, dict[str, _T_EXTRACTOR]] = {},
        verbose: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        _config = MULTI_INPUT_DEFAULT_CONFIG.copy()
        _config = update(_config, config)

        extractors: dict[str, nn.Module] = {}

        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            # Add single values to flatten layer
            if isinstance(subspace, spaces.Box) and subspace.shape == (1,):
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

            elif isinstance(subspace, spaces.Box):
                assert key in _config, "Config parameter must contain a entry for each key!"

                layers: list[nn.Module] = []
                prev_channels = subspace.shape[0]

                for _ in range(_config[key]["conv_layer_num"]):
                    layers.extend(
                        [
                            nn.Conv2d(
                                prev_channels,
                                _config[key]["conv_kernels_num"],
                                _config[key]["conv_kernel_size"],
                                stride=1,
                            ),
                            _config[key]["activation_fn"](),
                        ]
                    )
                    prev_channels = _config[key]["conv_kernels_num"]

                # Optionally, add pooling layer
                if _config[key]["pooling_layer"] is not None:
                    pl = _config[key]["pooling_layer"](**_config[key]["pooling_layer_kwargs"])  # type: ignore[misc]
                    layers.append(pl)

                layers.append(nn.Flatten())
                extractors[key] = nn.Sequential(*layers)

                total_concat_size += self.resolve_box_concat_size(subspace, extractors[key])
            else:
                raise NotImplementedError(f"Can not create network for type {type(subspace)} with key {key}!")

            if verbose:
                print(f"Adding feature extractor for {key} ({subspace.shape}):\n{extractors[key]}")

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

        if verbose:
            print(f"Total size of flatten layer: {total_concat_size}")

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return th.cat(encoded_tensor_list, dim=1)

    @staticmethod
    def resolve_box_concat_size(subspace: spaces.Box, network: nn.Module) -> int:
        with th.no_grad():
            tensor = th.from_numpy(subspace.sample()).float().unsqueeze(0)
            return cast(int, network(tensor).shape[1])
