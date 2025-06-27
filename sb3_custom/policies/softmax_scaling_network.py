from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch as th
import torch.nn as nn

from stable_baselines3.dqn.policies import QNetwork

if TYPE_CHECKING:
    from gymnasium import spaces

    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SoftmaxScalingQNetwork(QNetwork):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images,
        )

        self._softmax = nn.Softmax(dim=1)
        self.softmax_scaling_value: float | None = None

    def _predict(self, observation: th.Tensor | dict[str, th.Tensor], deterministic: bool = True) -> tuple[th.Tensor, th.Tensor]:
        assert self.softmax_scaling_value is not None

        q_values: th.Tensor = self(observation)

        if deterministic:
            # Greedy action
            return q_values.argmax(dim=1).reshape(-1), q_values

        # Sample action from distribution with q_values as weight
        # th.random.choice does not exist, so use th.multinomial:
        # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/4
        p = self._softmax(q_values / self.softmax_scaling_value)
        return th.multinomial(p, 1, replacement=True).reshape(-1), q_values
