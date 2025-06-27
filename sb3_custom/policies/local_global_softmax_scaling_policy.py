from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import torch as th
import torch.nn as nn

from gymnasium import spaces

from stable_baselines3.dqn.policies import DQNPolicy

from .local_global_feature_extractor import LocalGlobalFeatureExtractor
from .softmax_scaling_network import SoftmaxScalingQNetwork

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.type_aliases import Schedule


class LocalGlobalSoftmaxScalingPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: spaces.Space[Any],
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        softmax_scaling: float = 0.1,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=LocalGlobalFeatureExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        # Update global variable because we don't have access to the constructor of the network
        self.q_net.softmax_scaling_value = softmax_scaling  # type: ignore[assignment]
        self.q_net_target.softmax_scaling_value = softmax_scaling  # type: ignore[assignment]

    def make_q_net(self) -> SoftmaxScalingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        q_net = SoftmaxScalingQNetwork(**net_args)
        q_net.to(self.device)
        return q_net

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions, action_values = self.q_net._predict(obs_tensor, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        action_values = action_values.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, action_values, state  # type: ignore[return-value]
