from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING
import warnings
from warnings import warn

import gymnasium as gym
import numpy as np

from stable_baselines3.common.callbacks import EvalCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization

from drone_grid_env.wrappers import EpisodeInfoBufferWrapper
from sb3_custom.callbacks.utils import apply_wrappers, vec_env_to_gym_env

if TYPE_CHECKING:
    from typing import Any, Literal

    from gymnasium import Env
    from gymnasium.core import ActType, ObsType

    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.type_aliases import PolicyPredictor

KEYS_TO_IGNORE = {"episode", "terminal_observation", "TimeLimit.truncated"}


def custom_evaluate_policy(
    model: PolicyPredictor,
    env: gym.Env | VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    reward_threshold: float | None = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> tuple[float, float] | tuple[list[float], list[int]]:
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, action_values, states = model.predict(  # type: ignore[misc]
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )

        for i in range(env.num_envs):
            env.env_method("set_action_values", action_values[i, :], indices=i)

        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, f"Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


class EvaluationCallback(EvalCallback):
    """
    Evaluation callback which is equivalent to the stable_baselines3.common.callbacks.EvalCallback with
    a fixed seed. Because we use the rlzoo3 interface, we cannot add the environment as argument. Instead
    the environment will be copied from the training environment.
    """

    def __init__(
        self,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        n_eval_envs: int = 12,
        eval_freq: int = 1000,
        best_model_save_path: str | None = None,
        last_model_save_path: str | None = None,
        log_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        warn: bool = True,
        vec_env_class: Literal["dummy", "subproc"] = "dummy",
        fallback_env: Env[ObsType, ActType] | str | None = None,
        fallback_env_kwargs: dict[str, Any] | None = None,
        fallback_env_wrapper: list[str] | None = None,
        seed: int | None = None,
        verbose: int = 1,
    ):
        EventCallback.__init__(self, callback_after_eval, verbose)

        # Init from EvalCallback
        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward: float = -np.inf
        self.last_mean_reward: float = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        self.evaluations_results: list[float | list[float]] = []  # type: ignore[arg-type,assignment]
        self.evaluations_timesteps: list[int] = []
        self.evaluations_length: list[float | list[int]] = []  # type: ignore[arg-type,assignment]
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []

        self._fallback_env = fallback_env
        self._fallback_env_kwargs = fallback_env_kwargs
        self._fallback_env_wrapper = fallback_env_wrapper
        self._n_eval_envs = n_eval_envs
        self._vec_env_class = DummyVecEnv if vec_env_class == "dummy" else SubprocVecEnv
        self._seed = seed

        # Evaluation environment will be created later
        self.eval_env = None  # type: ignore[assignment]

        self.best_model_save_path = None
        if best_model_save_path is not None:
            if run_name := environ.get("WANDB_NAME", False):
                self.best_model_save_path = str(Path(best_model_save_path) / f"{run_name}_best.pt")
            else:
                self.best_model_save_path = str(Path(best_model_save_path) / "best.pt")

            print(f"Saving best model at {self.best_model_save_path}")

        self.last_model_save_path = None
        if last_model_save_path is not None:
            if run_name := environ.get("WANDB_NAME", False):
                self.last_model_save_path = str(Path(last_model_save_path) / f"{run_name}_last.pt")
            else:
                self.last_model_save_path = str(Path(last_model_save_path) / "last.pt")

            print(f"Saving last model at {self.last_model_save_path}")

        self.log_path = log_path
        if log_path is not None:
            if run_name := environ.get("WANDB_NAME", False):
                self.log_path = str(Path(log_path) / f"{run_name}_evaluations")
            else:
                self.log_path = str(Path(log_path) / "evaluations")

            print(f"Saving evaluation logs at {self.log_path}")

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warn(f"Training and eval env are not of the same type {self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None and not Path(self.best_model_save_path).parent.is_dir():
            Path(self.best_model_save_path).parent.mkdir(parents=True)

        if self.last_model_save_path is not None and not Path(self.last_model_save_path).parent.is_dir():
            Path(self.last_model_save_path).parent.mkdir(parents=True)

        if self.log_path is not None and not Path(self.log_path).parent.is_dir():
            Path(self.log_path).parent.mkdir()

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_env is None:
            training_env = vec_env_to_gym_env(
                self.training_env.unwrapped,
                fallback_env=self._fallback_env,
                fallback_env_kwargs=self._fallback_env_kwargs,
                fallback_env_wrapper=self._fallback_env_wrapper,
            )

            # Apply wrapper to store the info element when the environment was done
            training_env = EpisodeInfoBufferWrapper(training_env)
            self.eval_env = self._vec_env_class([lambda: deepcopy(training_env) for _ in range(self._n_eval_envs)])

            # Apply all wrappers from training env
            self.eval_env = apply_wrappers(self.training_env, self.eval_env)

        assert self.eval_env is not None
        assert self.training_env is not None

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            if self._seed is not None:
                self.eval_env.seed(self._seed)

            episode_rewards, episode_lengths = custom_evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(self.best_model_save_path)
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def _on_training_end(self) -> None:
        if self.last_model_save_path is not None:
            self.model.save(self.last_model_save_path)

    def _log_callback(self, locals: dict[str, Any], globals: dict[str, Any]) -> None:
        infos: list[dict[str, Any]] = [_info for _info in self.eval_env.get_attr("buffered_info") if _info is not None]

        for _info in infos:
            for k, v in _info.items():
                if k not in KEYS_TO_IGNORE:
                    self.logger.record(f"eval/{k}", v)
