from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium import Env, make

from rl_zoo3.utils import get_wrapper_class
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecCheckNan,
    VecEnvWrapper,
    VecExtractDictObs,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
    VecTransposeImage,
    VecVideoRecorder,
)

from .exceptions import NotEnoughArgumentsException

if TYPE_CHECKING:
    from typing import Any, TypeVar

    from gymnasium.core import ActType, ObsType

    from stable_baselines3.common.vec_env import VecEnv

    _WrappedVecEnv = TypeVar("_WrappedVecEnv", bound=VecEnvWrapper)


def apply_wrappers(base_vec_env: _WrappedVecEnv, goal_vec_env: VecEnv) -> _WrappedVecEnv:
    _vec_env = base_vec_env
    _wrappers_list: list[type[VecEnvWrapper]] = [
        VecTransposeImage,
        VecVideoRecorder,
        VecNormalize,
        VecMonitor,
        VecExtractDictObs,
        VecCheckNan,
    ]
    _applied_wrappers: list[tuple[type[VecEnvWrapper], tuple[Any, ...]]] = []

    while isinstance(_vec_env, VecEnvWrapper):
        if isinstance(_vec_env, VecFrameStack):
            _applied_wrappers.append((VecFrameStack, (_vec_env.stacked_obs.n_stack,)))
        else:
            for wrapper_type in _wrappers_list:
                if isinstance(_vec_env, wrapper_type):
                    _applied_wrappers.append((wrapper_type, ()))

        _vec_env = _vec_env.venv  # type: ignore[assignment]

    for wrapper_type, wrapper_args in reversed(_applied_wrappers):
        goal_vec_env = wrapper_type(goal_vec_env, *wrapper_args)

    return goal_vec_env


def vec_env_to_gym_env(
    maybe_vec_env: VecEnv | Env[ObsType, ActType] | None,
    fallback_env: Env[ObsType, ActType] | str | None = None,
    fallback_env_kwargs: dict[str, Any] | None = None,
    fallback_env_wrapper: list[str] | None = None,
    render_mode: str | None = None,
) -> Env[ObsType, ActType]:
    if render_mode != None:
        if fallback_env_kwargs is not None:
            fallback_env_kwargs["render_mode"] = render_mode
        else:
            fallback_env_kwargs = {"render_mode": render_mode}

    if isinstance(maybe_vec_env, (DummyVecEnv, SubprocVecEnv)):
        if fallback_env is None:
            raise NotEnoughArgumentsException("When using a VecEnv, the 'fallback_env' argument should be specified in the yaml file!")

        # Try to reconstrunct the environment using the fallback
        if isinstance(fallback_env, str):
            env = make(fallback_env) if fallback_env_kwargs is None else make(fallback_env, **fallback_env_kwargs)
        else:
            env = fallback_env
            env.render_mode = render_mode

        if fallback_env_wrapper is not None:
            _wrappers = get_wrapper_class({"env_wrapper": fallback_env_wrapper})
            if _wrappers is not None:
                env = _wrappers(env)

    elif isinstance(maybe_vec_env, Env):
        env = maybe_vec_env

    else:
        raise NotImplementedError(f"Cannot copy {type(maybe_vec_env)} because method is not implemented!")

    env.reset()

    return env
