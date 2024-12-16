from __future__ import annotations

from warnings import filterwarnings

from rl_zoo3.utils import ALGOS

from drone_grid_env import logger
from sb3_custom.custom_dqn_algorithm import CustomDQN
from sb3_custom.double_dqn_algorithm import DoubleDQN

filterwarnings("ignore", category=UserWarning)  # Ignore Gymnasium UserWarning

# Register custom algorithms
ALGOS["ddqn"] = DoubleDQN
ALGOS["custom_dqn"] = CustomDQN


logger.set_level(logger.WARN)


def main() -> None:
    from rl_zoo3.train import train as rlzoo3_train

    # Use Stable Baselines Zoo training script
    rlzoo3_train()


if __name__ == "__main__":
    main()
