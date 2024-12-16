from __future__ import annotations

import wandb

from stable_baselines3.common.callbacks import BaseCallback

try:
    from codecarbon import OfflineEmissionsTracker

    _CODECARBON_AVAILBLE = True
except ImportError:
    _CODECARBON_AVAILBLE = False


class CodeCarbonCallback(BaseCallback):
    def __init__(self, verbose: int = 0, country_iso_code: str = "NLD", save_path: str = "emissions"):
        super().__init__(verbose)

        self._tracker: OfflineEmissionsTracker | None = None
        self._country_iso_code = country_iso_code
        self._save_path = save_path

    def _on_training_start(self) -> None:
        if _CODECARBON_AVAILBLE:
            # Retrieve stuff from WandB
            if wandb.run is not None:
                self._tracker = OfflineEmissionsTracker(
                    country_iso_code=self._country_iso_code,
                    project_name=f"{wandb.run.project_name()}/{wandb.run.id}",
                    experiment_id=wandb.run.id,
                    experiment_name=wandb.run.name,
                    log_level="warning",
                )
            else:
                self._tracker = OfflineEmissionsTracker(
                    country_iso_code=self._country_iso_code,
                )
            self._tracker.start()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if _CODECARBON_AVAILBLE and self._tracker is not None:
            self._tracker.stop()

            if wandb.run is not None:
                if "energy_consumed" not in wandb.config:
                    wandb.config["energy_consumed"] = self._tracker.final_emissions_data.energy_consumed  # KwH

                if "total_emmissions" not in wandb.config:
                    wandb.config["total_emissions"] = self._tracker.final_emissions_data.emissions  # CO2 eqv

            self._tracker = None
