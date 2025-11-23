import getpass
import os

from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import SaverConfig
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta
from areal.controller.train_controller import TrainController
from areal.utils import timeutil
from areal.utils.checkpoint_retention import (
    CheckpointRetentionManager,
    RetentionPolicy,
)


class Saver:
    def __init__(self, config: SaverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )
        self._retention_managers: dict[str, CheckpointRetentionManager] = {}
        self._retention_initialized = False

    @staticmethod
    def get_save_root(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        path = os.path.join(
            f"{fileroot}/checkpoints/{getpass.getuser()}/{experiment_name}/{trial_name}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_model_save_root(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_save_root(experiment_name, trial_name, fileroot),
            name,
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_model_save_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        epoch: int,
        step: int,
        globalstep: int,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_model_save_root(experiment_name, trial_name, fileroot, name),
            f"epoch{epoch}epochstep{step}globalstep{globalstep}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_recover_checkpoint_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_model_save_root(experiment_name, trial_name, fileroot, name),
            "recover_checkpoint",
        )
        os.makedirs(path, exist_ok=True)
        return path

    def state_dict(self):
        return self.freq_ctl.state_dict()

    def load_state_dict(self, state_dict):
        self.freq_ctl.load_state_dict(state_dict)

    def _get_retention_manager(
        self, name: str = "default"
    ) -> CheckpointRetentionManager | None:
        """Get or create retention manager for a model."""
        if not self.config.enable_retention:
            return None

        if name not in self._retention_managers:
            model_save_root = Saver.get_model_save_root(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.fileroot,
                name,
            )

            # Determine archive root
            archive_root = self.config.archive_root
            if archive_root is None:
                archive_root = os.path.join(
                    f"{self.config.fileroot}/checkpoints_archive/{getpass.getuser()}/"
                    f"{self.config.experiment_name}/{self.config.trial_name}",
                    name,
                )

            # Create retention policies
            epoch_policy = RetentionPolicy(
                max_to_keep=self.config.epoch_max_to_keep,
                cleanup_action=self.config.epoch_cleanup_action,
                archive_root=(
                    archive_root
                    if self.config.epoch_cleanup_action != "delete"
                    else None
                ),
                protect=self.config.protect_epoch_checkpoints,
            )

            step_policy = RetentionPolicy(
                max_to_keep=self.config.step_max_to_keep,
                cleanup_action=self.config.step_cleanup_action,
                archive_root=(
                    archive_root
                    if self.config.step_cleanup_action != "delete"
                    else None
                ),
                protect=False,
            )

            # Create retention manager
            manager = CheckpointRetentionManager(
                model_save_root=model_save_root,
                epoch_policy=epoch_policy,
                step_policy=step_policy,
            )

            # On first initialization, scan existing checkpoints for backward compatibility
            if not self._retention_initialized:
                manager.scan_and_register_existing_checkpoints(
                    steps_per_epoch=self.ft_spec.steps_per_epoch
                )
                self._retention_initialized = True

            self._retention_managers[name] = manager

        return self._retention_managers[name]

    def save(
        self,
        engine: TrainEngine | TrainController,
        epoch: int,
        step: int,
        global_step: int,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        # Check if we should save based on frequency
        is_epoch_end = step == self.ft_spec.steps_per_epoch - 1
        if not self.freq_ctl.check(epochs=int(is_epoch_end), steps=1):
            return

        # Save checkpoint
        path = Saver.get_model_save_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            epoch,
            step,
            global_step,
            name,
        )
        weight_format = "hf"
        with_optim = False
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            processor=processor,
            base_model_path=base_model_path,
        )
        engine.save(meta)

        # Register checkpoint with retention manager
        retention_manager = self._get_retention_manager(name)
        if retention_manager is not None:
            retention_manager.register_checkpoint(
                checkpoint_path=path,
                epoch=epoch,
                step=step,
                global_step=global_step,
                is_epoch_checkpoint=is_epoch_end,
            )
