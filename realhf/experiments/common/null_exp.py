# Copyright 2025 Ant Group Inc.

import dataclasses

from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.quickstart.dataset import (
    PromptAnswerDatasetConfig,
    PromptOnlyDatasetConfig,
)
from realhf.api.quickstart.device_mesh import MFCConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig
from realhf.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class NullSFTConfig(CommonExperimentConfig):
    """Configuration for a null SFT experiment. Used for testing purposes.

    :param allocation: Configuration for device allocation and
        parallelism.
    :type allocation: MFCConfig
    :param dataset: Configuration for the dataset.
    :type dataset: PromptAnswerDatasetConfig
    """

    model: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    allocation: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    dataset: PromptAnswerDatasetConfig = dataclasses.field(
        default_factory=PromptAnswerDatasetConfig
    )

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        rpc = MFCDef(
            n_seqs=self.dataset.train_bs_n_seqs,
            name="trainDefault",
            mb_spec=self.allocation.mb_spec,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=ModelInterfaceAbstraction("null"),
            model_name="default",
            input_keys=["packed_input_ids", "prompt_mask"],
            log_return_value=True,
            model_type=self.model.type,
            model_path=self.model.path,
        )
        return {"trainDefault": rpc}

    @property
    def allocations(self):
        return {"trainDefault": self.allocation}

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "prompt_answer",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    dataset_path=self.dataset.train_path,
                    fill_to_max_length=self.dataset.fill_to_max_length,
                ),
            )
        ]

    @property
    def tokenizer_name_or_path(self):
        return self.model.path


register_quickstart_exp("null-sft", NullSFTConfig)


@dataclasses.dataclass
class NullPPOConfig(CommonExperimentConfig):
    """Configuration for a null PPO experiment.

    Used for testing purposes.
    """

    model: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    dataset: PromptOnlyDatasetConfig = dataclasses.field(
        default_factory=PromptOnlyDatasetConfig
    )
    dataset_filter_threshold: float = 0.2
    dataset_max_filter_percentage: float = 0.1

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        rw = MFCDef(
            n_seqs=self.dataset.train_bs_n_seqs,
            name="reward",
            mb_spec=self.inf.mb_spec,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=ModelInterfaceAbstraction("null"),
            model_name="default",
            input_keys=["packed_prompts"],
            output_keys=["rewards"],
            model_type=self.model.type,
            model_path=self.model.path,
        )
        rpc = MFCDef(
            n_seqs=self.dataset.train_bs_n_seqs,
            name="trainDefault",
            mb_spec=self.train.mb_spec,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=ModelInterfaceAbstraction("null"),
            model_name="default",
            input_keys=["packed_prompts", "rewards"],
            log_return_value=True,
            model_type=self.model.type,
            model_path=self.model.path,
        )
        return {"trainDefault": rpc, "reward": rw}

    @property
    def allocations(self):
        return {"trainDefault": self.train, "reward": self.inf}

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "math_prompt",
                args=dict(
                    max_length=self.dataset.max_prompt_len,
                    dataset_path=self.dataset.path,
                    fill_to_max_length=self.dataset.fill_to_max_length,
                    filter_threshold=self.dataset_filter_threshold,
                    max_filter_percentage=self.dataset_max_filter_percentage,
                ),
            )
        ]

    @property
    def tokenizer_name_or_path(self):
        return self.model.path


register_quickstart_exp("null-ppo", NullPPOConfig)
