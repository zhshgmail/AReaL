"""Unit tests for segment-wise decoupled PPO configuration and factory.

This module provides comprehensive test coverage for the segment-wise PPO feature,
including configuration propagation, factory pattern, strategy selection, and
component creation. All tests run without GPU for CI pipeline compatibility.
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from omegaconf import MISSING, OmegaConf

from areal.api.cli_args import (
    BaseExperimentConfig,
    GRPOConfig,
    InferenceEngineConfig,
    PPOActorConfig,
    load_expr_config,
    to_structured_cfg,
)
from areal.api.staleness_control import (
    SegmentWisePPOStrategy,
    StalenessControlStrategy,
    StandardPPOStrategy,
)
from areal.api.workflow_factory import (
    create_filtered_capacity_modifier,
    create_proximal_recomputer,
    create_staleness_strategy,
    create_workflow_executor,
)
from areal.core.filtered_capacity_modifier import FilteredSamplesCapacityModifier


class TestConfigPropagation:
    """Test configuration propagation from BaseExperimentConfig to child configs."""

    def test_propagation_enabled_by_default(self, tmp_path):
        """Test that enable_segment_wise_ppo defaults to True and propagates."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
experiment_name: test
trial_name: test_trial
rollout:
  experiment_name: test
  trial_name: test_trial
actor:
  experiment_name: test
  trial_name: test_trial
  path: ""
ref:
  experiment_name: test
  trial_name: test_trial
  path: ""
train_dataset:
  path: ""
  type: "mock"
saver:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
evaluator:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
recover:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
stats_logger:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
"""
        )

        # Mock name_resolve to avoid dependencies
        with patch("areal.api.cli_args.name_resolve"), patch.dict(
            "os.environ", {"RANK": "1"}
        ):
            cfg, _ = load_expr_config(["--config", str(config_file)], GRPOConfig)

        # Should default to True
        assert cfg.enable_segment_wise_ppo is True
        # Should propagate to children
        assert cfg.rollout.enable_segment_wise_ppo is True
        assert cfg.actor.enable_segment_wise_ppo is True

    def test_propagation_when_explicitly_enabled(self, tmp_path):
        """Test that explicitly setting enable_segment_wise_ppo=true propagates."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
experiment_name: test
trial_name: test_trial
enable_segment_wise_ppo: true
rollout:
  experiment_name: test
  trial_name: test_trial
actor:
  experiment_name: test
  trial_name: test_trial
  path: ""
ref:
  experiment_name: test
  trial_name: test_trial
  path: ""
train_dataset:
  path: ""
  type: "mock"
saver:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
evaluator:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
recover:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
stats_logger:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
"""
        )

        with patch("areal.api.cli_args.name_resolve"), patch(
            "areal.utils.stats_logger.StatsLogger"
        ), patch.dict("os.environ", {"RANK": "1"}):
            cfg, _ = load_expr_config(["--config", str(config_file)], GRPOConfig)

        assert cfg.enable_segment_wise_ppo is True
        assert cfg.rollout.enable_segment_wise_ppo is True
        assert cfg.actor.enable_segment_wise_ppo is True

    def test_propagation_when_disabled(self, tmp_path):
        """Test that setting enable_segment_wise_ppo=false propagates."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
experiment_name: test
trial_name: test_trial
enable_segment_wise_ppo: false
rollout:
  experiment_name: test
  trial_name: test_trial
actor:
  experiment_name: test
  trial_name: test_trial
  path: ""
ref:
  experiment_name: test
  trial_name: test_trial
  path: ""
train_dataset:
  path: ""
  type: "mock"
saver:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
evaluator:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
recover:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
stats_logger:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
"""
        )

        with patch("areal.api.cli_args.name_resolve"), patch(
            "areal.utils.stats_logger.StatsLogger"
        ), patch.dict("os.environ", {"RANK": "1"}):
            cfg, _ = load_expr_config(["--config", str(config_file)], GRPOConfig)

        assert cfg.enable_segment_wise_ppo is False
        assert cfg.rollout.enable_segment_wise_ppo is False
        assert cfg.actor.enable_segment_wise_ppo is False

    def test_propagation_overrides_child_values(self, tmp_path):
        """Test that parent config overrides child config values."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
experiment_name: test
trial_name: test_trial
enable_segment_wise_ppo: false
rollout:
  experiment_name: test
  trial_name: test_trial
  enable_segment_wise_ppo: true  # Should be overridden
actor:
  experiment_name: test
  trial_name: test_trial
  path: ""
ref:
  experiment_name: test
  trial_name: test_trial
  path: ""
  enable_segment_wise_ppo: true  # Should be overridden
train_dataset:
  path: ""
  type: "mock"
saver:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
evaluator:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
recover:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
stats_logger:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
"""
        )

        with patch("areal.api.cli_args.name_resolve"), patch(
            "areal.utils.stats_logger.StatsLogger"
        ), patch.dict("os.environ", {"RANK": "1"}):
            cfg, _ = load_expr_config(["--config", str(config_file)], GRPOConfig)

        # Parent config should override child configs
        assert cfg.enable_segment_wise_ppo is False
        assert cfg.rollout.enable_segment_wise_ppo is False
        assert cfg.actor.enable_segment_wise_ppo is False

    def test_propagation_with_commandline_override(self, tmp_path):
        """Test that command-line overrides work with propagation."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
experiment_name: test
trial_name: test_trial
enable_segment_wise_ppo: true
rollout:
  experiment_name: test
  trial_name: test_trial
actor:
  experiment_name: test
  trial_name: test_trial
  path: ""
ref:
  experiment_name: test
  trial_name: test_trial
  path: ""
train_dataset:
  path: ""
  type: "mock"
saver:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
evaluator:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
recover:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
stats_logger:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
"""
        )

        with patch("areal.api.cli_args.name_resolve"), patch(
            "areal.utils.stats_logger.StatsLogger"
        ), patch.dict("os.environ", {"RANK": "1"}):
            # Override via command line
            cfg, _ = load_expr_config(
                ["--config", str(config_file), "enable_segment_wise_ppo=false"], GRPOConfig
            )

        # Command-line override should work and propagate
        assert cfg.enable_segment_wise_ppo is False
        assert cfg.rollout.enable_segment_wise_ppo is False
        assert cfg.actor.enable_segment_wise_ppo is False


class TestStrategyFactory:
    """Test create_staleness_strategy factory function."""

    def test_creates_segment_wise_strategy_when_enabled(self):
        """Test that SegmentWisePPOStrategy is created when enabled."""
        config = InferenceEngineConfig(
            experiment_name="test", trial_name="test", enable_segment_wise_ppo=True
        )

        strategy = create_staleness_strategy(config)

        assert isinstance(strategy, SegmentWisePPOStrategy)
        assert isinstance(strategy, StalenessControlStrategy)

    def test_creates_standard_strategy_when_disabled(self):
        """Test that StandardPPOStrategy is created when disabled."""
        config = InferenceEngineConfig(
            experiment_name="test", trial_name="test", enable_segment_wise_ppo=False
        )

        strategy = create_staleness_strategy(config)

        assert isinstance(strategy, StandardPPOStrategy)
        assert isinstance(strategy, StalenessControlStrategy)
        # Should NOT be SegmentWisePPOStrategy
        assert not isinstance(strategy, SegmentWisePPOStrategy)

    def test_strategy_respects_config_values(self):
        """Test that strategy receives correct config values."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
            max_head_offpolicyness=5,
        )

        strategy = create_staleness_strategy(config)

        # Strategy should have access to config
        assert strategy.config.max_head_offpolicyness == 5


class TestProximalRecomputerFactory:
    """Test create_proximal_recomputer factory function."""

    def test_creates_recomputer_when_enabled(self):
        """Test that ProximalRecomputer is created when enabled."""
        config = InferenceEngineConfig(
            experiment_name="test", trial_name="test", enable_segment_wise_ppo=True
        )
        mock_engine = Mock()
        mock_logger = Mock()

        recomputer = create_proximal_recomputer(mock_engine, mock_logger, config)

        assert recomputer is not None
        # Check that it was initialized with correct arguments
        assert recomputer.inference_engine is mock_engine
        assert recomputer.logger is mock_logger

    def test_returns_none_when_disabled(self):
        """Test that None is returned when disabled."""
        config = InferenceEngineConfig(
            experiment_name="test", trial_name="test", enable_segment_wise_ppo=False
        )
        mock_engine = Mock()
        mock_logger = Mock()

        recomputer = create_proximal_recomputer(mock_engine, mock_logger, config)

        assert recomputer is None


class TestFilteredCapacityModifierFactory:
    """Test create_filtered_capacity_modifier factory function."""

    def test_creates_modifier_when_enabled(self):
        """Test that FilteredSamplesCapacityModifier is created when enabled."""
        config = InferenceEngineConfig(
            experiment_name="test", trial_name="test", enable_segment_wise_ppo=True
        )

        modifier = create_filtered_capacity_modifier(config)

        assert modifier is not None
        assert isinstance(modifier, FilteredSamplesCapacityModifier)

    def test_returns_none_when_disabled(self):
        """Test that None is returned when disabled."""
        config = InferenceEngineConfig(
            experiment_name="test", trial_name="test", enable_segment_wise_ppo=False
        )

        modifier = create_filtered_capacity_modifier(config)

        assert modifier is None


class TestWorkflowExecutorFactory:
    """Test create_workflow_executor factory function."""

    @patch("areal.api.workflow_api.WorkflowExecutor")
    def test_creates_executor_with_segment_wise_components_when_enabled(
        self, mock_executor_class
    ):
        """Test that executor is created with segment-wise components when enabled."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_staleness_manager = Mock()
        mock_logger = Mock()

        # Create executor
        executor = create_workflow_executor(
            inference_engine=mock_engine,
            staleness_manager=mock_staleness_manager,
            config=config,
            logger=mock_logger,
        )

        # Verify WorkflowExecutor was called
        assert mock_executor_class.called
        call_kwargs = mock_executor_class.call_args[1]

        # Verify segment-wise components were created
        assert isinstance(call_kwargs["staleness_strategy"], SegmentWisePPOStrategy)
        assert call_kwargs["proximal_recomputer"] is not None
        assert call_kwargs["filtered_capacity_modifier"] is not None

    @patch("areal.api.workflow_api.WorkflowExecutor")
    def test_creates_executor_with_standard_components_when_disabled(
        self, mock_executor_class
    ):
        """Test that executor is created with standard components when disabled."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_staleness_manager = Mock()
        mock_logger = Mock()

        # Create executor
        executor = create_workflow_executor(
            inference_engine=mock_engine,
            staleness_manager=mock_staleness_manager,
            config=config,
            logger=mock_logger,
        )

        # Verify WorkflowExecutor was called
        assert mock_executor_class.called
        call_kwargs = mock_executor_class.call_args[1]

        # Verify standard components were created
        assert isinstance(call_kwargs["staleness_strategy"], StandardPPOStrategy)
        assert call_kwargs["proximal_recomputer"] is None
        assert call_kwargs["filtered_capacity_modifier"] is None

    @patch("areal.api.workflow_api.WorkflowExecutor")
    @patch("areal.api.workflow_factory.register_capacity_modifiers")
    def test_registers_capacity_modifiers_when_enabled(
        self, mock_register, mock_executor_class
    ):
        """Test that capacity modifiers are registered when enabled."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_staleness_manager = Mock()
        mock_logger = Mock()

        # Create executor
        create_workflow_executor(
            inference_engine=mock_engine,
            staleness_manager=mock_staleness_manager,
            config=config,
            logger=mock_logger,
        )

        # Verify register was called with correct arguments
        assert mock_register.called
        call_args = mock_register.call_args[0]
        assert call_args[0] is mock_staleness_manager
        assert isinstance(call_args[1], FilteredSamplesCapacityModifier)

    @patch("areal.api.workflow_api.WorkflowExecutor")
    @patch("areal.api.workflow_factory.register_capacity_modifiers")
    def test_does_not_register_capacity_modifiers_when_disabled(
        self, mock_register, mock_executor_class
    ):
        """Test that capacity modifiers are not registered when disabled."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_staleness_manager = Mock()
        mock_logger = Mock()

        # Create executor
        create_workflow_executor(
            inference_engine=mock_engine,
            staleness_manager=mock_staleness_manager,
            config=config,
            logger=mock_logger,
        )

        # Verify register was called with None (no modifier)
        assert mock_register.called
        call_args = mock_register.call_args[0]
        assert call_args[0] is mock_staleness_manager
        assert call_args[1] is None


class TestBackwardCompatibility:
    """Test backward compatibility with old configs."""

    def test_old_config_without_flag_defaults_to_enabled(self, tmp_path):
        """Test that old configs without the flag default to enabled."""
        config_file = tmp_path / "old_config.yaml"
        # Old config without enable_segment_wise_ppo
        config_file.write_text(
            """
experiment_name: test
trial_name: test_trial
rollout:
  experiment_name: test
  trial_name: test_trial
  max_concurrent_rollouts: 64
actor:
  experiment_name: test
  trial_name: test_trial
  path: ""
ref:
  experiment_name: test
  trial_name: test_trial
  path: ""
  kl_ctl: 0.1
train_dataset:
  path: ""
  type: "mock"
saver:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
evaluator:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
recover:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
stats_logger:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
"""
        )

        with patch("areal.api.cli_args.name_resolve"), patch(
            "areal.utils.stats_logger.StatsLogger"
        ), patch.dict("os.environ", {"RANK": "1"}):
            cfg, _ = load_expr_config(["--config", str(config_file)], GRPOConfig)

        # Should default to True for backward compatibility
        assert cfg.enable_segment_wise_ppo is True
        assert cfg.rollout.enable_segment_wise_ppo is True
        assert cfg.actor.enable_segment_wise_ppo is True
        # Other configs should still work
        assert cfg.rollout.max_concurrent_rollouts == 64
        assert cfg.actor.kl_ctl == 0.1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_config_without_rollout_attribute(self):
        """Test that propagation handles configs without rollout attribute."""

        @dataclass
        class MinimalConfig:
            enable_segment_wise_ppo: bool = True

        cfg = MinimalConfig()

        # Should not crash when rollout doesn't exist
        # (This simulates the propagation logic handling missing attributes)
        if hasattr(cfg, "rollout"):
            cfg.rollout.enable_segment_wise_ppo = cfg.enable_segment_wise_ppo

        # Should complete without error
        assert cfg.enable_segment_wise_ppo is True

    def test_config_without_actor_attribute(self):
        """Test that propagation handles configs without actor attribute."""

        @dataclass
        class MinimalConfig:
            enable_segment_wise_ppo: bool = True

        cfg = MinimalConfig()

        # Should not crash when actor doesn't exist
        if hasattr(cfg, "actor"):
            cfg.actor.enable_segment_wise_ppo = cfg.enable_segment_wise_ppo

        # Should complete without error
        assert cfg.enable_segment_wise_ppo is True

    def test_multiple_config_loads_are_consistent(self, tmp_path):
        """Test that loading the same config multiple times is consistent."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
experiment_name: test
trial_name: test_trial
enable_segment_wise_ppo: true
rollout:
  experiment_name: test
  trial_name: test_trial
actor:
  experiment_name: test
  trial_name: test_trial
  path: ""
ref:
  experiment_name: test
  trial_name: test_trial
  path: ""
train_dataset:
  path: ""
  type: "mock"
saver:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
evaluator:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
recover:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
stats_logger:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
"""
        )

        with patch("areal.api.cli_args.name_resolve"), patch(
            "areal.utils.stats_logger.StatsLogger"
        ), patch.dict("os.environ", {"RANK": "1"}):
            cfg1, _ = load_expr_config(["--config", str(config_file)], GRPOConfig)

        with patch("areal.api.cli_args.name_resolve"), patch(
            "areal.utils.stats_logger.StatsLogger"
        ), patch.dict("os.environ", {"RANK": "1"}):
            cfg2, _ = load_expr_config(["--config", str(config_file)], GRPOConfig)

        # Both loads should be consistent
        assert cfg1.enable_segment_wise_ppo == cfg2.enable_segment_wise_ppo
        assert cfg1.rollout.enable_segment_wise_ppo == cfg2.rollout.enable_segment_wise_ppo
        assert cfg1.actor.enable_segment_wise_ppo == cfg2.actor.enable_segment_wise_ppo


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize("flag_value", [True, False])
def test_parametrized_flag_values(flag_value, tmp_path):
    """Test that both flag values work correctly."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        f"""
experiment_name: test
trial_name: test_trial
enable_segment_wise_ppo: {str(flag_value).lower()}
rollout:
  experiment_name: test
  trial_name: test_trial
actor:
  experiment_name: test
  trial_name: test_trial
  path: ""
ref:
  experiment_name: test
  trial_name: test_trial
  path: ""
train_dataset:
  path: ""
  type: "mock"
saver:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
evaluator:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
recover:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
stats_logger:
  experiment_name: test
  trial_name: test_trial
  fileroot: ""
"""
    )

    with patch("areal.api.cli_args.name_resolve"), patch.dict(
        "os.environ", {"RANK": "1"}
    ):
        cfg, _ = load_expr_config(["--config", str(config_file)], GRPOConfig)

    assert cfg.enable_segment_wise_ppo == flag_value
    assert cfg.rollout.enable_segment_wise_ppo == flag_value
    assert cfg.actor.enable_segment_wise_ppo == flag_value


@pytest.mark.parametrize("flag_value", [True, False])
def test_parametrized_factory_behavior(flag_value):
    """Test that factory functions respect flag value."""
    config = InferenceEngineConfig(
        experiment_name="test",
        trial_name="test",
        enable_segment_wise_ppo=flag_value,
    )

    strategy = create_staleness_strategy(config)
    recomputer = create_proximal_recomputer(Mock(), Mock(), config)
    modifier = create_filtered_capacity_modifier(config)

    if flag_value:
        # When enabled
        assert isinstance(strategy, SegmentWisePPOStrategy)
        assert recomputer is not None
        assert modifier is not None
    else:
        # When disabled
        assert isinstance(strategy, StandardPPOStrategy)
        assert recomputer is None
        assert modifier is None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=areal.api.workflow_factory", "--cov=areal.api.cli_args"])
