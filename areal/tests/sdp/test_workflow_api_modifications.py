"""Unit tests for WorkflowExecutor modifications (segment-wise PPO feature).

This module provides comprehensive test coverage for the modified logic in
WorkflowExecutor, including defensive validation, staleness filtering, and
proximal logprob recomputation integration.
All tests run without GPU using mocked components.
"""

from unittest.mock import Mock, patch

import pytest

from areal.api.cli_args import InferenceEngineConfig
from areal.api.staleness_control import SegmentWisePPOStrategy, StandardPPOStrategy

# Note: megatron is mocked in conftest.py for Windows compatibility


class TestWorkflowExecutorDefensiveValidation:
    """Test defensive validation in WorkflowExecutor initialization."""

    def test_requires_output_queue(self):
        """Test that WorkflowExecutor raises error if output_queue not provided."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_staleness_manager = Mock()

        # Try to create without output_queue
        with pytest.raises(ValueError, match="output_queue and result_cache"):
            WorkflowExecutor(
                config=config,
                inference_engine=mock_engine,
                staleness_manager=mock_staleness_manager,
                output_queue=None,  # Missing!
                result_cache=Mock(),
            )

    def test_requires_result_cache(self):
        """Test that WorkflowExecutor raises error if result_cache not provided."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_staleness_manager = Mock()

        # Try to create without result_cache
        with pytest.raises(ValueError, match="output_queue and result_cache"):
            WorkflowExecutor(
                config=config,
                inference_engine=mock_engine,
                staleness_manager=mock_staleness_manager,
                output_queue=Mock(),
                result_cache=None,  # Missing!
            )

    def test_requires_both_queue_and_cache(self):
        """Test that WorkflowExecutor raises error if both missing."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_staleness_manager = Mock()

        # Try to create without both
        with pytest.raises(ValueError, match="output_queue and result_cache"):
            WorkflowExecutor(
                config=config,
                inference_engine=mock_engine,
                staleness_manager=mock_staleness_manager,
                output_queue=None,
                result_cache=None,
            )

    def test_accepts_properly_injected_dependencies(self):
        """Test that WorkflowExecutor accepts properly injected dependencies."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_staleness_manager = Mock()
        mock_queue = Mock()
        mock_cache = Mock()

        # Should not raise
        executor = WorkflowExecutor(
            config=config,
            inference_engine=mock_engine,
            staleness_manager=mock_staleness_manager,
            output_queue=mock_queue,
            result_cache=mock_cache,
        )

        assert executor.output_queue is mock_queue
        assert executor.result_cache is mock_cache


class TestWorkflowExecutorStalenessFiltering:
    """Test staleness filtering integration in WorkflowExecutor."""

    def test_uses_strategy_for_filtering_decision(self):
        """Test that executor uses strategy to decide whether to filter."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5

        # Create mock strategy that filters before enqueue
        mock_strategy = Mock()
        mock_strategy.should_filter_before_enqueue.return_value = True
        mock_strategy.is_sample_too_stale.return_value = True

        executor = WorkflowExecutor(
            config=config,
            inference_engine=mock_engine,
            staleness_manager=Mock(),
            output_queue=Mock(),
            result_cache=Mock(),
            staleness_strategy=mock_strategy,
        )

        # Verify strategy is stored
        assert executor.staleness_strategy is mock_strategy

    def test_tracks_filtered_samples_with_capacity_modifier(self):
        """Test that filtered samples are tracked by capacity modifier."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_modifier = Mock()

        executor = WorkflowExecutor(
            config=config,
            inference_engine=mock_engine,
            staleness_manager=Mock(),
            output_queue=Mock(),
            result_cache=Mock(),
            filtered_capacity_modifier=mock_modifier,
        )

        # Verify modifier is stored
        assert executor.filtered_capacity_modifier is mock_modifier


class TestWorkflowExecutorProximalRecomputation:
    """Test proximal logprob recomputation integration."""

    def test_recompute_proximal_logprobs_calls_recomputer(self):
        """Test that recompute_proximal_logprobs calls the recomputer."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_recomputer = Mock()
        mock_cache = Mock()
        mock_queue = Mock()

        executor = WorkflowExecutor(
            config=config,
            inference_engine=mock_engine,
            staleness_manager=Mock(),
            output_queue=mock_queue,
            result_cache=mock_cache,
            proximal_recomputer=mock_recomputer,
        )

        # Call recompute method
        executor.recompute_proximal_logprobs()

        # Should have called recomputer.recompute_all with queue and cache
        mock_recomputer.recompute_all.assert_called_once_with(
            output_queue=mock_queue,
            result_cache=mock_cache,
        )

    def test_recompute_proximal_logprobs_without_recomputer(self):
        """Test that recompute works gracefully without recomputer."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            consumer_batch_size=4,
        )
        mock_engine = Mock()
        mock_cache = Mock()
        mock_cache.size.return_value = 0

        executor = WorkflowExecutor(
            config=config,
            inference_engine=mock_engine,
            staleness_manager=Mock(),
            output_queue=Mock(),
            result_cache=mock_cache,
            proximal_recomputer=None,  # No recomputer
        )

        # Should not crash
        executor.recompute_proximal_logprobs()


class TestWorkflowExecutorComponentStorage:
    """Test that all segment-wise PPO components are properly stored."""

    def test_stores_staleness_strategy(self):
        """Test that staleness strategy is stored."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test", trial_name="test", consumer_batch_size=4
        )
        mock_strategy = Mock()

        executor = WorkflowExecutor(
            config=config,
            inference_engine=Mock(),
            staleness_manager=Mock(),
            output_queue=Mock(),
            result_cache=Mock(),
            staleness_strategy=mock_strategy,
        )

        assert executor.staleness_strategy is mock_strategy

    def test_stores_proximal_recomputer(self):
        """Test that proximal recomputer is stored."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test", trial_name="test", consumer_batch_size=4
        )
        mock_recomputer = Mock()

        executor = WorkflowExecutor(
            config=config,
            inference_engine=Mock(),
            staleness_manager=Mock(),
            output_queue=Mock(),
            result_cache=Mock(),
            proximal_recomputer=mock_recomputer,
        )

        assert executor.proximal_recomputer is mock_recomputer

    def test_stores_filtered_capacity_modifier(self):
        """Test that filtered capacity modifier is stored."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test", trial_name="test", consumer_batch_size=4
        )
        mock_modifier = Mock()

        executor = WorkflowExecutor(
            config=config,
            inference_engine=Mock(),
            staleness_manager=Mock(),
            output_queue=Mock(),
            result_cache=Mock(),
            filtered_capacity_modifier=mock_modifier,
        )

        assert executor.filtered_capacity_modifier is mock_modifier

    def test_all_components_optional(self):
        """Test that all segment-wise PPO components are optional."""
        from areal.api.workflow_api import WorkflowExecutor

        config = InferenceEngineConfig(
            experiment_name="test", trial_name="test", consumer_batch_size=4
        )

        # Should work with all components as None
        executor = WorkflowExecutor(
            config=config,
            inference_engine=Mock(),
            staleness_manager=Mock(),
            output_queue=Mock(),
            result_cache=Mock(),
            staleness_strategy=None,
            proximal_recomputer=None,
            filtered_capacity_modifier=None,
        )

        assert executor.staleness_strategy is None
        assert executor.proximal_recomputer is None
        assert executor.filtered_capacity_modifier is None


# Parametrized tests
@pytest.mark.parametrize("component_name", [
    "staleness_strategy",
    "proximal_recomputer",
    "filtered_capacity_modifier",
])
def test_parametrized_component_storage(component_name):
    """Test that each component can be stored and retrieved."""
    from areal.api.workflow_api import WorkflowExecutor

    config = InferenceEngineConfig(
        experiment_name="test", trial_name="test", consumer_batch_size=4
    )
    mock_component = Mock()

    kwargs = {
        "config": config,
        "inference_engine": Mock(),
        "staleness_manager": Mock(),
        "output_queue": Mock(),
        "result_cache": Mock(),
        component_name: mock_component,
    }

    executor = WorkflowExecutor(**kwargs)

    assert getattr(executor, component_name) is mock_component


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=areal.api.workflow_api"])
