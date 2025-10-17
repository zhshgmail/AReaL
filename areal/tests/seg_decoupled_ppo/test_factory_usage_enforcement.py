"""
Tests for factory usage enforcement and guardrails.

These tests verify that:
1. Direct constructor usage with enable_segment_wise_ppo=True triggers warning
2. Factory usage with enable_segment_wise_ppo=True does NOT trigger warning
3. Backward compatibility: direct constructor without SDP enabled works fine
4. Warning messages are helpful and accurate
"""

import sys
import warnings
from unittest.mock import MagicMock, Mock

import pytest

# Mock uvloop for Windows compatibility
if sys.platform == 'win32':
    sys.modules['uvloop'] = MagicMock()

# Mock megatron to avoid import dependencies
sys.modules['megatron'] = MagicMock()
sys.modules['megatron.core'] = MagicMock()
sys.modules['megatron.core.parallel_state'] = MagicMock()

from areal.api.workflow_api import WorkflowExecutor, create_workflow_executor


class InferenceEngineConfig:
    """Mock config for testing."""

    def __init__(self, enable_sdp=False):
        self.max_concurrent_rollouts = 4
        self.consumer_batch_size = 2
        self.queue_size = 16
        self.max_head_offpolicyness = 2
        self.enable_rollout_tracing = False
        self.request_timeout = 30
        self.enable_segment_wise_ppo = enable_sdp


class MockInferenceEngine:
    """Mock inference engine for testing."""

    def __init__(self, version=0):
        self._version = version

    def get_version(self):
        return self._version

    def set_version(self, version):
        self._version = version


@pytest.fixture
def config_with_sdp():
    """Config with segment-wise PPO enabled."""
    return InferenceEngineConfig(enable_sdp=True)


@pytest.fixture
def config_without_sdp():
    """Config without segment-wise PPO (default behavior)."""
    return InferenceEngineConfig(enable_sdp=False)


@pytest.fixture
def mock_engine():
    """Mock inference engine."""
    return MockInferenceEngine(version=0)


class TestFactoryUsageEnforcement:
    """Test that factory usage is properly enforced when SDP is enabled."""

    def test_direct_constructor_with_sdp_triggers_warning(self, config_with_sdp, mock_engine):
        """
        CRITICAL TEST: Direct WorkflowExecutor() usage with enable_segment_wise_ppo=True
        should trigger a UserWarning indicating factory should be used.
        """
        with pytest.warns(UserWarning, match="WorkflowExecutor created with enable_segment_wise_ppo=True"):
            executor = WorkflowExecutor(
                config=config_with_sdp,
                inference_engine=mock_engine,
            )

        # Verify executor was created (warning doesn't prevent creation)
        assert executor is not None
        assert executor.config == config_with_sdp

    def test_warning_message_mentions_factory(self, config_with_sdp, mock_engine):
        """Test that warning message suggests using the factory."""
        with pytest.warns(UserWarning, match="Please use create_workflow_executor"):
            WorkflowExecutor(
                config=config_with_sdp,
                inference_engine=mock_engine,
            )

    def test_warning_message_mentions_feature_may_not_work(self, config_with_sdp, mock_engine):
        """Test that warning explains consequences of misconfiguration."""
        with pytest.warns(UserWarning, match="Segment-wise PPO features may not work correctly"):
            WorkflowExecutor(
                config=config_with_sdp,
                inference_engine=mock_engine,
            )

    def test_factory_usage_with_sdp_no_warning(self, config_with_sdp, mock_engine):
        """
        CRITICAL TEST: Using create_workflow_executor() with enable_segment_wise_ppo=True
        should NOT trigger any warning (proper components injected).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors

            # This should NOT raise (no warning should be emitted)
            executor = create_workflow_executor(
                config=config_with_sdp,
                inference_engine=mock_engine,
            )

        # Verify executor was created properly
        assert executor is not None
        assert executor.config == config_with_sdp
        assert executor.staleness_strategy is not None  # Components injected
        assert executor.proximal_recomputer is not None

    def test_direct_constructor_without_sdp_no_warning(self, config_without_sdp, mock_engine):
        """
        BACKWARD COMPATIBILITY TEST: Direct constructor without enable_segment_wise_ppo
        should work fine without warnings (default AReaL behavior).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors

            # This should NOT raise (backward compatible)
            executor = WorkflowExecutor(
                config=config_without_sdp,
                inference_engine=mock_engine,
            )

        # Verify executor works in standard mode
        assert executor is not None
        assert executor.config == config_without_sdp
        # In standard mode, these may be None (not needed)


class TestFactoryComponentInjection:
    """Test that factory properly injects required components."""

    def test_factory_creates_staleness_strategy(self, config_with_sdp, mock_engine):
        """Test that factory injects StalenessControlStrategy when SDP enabled."""
        executor = create_workflow_executor(config_with_sdp, mock_engine)

        assert executor.staleness_strategy is not None
        # Verify it's the correct strategy for SDP
        from areal.api.staleness_control import SegmentWisePPOStrategy
        assert isinstance(executor.staleness_strategy, SegmentWisePPOStrategy)

    def test_factory_creates_proximal_recomputer(self, config_with_sdp, mock_engine):
        """Test that factory injects ProximalRecomputer when SDP enabled."""
        executor = create_workflow_executor(config_with_sdp, mock_engine)

        assert executor.proximal_recomputer is not None
        # Verify it's properly configured
        from areal.api.proximal_recomputer import ProximalRecomputer
        assert isinstance(executor.proximal_recomputer, ProximalRecomputer)

    def test_factory_without_sdp_uses_standard_strategy(self, config_without_sdp, mock_engine):
        """Test that factory respects backward compatibility (standard PPO strategy when SDP disabled)."""
        executor = create_workflow_executor(config_without_sdp, mock_engine)

        # Without SDP, should use StandardPPOStrategy (backward compatible)
        from areal.api.staleness_control import StandardPPOStrategy
        assert isinstance(executor.staleness_strategy, StandardPPOStrategy)
        # Proximal recomputer should be None (not needed for standard PPO)
        assert executor.proximal_recomputer is None


class TestRegressionPrevention:
    """
    Tests to prevent regression where client code might be changed back
    to direct constructor usage instead of factory.
    """

    def test_client_code_pattern_detection(self, config_with_sdp, mock_engine):
        """
        Test that simulates client code mistakenly using direct constructor.
        This pattern should trigger warning.
        """
        # Simulate what would happen if someone changed engine code back:
        # self.workflow_executor = WorkflowExecutor(config, engine)

        with pytest.warns(UserWarning, match="factory was not used"):
            # This is the WRONG pattern
            workflow_executor = WorkflowExecutor(
                config=config_with_sdp,
                inference_engine=mock_engine,
            )

        # Feature won't work properly - gets default strategy instead of SDP strategy
        from areal.api.staleness_control import StandardPPOStrategy, SegmentWisePPOStrategy
        # Constructor creates StandardPPOStrategy by default (not the SDP strategy needed)
        assert isinstance(workflow_executor.staleness_strategy, StandardPPOStrategy)
        assert not isinstance(workflow_executor.staleness_strategy, SegmentWisePPOStrategy)
        assert workflow_executor.proximal_recomputer is None

    def test_correct_client_pattern_no_warning(self, config_with_sdp, mock_engine):
        """
        Test the CORRECT pattern that client code should use.
        This should NOT trigger warning.
        """
        # Simulate correct client code:
        # from areal.api.workflow_api import create_workflow_executor
        # self.workflow_executor = create_workflow_executor(config, engine)

        with warnings.catch_warnings():
            warnings.simplefilter("error")

            # This is the CORRECT pattern
            workflow_executor = create_workflow_executor(
                config=config_with_sdp,
                inference_engine=mock_engine,
            )

        # Feature will work properly
        assert workflow_executor.staleness_strategy is not None
        assert workflow_executor.proximal_recomputer is not None


class TestGuardrailEdgeCases:
    """Test edge cases for the guardrail mechanism."""

    def test_config_without_enable_sdp_attribute(self, mock_engine):
        """Test handling of config that doesn't have enable_segment_wise_ppo attribute."""
        # Create config without the attribute
        config = type('Config', (), {
            'max_concurrent_rollouts': 4,
            'consumer_batch_size': 2,
            'queue_size': 16,
            'max_head_offpolicyness': 2,
            'enable_rollout_tracing': False,
            'request_timeout': 30,
        })()

        # Should not crash, should treat as False (backward compatible)
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            executor = WorkflowExecutor(
                config=config,
                inference_engine=mock_engine,
            )

        assert executor is not None

    def test_partial_component_injection_triggers_warning(self, config_with_sdp, mock_engine):
        """
        Test that even partial component injection (missing one) triggers warning.
        This catches cases where someone might manually inject only some components.
        """
        from areal.api.staleness_control import SegmentWisePPOStrategy

        # Manually inject only staleness_strategy (missing proximal_recomputer)
        with pytest.warns(UserWarning, match="without proper component injection"):
            executor = WorkflowExecutor(
                config=config_with_sdp,
                inference_engine=mock_engine,
                staleness_strategy=SegmentWisePPOStrategy(config=config_with_sdp),
                # Missing proximal_recomputer!
            )

        assert executor.staleness_strategy is not None
        assert executor.proximal_recomputer is None  # Missing!


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
