"""Unit tests for staleness control strategies.

This module provides comprehensive test coverage for StalenessControlStrategy,
SegmentWisePPOStrategy, and StandardPPOStrategy classes.
All tests run without GPU for CI pipeline compatibility.
"""

import pytest
import torch
from tensordict import TensorDict

from areal.api.cli_args import InferenceEngineConfig
from areal.api.staleness_control import (
    SegmentWisePPOStrategy,
    StalenessControlStrategy,
    StandardPPOStrategy,
)


def create_test_tensordict(output_version):
    """Create a TensorDict from simple test format.

    Args:
        output_version: List of version numbers (one per output token)

    Returns:
        TensorDict with versions and loss_mask keys
    """
    # Convert output_version list to versions tensor
    # Assume all tokens are output tokens (loss_mask = 1)
    versions = torch.tensor([output_version], dtype=torch.int64)  # Shape: [1, seq_len]
    loss_mask = torch.ones_like(versions)  # All tokens are output tokens

    return TensorDict({
        "versions": versions,
        "loss_mask": loss_mask,
    }, batch_size=[1])


class TestStalenessControlStrategyInterface:
    """Test the StalenessControlStrategy base class."""

    def test_base_strategy_can_be_instantiated(self):
        """Test that StalenessControlStrategy can be instantiated (it's not abstract).

        The base strategy provides default behavior (no filtering), which is valid.
        """
        strategy = StalenessControlStrategy(None)
        assert strategy is not None
        assert strategy.config is None


class TestStandardPPOStrategy:
    """Test StandardPPOStrategy implementation."""

    def test_initialization(self):
        """Test StandardPPOStrategy initialization."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
            max_head_offpolicyness=5,
        )

        strategy = StandardPPOStrategy(config)

        assert strategy.config is config
        assert isinstance(strategy, StalenessControlStrategy)

    def test_should_filter_before_enqueue_returns_false(self):
        """Test that StandardPPOStrategy does not filter before enqueue."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
        )
        strategy = StandardPPOStrategy(config)

        # Should return False - standard PPO has no staleness control
        assert strategy.should_filter_before_enqueue() is False

    def test_is_sample_too_stale_with_fresh_sample(self):
        """Test staleness check for fresh samples."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
            max_head_offpolicyness=2,
        )
        strategy = StandardPPOStrategy(config)

        # Sample at version 5, current version 5 (staleness = 0)
        trajectory = create_test_tensordict([5, 5, 5])
        is_stale = strategy.is_sample_too_stale(trajectory,5, config=config)

        assert is_stale is False  # Not stale

    def test_is_sample_too_stale_with_stale_sample(self):
        """Test staleness check for stale samples."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
            max_head_offpolicyness=2,
        )
        strategy = StandardPPOStrategy(config)

        # Sample at version 3, current version 6 (staleness = 3 > max 2)
        trajectory = create_test_tensordict([3, 3, 3])
        is_stale = strategy.is_sample_too_stale(trajectory,6, config=config)

        # StandardPPOStrategy doesn't check staleness - always returns False
        assert is_stale is False

    def test_is_sample_too_stale_at_boundary(self):
        """Test staleness check at exact boundary."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
            max_head_offpolicyness=2,
        )
        strategy = StandardPPOStrategy(config)

        # Sample at version 3, current version 5 (staleness = 2 == max 2)
        trajectory = create_test_tensordict([3, 3, 3])
        is_stale = strategy.is_sample_too_stale(trajectory,5, config=config)

        # StandardPPOStrategy doesn't check staleness - always returns False
        assert is_stale is False

    def test_is_sample_too_stale_with_mixed_versions(self):
        """Test staleness check with tokens from different versions."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
            max_head_offpolicyness=2,
        )
        strategy = StandardPPOStrategy(config)

        # Mixed versions: head token determines staleness
        trajectory = create_test_tensordict([3, 4, 5])  # Head is version 3
        is_stale = strategy.is_sample_too_stale(trajectory,6, config=config)

        # StandardPPOStrategy doesn't check staleness - always returns False
        assert is_stale is False

    def test_is_sample_too_stale_with_zero_max_offpolicyness(self):
        """Test staleness check with zero max offpolicyness."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
            max_head_offpolicyness=0,
        )
        strategy = StandardPPOStrategy(config)

        # Sample at version 4, current version 5
        trajectory = create_test_tensordict([4, 4, 4])
        is_stale = strategy.is_sample_too_stale(trajectory,5, config=config)

        # StandardPPOStrategy doesn't check staleness - always returns False
        assert is_stale is False


class TestSegmentWisePPOStrategy:
    """Test SegmentWisePPOStrategy implementation."""

    def test_initialization(self):
        """Test SegmentWisePPOStrategy initialization."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
            max_head_offpolicyness=5,
        )

        strategy = SegmentWisePPOStrategy(config)

        assert strategy.config is config
        assert isinstance(strategy, StalenessControlStrategy)

    def test_should_filter_before_enqueue_returns_true(self):
        """Test that SegmentWisePPOStrategy filters before enqueue."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
        )
        strategy = SegmentWisePPOStrategy(config)

        # Should return True - aggressive filtering for segment-wise PPO
        assert strategy.should_filter_before_enqueue() is True

    def test_is_sample_too_stale_with_fresh_sample(self):
        """Test staleness check for fresh samples (v == current)."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
            max_head_offpolicyness=2,
        )
        strategy = SegmentWisePPOStrategy(config)

        # Sample at version 5, current version 5 (staleness = 0)
        trajectory = create_test_tensordict([5, 5, 5])
        is_stale = strategy.is_sample_too_stale(trajectory,5, config=config)

        assert is_stale is False  # Not stale

    def test_is_sample_too_stale_with_v_minus_1_sample(self):
        """Test that v-1 samples are NOT filtered (will be recomputed)."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
            max_head_offpolicyness=2,
        )
        strategy = SegmentWisePPOStrategy(config)

        # Sample at version 4, current version 5 (v-1)
        trajectory = create_test_tensordict([4, 4, 4])
        is_stale = strategy.is_sample_too_stale(trajectory,5, config=config)

        # v-1 samples are NOT too stale (will be recomputed)
        assert is_stale is False

    def test_is_sample_too_stale_with_v_minus_2_sample(self):
        """Test that v-2 samples ARE filtered."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
            max_head_offpolicyness=2,
        )
        strategy = SegmentWisePPOStrategy(config)

        # Sample at version 3, current version 5 (v-2)
        trajectory = create_test_tensordict([3, 3, 3])
        is_stale = strategy.is_sample_too_stale(trajectory,5, config=config)

        # v-2 is too stale (beyond recomputation capability)
        assert is_stale is True

    def test_is_sample_too_stale_respects_max_offpolicyness(self):
        """Test that max_head_offpolicyness is still respected."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
            max_head_offpolicyness=5,
        )
        strategy = SegmentWisePPOStrategy(config)

        # Sample at version 1, current version 5 (staleness = 4)
        trajectory = create_test_tensordict([1, 1, 1])
        is_stale = strategy.is_sample_too_stale(trajectory,5, config=config)

        # 4 < 5, so within max_offpolicyness, but v-4 is too far
        assert is_stale is True

    def test_is_sample_too_stale_with_mixed_versions(self):
        """Test staleness with mixed token versions."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
            max_head_offpolicyness=2,
        )
        strategy = SegmentWisePPOStrategy(config)

        # Mixed: head is v-1, tail is current
        trajectory = create_test_tensordict([4, 5, 5])
        is_stale = strategy.is_sample_too_stale(trajectory,5, config=config)

        # Head (v-1) is acceptable
        assert is_stale is False


class TestStrategyComparison:
    """Test differences between StandardPPO and SegmentWisePPO strategies."""

    def test_filtering_behavior_difference(self):
        """Test that strategies have different filtering behavior."""
        config_standard = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
        )
        config_segment = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
        )

        standard = StandardPPOStrategy(config_standard)
        segment_wise = SegmentWisePPOStrategy(config_segment)

        # Standard PPO does not filter (backward compatibility)
        assert standard.should_filter_before_enqueue() is False
        # Segment-wise PPO filters aggressively
        assert segment_wise.should_filter_before_enqueue() is True

    def test_v_minus_1_handling_difference(self):
        """Test that v-1 samples are handled differently."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            max_head_offpolicyness=2,
        )

        trajectory = create_test_tensordict([4, 4, 4])
        current_version = 5

        # Standard PPO: v-1 is NOT stale (within max_offpolicyness=2)
        standard = StandardPPOStrategy(config)
        standard_result = standard.is_sample_too_stale(trajectory, current_version, config)

        # Segment-wise PPO: v-1 is NOT stale (will be recomputed)
        segment_wise = SegmentWisePPOStrategy(config)
        segment_result = segment_wise.is_sample_too_stale(trajectory, current_version, config)

        # Both should accept v-1, but for different reasons
        assert standard_result is False
        assert segment_result is False

    def test_v_minus_2_handling_difference(self):
        """Test that v-2 samples are handled differently."""
        config_standard = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
            max_head_offpolicyness=5,  # Large enough
        )
        config_segment = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
            max_head_offpolicyness=5,
        )

        trajectory = create_test_tensordict([3, 3, 3])
        current_version = 5

        # Standard PPO: v-2 is ok (within max_offpolicyness=5)
        standard = StandardPPOStrategy(config_standard)
        standard_result = standard.is_sample_too_stale(trajectory, current_version, config_standard)

        # Segment-wise PPO: v-2 is too stale (beyond v-1)
        segment_wise = SegmentWisePPOStrategy(config_segment)
        segment_result = segment_wise.is_sample_too_stale(trajectory, current_version, config_segment)

        # Different results!
        assert standard_result is False  # Standard accepts v-2
        assert segment_result is True    # Segment-wise rejects v-2


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize("strategy_class", [StandardPPOStrategy, SegmentWisePPOStrategy])
def test_parametrized_initialization(strategy_class):
    """Test that both strategies can be initialized."""
    config = InferenceEngineConfig(
        experiment_name="test",
        trial_name="test",
    )

    strategy = strategy_class(config)

    assert strategy.config is config
    assert isinstance(strategy, StalenessControlStrategy)


@pytest.mark.parametrize("current_version,sample_version,max_offpolicyness,expected", [
    (5, 5, 2, False),  # Same version - StandardPPOStrategy doesn't check staleness
    (5, 4, 2, False),  # StandardPPOStrategy doesn't check staleness
    (5, 3, 2, False),  # StandardPPOStrategy doesn't check staleness
    (5, 2, 2, False),  # StandardPPOStrategy doesn't check staleness
    (10, 9, 0, False), # StandardPPOStrategy doesn't check staleness
])
def test_parametrized_standard_staleness(current_version, sample_version, max_offpolicyness, expected):
    """Test StandardPPOStrategy with various staleness scenarios."""
    config = InferenceEngineConfig(
        experiment_name="test",
        trial_name="test",
        enable_segment_wise_ppo=False,
        max_head_offpolicyness=max_offpolicyness,
    )
    strategy = StandardPPOStrategy(config)

    trajectory = create_test_tensordict([sample_version])
    is_stale = strategy.is_sample_too_stale(trajectory, current_version, config)

    assert is_stale == expected


@pytest.mark.parametrize("current_version,sample_version,expected", [
    (5, 5, False),  # Same version - not stale
    (5, 4, False),  # v-1 - not stale (will recompute)
    (5, 3, True),   # v-2 - too stale
    (5, 2, True),   # v-3 - too stale
    (10, 9, False), # v-1 - not stale
    (10, 8, True),  # v-2 - too stale
])
def test_parametrized_segment_wise_staleness(current_version, sample_version, expected):
    """Test SegmentWisePPOStrategy with various staleness scenarios."""
    config = InferenceEngineConfig(
        experiment_name="test",
        trial_name="test",
        enable_segment_wise_ppo=True,
        max_head_offpolicyness=10,  # Large enough to not interfere
    )
    strategy = SegmentWisePPOStrategy(config)

    trajectory = create_test_tensordict([sample_version])
    is_stale = strategy.is_sample_too_stale(trajectory, current_version, config)

    assert is_stale == expected


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=areal.api.staleness_control"])
