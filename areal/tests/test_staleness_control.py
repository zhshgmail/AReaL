"""Unit tests for staleness control strategies.

This module provides comprehensive test coverage for StalenessControlStrategy,
SegmentWisePPOStrategy, and StandardPPOStrategy classes.
All tests run without GPU for CI pipeline compatibility.
"""

import pytest

from areal.api.cli_args import InferenceEngineConfig
from areal.api.staleness_control import (
    SegmentWisePPOStrategy,
    StalenessControlStrategy,
    StandardPPOStrategy,
)


class TestStalenessControlStrategyInterface:
    """Test the abstract StalenessControlStrategy interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that StalenessControlStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StalenessControlStrategy(None)


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

    def test_should_filter_before_enqueue_returns_true(self):
        """Test that StandardPPOStrategy always filters before enqueue."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=False,
        )
        strategy = StandardPPOStrategy(config)

        # Should always return True for standard PPO
        assert strategy.should_filter_before_enqueue() is True

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
        trajectory = {"output_version": [5, 5, 5]}
        is_stale = strategy.is_sample_too_stale(trajectory, current_version=5, config=config)

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
        trajectory = {"output_version": [3, 3, 3]}
        is_stale = strategy.is_sample_too_stale(trajectory, current_version=6, config=config)

        assert is_stale is True  # Too stale

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
        trajectory = {"output_version": [3, 3, 3]}
        is_stale = strategy.is_sample_too_stale(trajectory, current_version=5, config=config)

        assert is_stale is False  # At boundary, still acceptable

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
        trajectory = {"output_version": [3, 4, 5]}  # Head is version 3
        is_stale = strategy.is_sample_too_stale(trajectory, current_version=6, config=config)

        assert is_stale is True  # Head is too stale (6 - 3 = 3 > 2)

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
        trajectory = {"output_version": [4, 4, 4]}
        is_stale = strategy.is_sample_too_stale(trajectory, current_version=5, config=config)

        assert is_stale is True  # Any staleness is too much


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

    def test_should_filter_before_enqueue_returns_false(self):
        """Test that SegmentWisePPOStrategy defers filtering."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            enable_segment_wise_ppo=True,
        )
        strategy = SegmentWisePPOStrategy(config)

        # Should return False - filtering happens after recomputation
        assert strategy.should_filter_before_enqueue() is False

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
        trajectory = {"output_version": [5, 5, 5]}
        is_stale = strategy.is_sample_too_stale(trajectory, current_version=5, config=config)

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
        trajectory = {"output_version": [4, 4, 4]}
        is_stale = strategy.is_sample_too_stale(trajectory, current_version=5, config=config)

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
        trajectory = {"output_version": [3, 3, 3]}
        is_stale = strategy.is_sample_too_stale(trajectory, current_version=5, config=config)

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
        trajectory = {"output_version": [1, 1, 1]}
        is_stale = strategy.is_sample_too_stale(trajectory, current_version=5, config=config)

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
        trajectory = {"output_version": [4, 5, 5]}
        is_stale = strategy.is_sample_too_stale(trajectory, current_version=5, config=config)

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

        # Standard filters before enqueue
        assert standard.should_filter_before_enqueue() is True
        # Segment-wise filters after recomputation
        assert segment_wise.should_filter_before_enqueue() is False

    def test_v_minus_1_handling_difference(self):
        """Test that v-1 samples are handled differently."""
        config = InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            max_head_offpolicyness=2,
        )

        trajectory = {"output_version": [4, 4, 4]}
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

        trajectory = {"output_version": [3, 3, 3]}
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
    (5, 5, 2, False),  # Same version - not stale
    (5, 4, 2, False),  # v-1, within limit
    (5, 3, 2, False),  # v-2, at boundary
    (5, 2, 2, True),   # v-3, beyond limit
    (10, 9, 0, True),  # v-1 but max=0
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

    trajectory = {"output_version": [sample_version]}
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

    trajectory = {"output_version": [sample_version]}
    is_stale = strategy.is_sample_too_stale(trajectory, current_version, config)

    assert is_stale == expected


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=areal.api.staleness_control"])
