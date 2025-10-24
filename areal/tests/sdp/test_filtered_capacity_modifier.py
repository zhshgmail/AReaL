"""Unit tests for FilteredSamplesCapacityModifier.

This module provides comprehensive test coverage for FilteredSamplesCapacityModifier,
which tracks filtered samples for capacity calculation in segment-wise PPO.
All tests run without GPU.
"""

import pytest
from unittest.mock import Mock

from areal.core.filtered_capacity_modifier import FilteredSamplesCapacityModifier


class TestFilteredSamplesCapacityModifierInitialization:
    """Test FilteredSamplesCapacityModifier initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        modifier = FilteredSamplesCapacityModifier()

        assert modifier.get_filtered_count() == 0

    def test_initial_state(self):
        """Test that initial filtered count is zero."""
        modifier = FilteredSamplesCapacityModifier()

        # modify_capacity should return base_capacity when no samples filtered
        mock_stats = Mock()
        adjustment = modifier.modify_capacity(100, 0, mock_stats)
        assert adjustment == 100  # No adjustment yet


class TestOnSampleFiltered:
    """Test on_samples_filtered method."""

    def test_single_sample_filtered(self):
        """Test filtering a single sample."""
        modifier = FilteredSamplesCapacityModifier()
        mock_stats = Mock()

        modifier.on_samples_filtered(1)

        assert modifier.get_filtered_count() == 1
        assert modifier.modify_capacity(0, 0, mock_stats) == 1

    def test_multiple_samples_filtered(self):
        """Test filtering multiple samples."""
        modifier = FilteredSamplesCapacityModifier()
        mock_stats = Mock()

        modifier.on_samples_filtered(5)

        assert modifier.get_filtered_count() == 5
        assert modifier.modify_capacity(0, 0, mock_stats) == 5

    def test_filtered_count_accumulates(self):
        """Test that filtered count accumulates correctly."""
        modifier = FilteredSamplesCapacityModifier()

        modifier.on_samples_filtered(1)
        assert modifier.get_filtered_count() == 1

        modifier.on_samples_filtered(1)
        assert modifier.get_filtered_count() == 2

        modifier.on_samples_filtered(1)
        assert modifier.get_filtered_count() == 3


class TestReset:
    """Test reset method."""

    def test_reset_clears_count(self):
        """Test that reset clears the filtered count."""
        modifier = FilteredSamplesCapacityModifier()
        mock_stats = Mock()

        # Filter some samples
        modifier.on_samples_filtered(10)

        assert modifier.get_filtered_count() == 10

        # Reset
        modifier.reset()

        assert modifier.get_filtered_count() == 0
        assert modifier.modify_capacity(0, 0, mock_stats) == 0

    def test_reset_on_fresh_modifier(self):
        """Test reset on newly created modifier."""
        modifier = FilteredSamplesCapacityModifier()
        mock_stats = Mock()

        # Reset without filtering
        modifier.reset()

        assert modifier.get_filtered_count() == 0
        assert modifier.modify_capacity(0, 0, mock_stats) == 0

    def test_multiple_resets(self):
        """Test multiple consecutive resets."""
        modifier = FilteredSamplesCapacityModifier()

        modifier.on_samples_filtered(1)
        modifier.reset()
        assert modifier.get_filtered_count() == 0

        modifier.reset()
        assert modifier.get_filtered_count() == 0

        modifier.reset()
        assert modifier.get_filtered_count() == 0


class TestGetAdjustment:
    """Test get_adjustment method."""

    def test_adjustment_equals_filtered_count(self):
        """Test that adjustment equals filtered count."""
        modifier = FilteredSamplesCapacityModifier()

        for i in range(1, 11):
            modifier.on_samples_filtered(1)
            assert modifier.get_filtered_count() == i

    def test_adjustment_after_reset(self):
        """Test adjustment after reset."""
        modifier = FilteredSamplesCapacityModifier()

        # Filter and reset
        modifier.on_samples_filtered(1)
        modifier.on_samples_filtered(1)
        modifier.reset()

        assert modifier.get_filtered_count() == 0

    def test_adjustment_is_readonly(self):
        """Test that get_adjustment doesn't modify state."""
        modifier = FilteredSamplesCapacityModifier()

        modifier.on_samples_filtered(1)
        modifier.on_samples_filtered(1)

        # Multiple calls should return same value
        adj1 = modifier.get_filtered_count()
        adj2 = modifier.get_filtered_count()
        adj3 = modifier.get_filtered_count()

        assert adj1 == adj2 == adj3 == 2
        assert modifier.get_filtered_count() == 2


class TestFilterResetCycle:
    """Test typical filter-reset cycles."""

    def test_single_cycle(self):
        """Test a single filter-reset cycle."""
        modifier = FilteredSamplesCapacityModifier()

        # Filter phase
        for _ in range(5):
            modifier.on_samples_filtered(1)

        assert modifier.get_filtered_count() == 5

        # Reset phase
        modifier.reset()

        assert modifier.get_filtered_count() == 0

    def test_multiple_cycles(self):
        """Test multiple filter-reset cycles."""
        modifier = FilteredSamplesCapacityModifier()

        # Cycle 1
        for _ in range(3):
            modifier.on_samples_filtered(1)
        assert modifier.get_filtered_count() == 3
        modifier.reset()
        assert modifier.get_filtered_count() == 0

        # Cycle 2
        for _ in range(7):
            modifier.on_samples_filtered(1)
        assert modifier.get_filtered_count() == 7
        modifier.reset()
        assert modifier.get_filtered_count() == 0

        # Cycle 3
        for _ in range(2):
            modifier.on_samples_filtered(1)
        assert modifier.get_filtered_count() == 2

    def test_varying_filter_counts_per_cycle(self):
        """Test cycles with varying filtered counts."""
        modifier = FilteredSamplesCapacityModifier()

        test_counts = [1, 5, 0, 10, 3, 0, 8]

        for count in test_counts:
            # Filter
            for _ in range(count):
                modifier.on_samples_filtered(1)

            # Verify
            assert modifier.get_filtered_count() == count

            # Reset for next cycle
            modifier.reset()
            assert modifier.get_filtered_count() == 0


class TestThreadSafety:
    """Test thread safety considerations."""

    def test_concurrent_filtering(self):
        """Test concurrent filtering (simplified - real threading tested elsewhere)."""
        import threading

        modifier = FilteredSamplesCapacityModifier()

        def filter_many():
            for _ in range(100):
                modifier.on_samples_filtered(1)

        threads = [threading.Thread(target=filter_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 500 total (5 threads * 100 each)
        # Note: Without explicit locking, this might fail in real concurrent scenarios
        # This test documents expected behavior, not guaranteed thread-safety
        assert modifier.get_filtered_count() == 500


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_filtered_count(self):
        """Test with very large filtered count."""
        modifier = FilteredSamplesCapacityModifier()

        # Filter 10000 samples
        for _ in range(10000):
            modifier.on_samples_filtered(1)

        assert modifier.get_filtered_count() == 10000
        assert modifier.get_filtered_count() == 10000

    def test_reset_without_filtering(self):
        """Test reset when nothing was filtered."""
        modifier = FilteredSamplesCapacityModifier()

        modifier.reset()

        assert modifier.get_filtered_count() == 0
        assert modifier.get_filtered_count() == 0

    def test_filter_after_multiple_resets(self):
        """Test filtering after multiple resets."""
        modifier = FilteredSamplesCapacityModifier()

        # Multiple resets
        for _ in range(5):
            modifier.reset()

        # Then filter
        modifier.on_samples_filtered(1)
        modifier.on_samples_filtered(1)

        assert modifier.get_filtered_count() == 2
        assert modifier.get_filtered_count() == 2


class TestCapacityModifierInterface:
    """Test CapacityModifier interface compliance."""

    def test_implements_required_methods(self):
        """Test that all required methods are implemented."""
        modifier = FilteredSamplesCapacityModifier()

        # Should have modify_capacity method (from CapacityModifier interface)
        assert hasattr(modifier, 'modify_capacity')
        assert callable(modifier.modify_capacity)

        # Should have helper methods
        assert hasattr(modifier, 'get_filtered_count')
        assert callable(modifier.get_filtered_count)

        assert hasattr(modifier, 'on_samples_filtered')
        assert callable(modifier.on_samples_filtered)

        # Should have reset method
        assert hasattr(modifier, 'reset')
        assert callable(modifier.reset)

    def test_get_filtered_count_returns_int(self):
        """Test that get_filtered_count returns an integer."""
        modifier = FilteredSamplesCapacityModifier()

        modifier.on_samples_filtered(1)
        count = modifier.get_filtered_count()

        assert isinstance(count, int)

    def test_methods_are_chainable_where_appropriate(self):
        """Test method return values for potential chaining."""
        modifier = FilteredSamplesCapacityModifier()
        mock_stats = Mock()

        # on_samples_filtered returns None (not chainable)
        result = modifier.on_samples_filtered(1)
        assert result is None

        # reset returns None (not chainable)
        result = modifier.reset()
        assert result is None

        # get_filtered_count returns int (informational, not chainable)
        result = modifier.get_filtered_count()
        assert isinstance(result, int)

        # modify_capacity returns int
        result = modifier.modify_capacity(100, 0, mock_stats)
        assert isinstance(result, int)


# Parametrized tests
@pytest.mark.parametrize("filter_count", [0, 1, 5, 10, 50, 100, 1000])
def test_parametrized_filter_counts(filter_count):
    """Test with various filter counts."""
    modifier = FilteredSamplesCapacityModifier()

    for _ in range(filter_count):
        modifier.on_samples_filtered(1)

    assert modifier.get_filtered_count() == filter_count
    assert modifier.get_filtered_count() == filter_count


@pytest.mark.parametrize("num_cycles", [1, 3, 5, 10])
def test_parametrized_cycles(num_cycles):
    """Test multiple filter-reset cycles."""
    modifier = FilteredSamplesCapacityModifier()

    for cycle in range(num_cycles):
        # Filter some samples
        for _ in range(5):
            modifier.on_samples_filtered(1)

        assert modifier.get_filtered_count() == 5

        # Reset
        modifier.reset()

        assert modifier.get_filtered_count() == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=areal.core.filtered_capacity_modifier"])
