"""Unit tests for StalenessManager capacity modifier extension.

This module provides comprehensive test coverage for the capacity modifier
extension point added to StalenessManager for segment-wise PPO.
All tests run without GPU.
"""

from unittest.mock import Mock

import pytest

from areal.core.staleness_manager import StalenessManager


class TestCapacityModifierRegistration:
    """Test capacity modifier registration."""

    def test_register_single_modifier(self):
        """Test registering a single capacity modifier."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        mock_modifier = Mock()
        manager.register_capacity_modifier(mock_modifier)

        assert len(manager.capacity_modifiers) == 1
        assert manager.capacity_modifiers[0] is mock_modifier

    def test_register_multiple_modifiers(self):
        """Test registering multiple capacity modifiers."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        modifier1 = Mock()
        modifier2 = Mock()
        modifier3 = Mock()

        manager.register_capacity_modifier(modifier1)
        manager.register_capacity_modifier(modifier2)
        manager.register_capacity_modifier(modifier3)

        assert len(manager.capacity_modifiers) == 3
        assert manager.capacity_modifiers[0] is modifier1
        assert manager.capacity_modifiers[1] is modifier2
        assert manager.capacity_modifiers[2] is modifier3

    def test_modifiers_list_starts_empty(self):
        """Test that modifiers list starts empty."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        assert manager.capacity_modifiers == []


class TestCapacityModifierApplication:
    """Test that capacity modifiers are applied during get_capacity."""

    def test_modifier_is_called_during_get_capacity(self):
        """Test that registered modifier is called."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        mock_modifier = Mock()
        mock_modifier.modify_capacity.return_value = 5  # Modified capacity

        manager.register_capacity_modifier(mock_modifier)

        capacity = manager.get_capacity(current_version=0)

        # Modifier should have been called
        mock_modifier.modify_capacity.assert_called_once()

    def test_modifier_receives_correct_arguments(self):
        """Test that modifier receives base capacity, version, and stats."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        mock_modifier = Mock()
        mock_modifier.modify_capacity.return_value = 8

        manager.register_capacity_modifier(mock_modifier)

        # Base capacity at v0: min(10, (2+0+1)*4) = min(10, 12) = 10
        capacity = manager.get_capacity(current_version=0)

        # Check call arguments
        call_args = mock_modifier.modify_capacity.call_args
        base_capacity = call_args[0][0]
        current_version = call_args[0][1]
        stats = call_args[0][2]

        assert base_capacity == 10  # Base capacity
        assert current_version == 0
        assert hasattr(stats, 'submitted')
        assert hasattr(stats, 'running')
        assert hasattr(stats, 'accepted')

    def test_modifier_can_increase_capacity(self):
        """Test that modifier can increase capacity."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Modifier that adds 5 to capacity
        mock_modifier = Mock()
        mock_modifier.modify_capacity.side_effect = lambda cap, ver, stats: cap + 5

        manager.register_capacity_modifier(mock_modifier)

        # Base capacity: 10
        # Modified: 10 + 5 = 15
        capacity = manager.get_capacity(current_version=0)

        assert capacity == 15

    def test_modifier_can_decrease_capacity(self):
        """Test that modifier can decrease capacity."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Modifier that subtracts 3 from capacity
        mock_modifier = Mock()
        mock_modifier.modify_capacity.side_effect = lambda cap, ver, stats: cap - 3

        manager.register_capacity_modifier(mock_modifier)

        # Base capacity: 10
        # Modified: 10 - 3 = 7
        capacity = manager.get_capacity(current_version=0)

        assert capacity == 7

    def test_modifier_can_return_zero_capacity(self):
        """Test that modifier can reduce capacity to zero."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Modifier that sets capacity to 0
        mock_modifier = Mock()
        mock_modifier.modify_capacity.return_value = 0

        manager.register_capacity_modifier(mock_modifier)

        capacity = manager.get_capacity(current_version=0)

        assert capacity == 0

    def test_modifier_can_return_negative_capacity(self):
        """Test that modifier can return negative capacity."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Modifier that returns negative
        mock_modifier = Mock()
        mock_modifier.modify_capacity.return_value = -5

        manager.register_capacity_modifier(mock_modifier)

        capacity = manager.get_capacity(current_version=0)

        assert capacity == -5


class TestMultipleModifierApplication:
    """Test application of multiple modifiers in sequence."""

    def test_modifiers_applied_in_registration_order(self):
        """Test that modifiers are applied in registration order."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Modifier 1: add 5
        modifier1 = Mock()
        modifier1.modify_capacity.side_effect = lambda cap, ver, stats: cap + 5

        # Modifier 2: multiply by 2
        modifier2 = Mock()
        modifier2.modify_capacity.side_effect = lambda cap, ver, stats: cap * 2

        manager.register_capacity_modifier(modifier1)
        manager.register_capacity_modifier(modifier2)

        # Base: 10
        # After modifier1: 10 + 5 = 15
        # After modifier2: 15 * 2 = 30
        capacity = manager.get_capacity(current_version=0)

        assert capacity == 30

    def test_order_matters(self):
        """Test that registration order affects final result."""
        # Test case 1: add then multiply
        manager1 = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        mod1a = Mock()
        mod1a.modify_capacity.side_effect = lambda cap, ver, stats: cap + 5
        mod1b = Mock()
        mod1b.modify_capacity.side_effect = lambda cap, ver, stats: cap * 2

        manager1.register_capacity_modifier(mod1a)
        manager1.register_capacity_modifier(mod1b)

        capacity1 = manager1.get_capacity(current_version=0)
        # (10 + 5) * 2 = 30

        # Test case 2: multiply then add
        manager2 = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        mod2a = Mock()
        mod2a.modify_capacity.side_effect = lambda cap, ver, stats: cap * 2
        mod2b = Mock()
        mod2b.modify_capacity.side_effect = lambda cap, ver, stats: cap + 5

        manager2.register_capacity_modifier(mod2a)
        manager2.register_capacity_modifier(mod2b)

        capacity2 = manager2.get_capacity(current_version=0)
        # (10 * 2) + 5 = 25

        assert capacity1 == 30
        assert capacity2 == 25
        assert capacity1 != capacity2  # Order matters!

    def test_three_modifiers_in_sequence(self):
        """Test three modifiers applied in sequence."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        mod1 = Mock()
        mod1.modify_capacity.side_effect = lambda cap, ver, stats: cap + 5

        mod2 = Mock()
        mod2.modify_capacity.side_effect = lambda cap, ver, stats: cap - 3

        mod3 = Mock()
        mod3.modify_capacity.side_effect = lambda cap, ver, stats: cap * 2

        manager.register_capacity_modifier(mod1)
        manager.register_capacity_modifier(mod2)
        manager.register_capacity_modifier(mod3)

        # Base: 10
        # After mod1: 10 + 5 = 15
        # After mod2: 15 - 3 = 12
        # After mod3: 12 * 2 = 24
        capacity = manager.get_capacity(current_version=0)

        assert capacity == 24


class TestModifierWithStatefulManager:
    """Test modifiers with stateful manager (rollouts submitted/accepted)."""

    def test_modifier_sees_updated_stats(self):
        """Test that modifier sees current stats."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Track what stats the modifier sees
        stats_seen = []

        def capture_stats(cap, ver, stats):
            stats_seen.append((stats.submitted, stats.running, stats.accepted))
            return cap

        mock_modifier = Mock()
        mock_modifier.modify_capacity.side_effect = capture_stats

        manager.register_capacity_modifier(mock_modifier)

        # Submit some rollouts
        for _ in range(3):
            manager.on_rollout_submitted()

        # Get capacity - modifier should see current stats
        manager.get_capacity(current_version=0)

        # Check that modifier saw the stats
        assert len(stats_seen) == 1
        assert stats_seen[0] == (3, 3, 0)  # 3 submitted, 3 running, 0 accepted

    def test_modifier_with_different_manager_states(self):
        """Test modifier behavior with different manager states."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Modifier that subtracts accepted count from capacity
        mock_modifier = Mock()
        mock_modifier.modify_capacity.side_effect = lambda cap, ver, stats: cap - stats.accepted

        manager.register_capacity_modifier(mock_modifier)

        # State 1: No rollouts
        capacity1 = manager.get_capacity(current_version=0)
        assert capacity1 == 10  # 10 - 0 = 10

        # State 2: 5 accepted rollouts
        for _ in range(5):
            manager.on_rollout_submitted()
            manager.on_rollout_accepted()

        capacity2 = manager.get_capacity(current_version=0)
        assert capacity2 == 5  # 10 - 5 = 5


class TestBackwardCompatibility:
    """Test that modifiers don't break existing functionality."""

    def test_without_modifiers_behavior_unchanged(self):
        """Test that manager works same without modifiers."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Don't register any modifiers
        capacity = manager.get_capacity(current_version=0)

        # Should work as before
        assert capacity == 10  # min(10, 12)

    def test_identity_modifier_no_change(self):
        """Test that identity modifier doesn't change capacity."""
        manager = StalenessManager(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Identity modifier
        mock_modifier = Mock()
        mock_modifier.modify_capacity.side_effect = lambda cap, ver, stats: cap

        manager.register_capacity_modifier(mock_modifier)

        capacity = manager.get_capacity(current_version=0)

        assert capacity == 10  # Unchanged


# Parametrized tests
@pytest.mark.parametrize("num_modifiers", [1, 2, 3, 5, 10])
def test_parametrized_multiple_modifiers(num_modifiers):
    """Test with varying numbers of modifiers."""
    manager = StalenessManager(
        max_concurrent_rollouts=10,
        consumer_batch_size=4,
        max_staleness=2,
    )

    # Each modifier adds 1
    for _ in range(num_modifiers):
        modifier = Mock()
        modifier.modify_capacity.side_effect = lambda cap, ver, stats: cap + 1
        manager.register_capacity_modifier(modifier)

    capacity = manager.get_capacity(current_version=0)

    # Base: 10, plus num_modifiers
    assert capacity == 10 + num_modifiers


@pytest.mark.parametrize("adjustment", [-5, -1, 0, 1, 5, 10])
def test_parametrized_capacity_adjustments(adjustment):
    """Test modifiers with various adjustment values."""
    manager = StalenessManager(
        max_concurrent_rollouts=10,
        consumer_batch_size=4,
        max_staleness=2,
    )

    mock_modifier = Mock()
    mock_modifier.modify_capacity.side_effect = lambda cap, ver, stats: cap + adjustment

    manager.register_capacity_modifier(mock_modifier)

    capacity = manager.get_capacity(current_version=0)

    assert capacity == 10 + adjustment


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=areal.core.staleness_manager"])
