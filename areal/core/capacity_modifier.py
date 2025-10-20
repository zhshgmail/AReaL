"""Capacity modifier interface for extending StalenessManager.

This module provides the extension point for customizing capacity
calculation in StalenessManager.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from areal.api.io_struct import RolloutStat


class CapacityModifier(ABC):
    """Abstract interface for modifying capacity calculation.

    Capacity modifiers are registered with StalenessManager to allow
    custom logic for adjusting the calculated capacity. This is useful
    for features that filter samples (e.g., segment-wise PPO) and need
    to compensate for filtered samples in capacity calculation.

    Example:
        >>> class MyModifier(CapacityModifier):
        ...     def modify_capacity(self, base, version, stats):
        ...         return base + 10  # Always allow 10 more rollouts
        ...
        >>> manager = StalenessManager(...)
        >>> manager.register_capacity_modifier(MyModifier())
    """

    @abstractmethod
    def modify_capacity(
        self,
        base_capacity: int,
        current_version: int,
        stats: RolloutStat,
    ) -> int:
        """Modify the base capacity calculation.

        Args:
            base_capacity: Base capacity calculated by StalenessManager
            current_version: Current model version
            stats: Current rollout statistics (submitted, accepted, running)

        Returns:
            Modified capacity value

        Note:
            Modifiers are applied sequentially in registration order.
            Each modifier receives the output of the previous modifier.
        """
        pass
