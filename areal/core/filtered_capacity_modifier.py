"""Capacity modifier that compensates for filtered samples.

This module provides a CapacityModifier implementation that tracks samples
filtered by staleness control and adjusts capacity accordingly to prevent
deadlock in segment-wise PPO training.
"""

from __future__ import annotations

from threading import Lock

from areal.api.io_struct import RolloutStat
from areal.core.capacity_modifier import CapacityModifier


class FilteredSamplesCapacityModifier(CapacityModifier):
    """Compensates capacity for samples filtered by staleness control.

    Problem:
    When samples are filtered out (e.g., by SegmentWisePPOStrategy), they are
    counted in StalenessManager's `accepted` count but aren't actually available
    for training. This causes get_capacity() to become too restrictive, potentially
    leading to deadlock where no new rollouts can be generated.

    Solution:
    This modifier tracks the number of filtered samples and adds that count back
    to the capacity calculation, ensuring that filtered samples don't block new
    rollout generation.

    Thread Safety:
    All methods are thread-safe and can be called from multiple threads.

    Example:
        >>> modifier = FilteredSamplesCapacityModifier()
        >>> staleness_manager.register_capacity_modifier(modifier)
        >>> # When samples are filtered:
        >>> modifier.on_samples_filtered(5)
        >>> # Capacity will be increased by 5 to compensate
    """

    def __init__(self, logger=None):
        """Initialize the modifier with zero filtered count.

        Args:
            logger: Logger instance for debugging (optional)
        """
        self.lock = Lock()
        self._filtered_count = 0
        self._version_filtered_count: dict[int, int] = {}
        self.logger = logger

    def on_samples_filtered(self, count: int, version: int | None = None) -> None:
        """Record that samples were filtered.

        This should be called by the staleness control strategy whenever samples
        are dropped due to staleness.

        Args:
            count: Number of samples filtered
            version: Model version when filtering occurred (optional, for tracking)
        """
        if count <= 0:
            return

        with self.lock:
            old_count = self._filtered_count
            self._filtered_count += count
            if version is not None:
                self._version_filtered_count[version] = (
                    self._version_filtered_count.get(version, 0) + count
                )

            # Debug logging
            if self.logger:
                self.logger.debug(
                    f"[CapacityModifier] Filtered samples count updated: "
                    f"added={count}, total={self._filtered_count} (was {old_count}), "
                    f"version={version}"
                )

    def on_capacity_consumed(self, count: int) -> None:
        """Record that capacity was consumed (samples taken for training).

        This reduces the filtered count proportionally, assuming that some of
        the consumed capacity corresponds to the compensation we added.

        Args:
            count: Number of samples consumed from cache
        """
        if count <= 0:
            return

        with self.lock:
            # Reduce filtered count, but don't go negative
            reduction = min(count, self._filtered_count)
            self._filtered_count = max(0, self._filtered_count - reduction)

    def modify_capacity(
        self,
        base_capacity: int,
        current_version: int,
        stats: RolloutStat,
    ) -> int:
        """Add filtered count back to capacity.

        Args:
            base_capacity: Base capacity calculated by StalenessManager
            current_version: Current model version
            stats: Current rollout statistics

        Returns:
            Adjusted capacity (base_capacity + filtered_count)
        """
        with self.lock:
            modified_capacity = base_capacity + self._filtered_count

            # Debug logging for capacity adjustment
            if self.logger and self._filtered_count > 0:
                self.logger.debug(
                    f"[CapacityModifier] Capacity adjusted: "
                    f"base={base_capacity}, filtered_count={self._filtered_count}, "
                    f"adjusted={modified_capacity}, version={current_version}"
                )

            return modified_capacity

    def get_filtered_count(self) -> int:
        """Get current filtered sample count.

        Returns:
            Number of filtered samples being compensated for
        """
        with self.lock:
            return self._filtered_count

    def get_version_stats(self) -> dict[int, int]:
        """Get filtered sample counts by version.

        Returns:
            Dictionary mapping version -> filtered count
        """
        with self.lock:
            return self._version_filtered_count.copy()

    def reset(self) -> None:
        """Reset filtered count (for testing or manual intervention)."""
        with self.lock:
            self._filtered_count = 0
            self._version_filtered_count.clear()
