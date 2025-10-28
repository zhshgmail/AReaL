"""Filter protocol for admission control.

This module defines the Filter interface used by both Queue and Cache
implementations for admission control (accept/reject items).
"""

from __future__ import annotations

from typing import Any, Protocol


class Filter(Protocol):
    """Protocol for admission control filters.

    Filters decide whether an item should be accepted when added to
    a queue or cache. They can be registered on any QueueAPI or CacheAPI
    implementation.

    Filters are checked during add operations (put, append, etc.) and
    can reject items by returning False.

    Use cases:
    - Reject over-stale samples
    - Enforce capacity limits based on metadata
    - Validate data format
    - Apply custom admission policies

    Examples
    --------
    >>> class StalenessFilter:
    ...     def should_accept(self, item, context):
    ...         staleness = self._calculate_staleness(item, context)
    ...         if staleness > self.max_staleness:
    ...             context.logger.warning(f"Rejecting stale item: {staleness}")
    ...             return False
    ...         return True
    """

    def should_accept(self, item: Any, context: Any) -> bool:
        """Check if item should be accepted.

        Parameters
        ----------
        item : Any
            Item to be added (typically TensorDict sample)
        context : Any
            Context with engine, config, logger (typically EventContext)

        Returns
        -------
        bool
            True if item should be accepted, False to reject

        Raises
        ------
        Exception
            Can raise exception to reject item with error message
        """
        ...
