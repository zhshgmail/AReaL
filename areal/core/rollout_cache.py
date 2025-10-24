"""Local rollout cache implementation.

This module contains the concrete implementation of RolloutCache using Python list.
"""

from __future__ import annotations

from typing import Any, Callable, List

from areal.api.cache_api import RolloutCache


class LocalRolloutCache(RolloutCache):
    """Local implementation of RolloutCache using Python list.

    This is the default implementation used when no distributed
    cache is configured.
    """

    def __init__(self):
        """Initialize local cache with empty list."""
        self._cache: List[Any] = []

    def add(self, item: Any) -> None:
        """Add item to cache."""
        self._cache.append(item)

    def get_all(self) -> List[Any]:
        """Get copy of all items in cache."""
        return self._cache.copy()

    def take_first_n(self, n: int) -> List[Any]:
        """Remove and return first n items from cache."""
        result = self._cache[:n]
        self._cache = self._cache[n:]
        return result

    def size(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)

    def clear(self) -> None:
        """Remove all items from cache."""
        self._cache.clear()

    def filter_inplace(self, predicate: Callable[[Any], bool]) -> int:
        """Remove items not matching predicate.

        Args:
            predicate: Function that returns True for items to keep

        Returns:
            Number of items removed
        """
        original_size = len(self._cache)
        self._cache = [item for item in self._cache if predicate(item)]
        return original_size - len(self._cache)
