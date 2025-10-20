"""Cache abstraction for rollout workflow.

This module provides abstract cache interface that can be implemented
using local Python list, distributed cache (Ray, Redis), etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List


class RolloutCache(ABC):
    """Abstract cache interface for rollout results.

    This abstraction allows the cache implementation to be swapped
    (e.g., local list, Ray object store, Redis cache, etc.) without
    changing the WorkflowExecutor code.
    """

    @abstractmethod
    def add(self, item: Any) -> None:
        """Add item to cache.

        Args:
            item: Item to add to cache
        """
        pass

    @abstractmethod
    def get_all(self) -> List[Any]:
        """Get read-only view of all items in cache.

        Returns:
            List of all items (may be a copy depending on implementation)
        """
        pass

    @abstractmethod
    def take_first_n(self, n: int) -> List[Any]:
        """Remove and return first n items from cache.

        Args:
            n: Number of items to take

        Returns:
            List of first n items (or all items if cache has fewer than n)
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Return number of items in cache.

        Returns:
            Number of items currently in cache
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all items from cache."""
        pass

    @abstractmethod
    def filter_inplace(self, predicate: Callable[[Any], bool]) -> int:
        """Remove items not matching predicate.

        Modifies cache in-place by keeping only items where predicate(item) is True.

        Args:
            predicate: Function that returns True for items to keep

        Returns:
            Number of items removed
        """
        pass


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


# Future implementations:
# class RayRolloutCache(RolloutCache):
#     """Distributed cache implementation using Ray object store."""
#     pass
#
# class RedisRolloutCache(RolloutCache):
#     """Distributed cache implementation using Redis."""
#     pass
