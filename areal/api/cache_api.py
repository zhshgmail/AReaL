"""Cache API base class for result storage.

This module defines the Cache interface that WorkflowExecutor uses.
Concrete implementations can use different backends (local list,
distributed cache, persistent storage, etc.) without changing client code.
"""

from __future__ import annotations

import abc
from typing import Any, Generic, Iterator, TypeVar

# Type variable for generic cache item types
T = TypeVar("T")


class CacheAPI(abc.ABC, Generic[T]):
    """Abstract base class defining the Cache interface for WorkflowExecutor.

    This interface abstracts cache operations, allowing different
    implementations (local, distributed, persistent, etc.) without
    changing client code.

    Implementations should handle:
    - Efficient storage and retrieval
    - Filter-based admission control (if applicable)
    - Thread safety (if needed)

    Examples of concrete implementations:
    - LocalCache: Uses list with local filters
    - RedisCache: Distributed cache using Redis
    - EtcdCache: Distributed cache using Etcd
    - MemcachedCache: Distributed cache using Memcached
    """

    @abc.abstractmethod
    def append(self, item: T) -> None:
        """Append an item to the cache.

        Parameters
        ----------
        item : T
            Item to append to cache
        """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of items in cache.

        Returns
        -------
        int
            Number of items
        """
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Iterate over cache items.

        Yields
        ------
        T
            Items in cache
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int | slice) -> T:
        """Get item(s) by index.

        Parameters
        ----------
        index : int | slice
            Index or slice

        Returns
        -------
        T
            Item(s) at index
        """
        pass

    @abc.abstractmethod
    def __setitem__(self, index: int, value: T) -> None:
        """Set item by index.

        Parameters
        ----------
        index : int
            Index
        value : T
            Value to set
        """
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all items from cache."""
        pass

    @abc.abstractmethod
    def extend(self, items: list[T]) -> None:
        """Extend cache with multiple items.

        Parameters
        ----------
        items : list[T]
            Items to extend cache with
        """
        pass

    @abc.abstractmethod
    def pop(self, index: int = -1) -> T:
        """Remove and return item at index.

        Parameters
        ----------
        index : int, optional
            Index to pop. Default is -1 (last item).

        Returns
        -------
        T
            Popped item
        """
        pass
