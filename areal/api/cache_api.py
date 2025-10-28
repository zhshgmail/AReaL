"""Cache API protocol for result storage.

This module defines the Cache interface that WorkflowExecutor uses.
Concrete implementations can use different backends (local list,
distributed cache, persistent storage, etc.) without changing client code.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol


class CacheAPI(Protocol):
    """Protocol defining the Cache interface for WorkflowExecutor.

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

    def append(self, item: Any) -> None:
        """Append an item to the cache.

        Parameters
        ----------
        item : Any
            Item to append to cache
        """
        ...

    def __len__(self) -> int:
        """Return the number of items in cache.

        Returns
        -------
        int
            Number of items
        """
        ...

    def __iter__(self) -> Iterator[Any]:
        """Iterate over cache items.

        Yields
        ------
        Any
            Items in cache
        """
        ...

    def __getitem__(self, index: int | slice) -> Any:
        """Get item(s) by index.

        Parameters
        ----------
        index : int | slice
            Index or slice

        Returns
        -------
        Any
            Item(s) at index
        """
        ...

    def __setitem__(self, index: int, value: Any) -> None:
        """Set item by index.

        Parameters
        ----------
        index : int
            Index
        value : Any
            Value to set
        """
        ...

    def clear(self) -> None:
        """Clear all items from cache."""
        ...

    def extend(self, items: list[Any]) -> None:
        """Extend cache with multiple items.

        Parameters
        ----------
        items : list[Any]
            Items to extend cache with
        """
        ...

    def pop(self, index: int = -1) -> Any:
        """Remove and return item at index.

        Parameters
        ----------
        index : int, optional
            Index to pop. Default is -1 (last item).

        Returns
        -------
        Any
            Popped item
        """
        ...
