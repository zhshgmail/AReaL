"""Local cache implementation with filter support.

This module provides a local (in-process) cache implementation using list
internally, with support for filter-based admission control.

This is one concrete implementation of CacheAPI. Future implementations could use
Redis, Etcd, Memcached, etc. for distributed caches.
"""

from __future__ import annotations

from typing import Any


class LocalCache:
    """Local (in-process) cache implementation with filter support.

    This class implements CacheAPI using list internally, with support
    for filter-based admission control. Filters are registered on the cache
    and checked during append operations.

    The cache owns its filters and doesn't expose them to consumers.
    From WorkflowExecutor's perspective, it just calls append() and doesn't
    know WHY items might be rejected.

    This is suitable for single-process applications. For distributed systems,
    use RedisCache, EtcdCache, or other distributed implementations.

    Attributes
    ----------
    _cache : list
        Internal list for storage
    _filters : list
        List of filters registered on this cache
    _filter_context : Any
        Context passed to filters when checking items

    Examples
    --------
    >>> cache = LocalCache()
    >>> cache.register_filter(StalenessFilter(max_staleness=2))
    >>> cache.append(item)  # May be silently dropped by filter
    """

    def __init__(self, filter_context: Any | None = None):
        """Initialize filterable cache.

        Parameters
        ----------
        filter_context : Any | None, optional
            Context passed to filters when checking items. Default is None.
        """
        self._cache: list[Any] = []
        self._filters: list = []
        self._filter_context: Any = filter_context

    def register_filter(self, filter_obj) -> None:
        """Register a filter for admission control.

        Filters are checked in registration order when append() is called.
        If any filter rejects the item, the item is silently dropped.

        Parameters
        ----------
        filter_obj : QueueFilter
            Filter implementing should_accept(item, context) -> bool
        """
        self._filters.append(filter_obj)

    def set_filter_context(self, context: Any) -> None:
        """Set the context passed to filters.

        Parameters
        ----------
        context : Any
            Context object (typically EventContext)
        """
        self._filter_context = context

    def add(self, item: Any, context: Any) -> bool:
        """Add item to cache after checking filters.

        This method checks all registered filters. If any filter
        rejects the item (returns False), the item is not added.

        Parameters
        ----------
        item : Any
            Item to add to cache
        context : Any
            Context passed to filters (typically EventContext)

        Returns
        -------
        bool
            True if item was added, False if rejected by filters
        """
        # Check all filters
        for filter_obj in self._filters:
            try:
                if not filter_obj.should_accept(item, context):
                    # Rejected by filter
                    return False
            except Exception:
                # On filter error, reject for safety
                return False

        # All filters passed, add to cache
        self._cache.append(item)
        return True

    def append(self, item: Any) -> None:
        """Append item to cache after checking filters.

        Filters are checked if filters are registered and context is available.
        If any filter rejects the item, it is silently dropped (not added).

        Parameters
        ----------
        item : Any
            Item to append
        """
        # Check filters if available
        if self._filters and self._filter_context:
            for filter_obj in self._filters:
                try:
                    if not filter_obj.should_accept(item, self._filter_context):
                        # Silently drop rejected items
                        return
                except Exception:
                    # On filter error, drop for safety
                    return

        # Filters passed or no filters, add to cache
        self._cache.append(item)

    def __len__(self) -> int:
        """Return number of items in cache.

        Returns
        -------
        int
            Number of items
        """
        return len(self._cache)

    def __iter__(self):
        """Iterate over cache items.

        Yields
        ------
        Any
            Items in cache
        """
        return iter(self._cache)

    def __getitem__(self, index):
        """Get item by index.

        Parameters
        ----------
        index : int or slice
            Index or slice

        Returns
        -------
        Any
            Item(s) at index
        """
        return self._cache[index]

    def __setitem__(self, index, value):
        """Set item by index.

        Parameters
        ----------
        index : int
            Index
        value : Any
            Value to set
        """
        self._cache[index] = value

    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()

    def extend(self, items) -> None:
        """Extend cache with items (bypasses filters).

        Parameters
        ----------
        items : iterable
            Items to extend with
        """
        self._cache.extend(items)

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
        return self._cache.pop(index)
