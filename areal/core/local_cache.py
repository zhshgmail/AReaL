"""Local cache implementation with filter support.

This module provides a local (in-process) cache implementation using list
internally, with support for filter-based admission control and event handling.

This is one concrete implementation of CacheAPI. Future implementations could use
Redis, Etcd, Memcached, etc. for distributed caches.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from areal.api.cache_event_handler import CacheEventContext
from areal.api.event_api import EventContext


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

    def __init__(
        self,
        filter_context: Any | None = None,
        engine: Any | None = None,
        config: Any | None = None,
        logger: Any | None = None,
    ):
        """Initialize filterable cache with event support.

        Parameters
        ----------
        filter_context : Any | None, optional
            Context passed to filters when checking items. Default is None.
        engine : Any | None, optional
            Inference engine reference. Default is None.
        config : Any | None, optional
            Configuration object. Default is None.
        logger : Any | None, optional
            Logger instance. Default is None.
        """
        self._cache: list[Any] = []
        self._filters: list = []
        self._filter_context: Any = filter_context
        self._event_handlers: list = []
        self._engine = engine
        self._config = config
        self._logger = logger

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

    def register_event_handler(self, handler) -> None:
        """Register a cache event handler.

        Parameters
        ----------
        handler : CacheEventHandler
            Handler implementing on_cache_event(context)
        """
        self._event_handlers.append(handler)

    def on_event(self, context: EventContext) -> None:
        """Handle global event by propagating to cache event handlers.

        This method is called by EventPropagator when a global event fires.
        It creates a CacheEventContext with metadata and calls registered
        cache event handlers.

        Parameters
        ----------
        context : EventContext
            Global event context
        """
        if not self._event_handlers:
            return

        # Create cache-specific context with metadata
        cache_context = CacheEventContext(
            event_type=context.event_type,
            engine=self._engine or context.engine,
            config=self._config or context.config,
            logger=self._logger or context.logger,
            cache_metadata={
                "size": len(self._cache),
                # Provide access method for processing items
                "process_items": self.process_all_items,
            },
        )

        # Propagate to cache event handlers
        for handler in self._event_handlers:
            try:
                handler.on_cache_event(cache_context)
            except Exception as e:
                if self._logger:
                    self._logger.error(
                        f"Error in cache event handler {type(handler).__name__}: {e}"
                    )
                    traceback.print_exc()

    def process_all_items(
        self,
        processor: Callable[[Any, int], int],
    ) -> int:
        """Process all items in the cache using a processor function.

        This method provides safe access to cache items for event handlers.
        It iterates through all items and allows the processor to modify them.

        Parameters
        ----------
        processor : Callable[[Any, int], int]
            Function that processes an item and returns count of changes made.
            Signature: processor(item, item_index) -> int

        Returns
        -------
        int
            Total count of changes made by processor across all items
        """
        total_changes = 0

        try:
            for idx, item in enumerate(self._cache):
                try:
                    changes = processor(item, idx)
                    total_changes += changes
                except Exception:
                    if self._logger:
                        self._logger.error(f"Error processing cache item #{idx}:")
                    traceback.print_exc()

            if self._logger:
                self._logger.debug(
                    f"Cache process: processed {len(self._cache)} items"
                )

        except Exception:
            if self._logger:
                self._logger.error("Error in process_all_items:")
            traceback.print_exc()

        return total_changes
