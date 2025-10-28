"""Local queue implementation with filter support.

This module provides a local (in-process) queue implementation using queue.Queue
internally, with support for filter-based admission control.

This is one concrete implementation of QueueAPI. Future implementations could use
ZeroMQ, Redis, Etcd, etc. for distributed queues.
"""

from __future__ import annotations

import queue
from typing import Any


class LocalQueue:
    """Local (in-process) queue implementation with filter support.

    This class implements QueueAPI using queue.Queue internally, with support
    for filter-based admission control. Filters are registered on the queue
    and checked during put operations.

    The queue owns its filters and doesn't expose them to consumers.
    From AsyncTaskRunner's perspective, it just calls put() and doesn't
    know WHY items might be rejected.

    This is suitable for single-process applications. For distributed systems,
    use ZeroMQQueue, RedisQueue, or other distributed implementations.

    Attributes
    ----------
    _queue : queue.Queue
        Internal queue for storage
    _filters : list
        List of filters registered on this queue
    _filter_context : Any
        Context passed to filters when checking items

    Examples
    --------
    >>> queue = LocalQueue(maxsize=10)
    >>> queue.register_filter(StalenessFilter(max_staleness=2))
    >>> queue.put_nowait(item)  # May be silently dropped by filter
    """

    def __init__(self, maxsize: int = 0, filter_context: Any | None = None):
        """Initialize filterable queue.

        Parameters
        ----------
        maxsize : int, optional
            Maximum queue size (0 = unlimited). Default is 0.
        filter_context : Any | None, optional
            Context passed to filters when checking items. Default is None.
        """
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._filters: list = []
        self._filter_context: Any = filter_context

    def register_filter(self, filter_obj) -> None:
        """Register a filter for admission control.

        Filters are checked in registration order when put_nowait() is called.
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
        """Add item to queue after checking filters.

        This method checks all registered filters. If any filter
        rejects the item (returns False), the item is not added.

        Parameters
        ----------
        item : Any
            Item to add to queue
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

        # All filters passed, add to queue
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            # Queue full is not a filter rejection
            raise

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        """Put item directly to internal queue (bypasses filters).

        This is for internal use when filters should not be applied.

        Parameters
        ----------
        item : Any
            Item to put
        block : bool, optional
            Whether to block if queue full. Default is True.
        timeout : float | None, optional
            Timeout in seconds. Default is None.
        """
        self._queue.put(item, block=block, timeout=timeout)

    def put_nowait(self, item: Any) -> None:
        """Put item to queue after checking filters.

        Filters are checked if filters are registered and context is available.
        If any filter rejects the item, it is silently dropped (not added).

        Parameters
        ----------
        item : Any
            Item to put
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

        # Filters passed or no filters, add to queue
        self._queue.put_nowait(item)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Get item from queue.

        Parameters
        ----------
        block : bool, optional
            Whether to block if queue empty. Default is True.
        timeout : float | None, optional
            Timeout in seconds. Default is None.

        Returns
        -------
        Any
            Item from queue
        """
        return self._queue.get(block=block, timeout=timeout)

    def get_nowait(self) -> Any:
        """Get item from queue without blocking.

        Returns
        -------
        Any
            Item from queue
        """
        return self._queue.get_nowait()

    def qsize(self) -> int:
        """Return approximate queue size.

        Returns
        -------
        int
            Number of items in queue
        """
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if queue is empty.

        Returns
        -------
        bool
            Whether queue is empty
        """
        return self._queue.empty()

    def full(self) -> bool:
        """Return True if queue is full.

        Returns
        -------
        bool
            Whether queue is full
        """
        return self._queue.full()
