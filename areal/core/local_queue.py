"""Local queue implementation with filter support.

This module provides a local (in-process) queue implementation using queue.Queue
internally, with support for filter-based admission control and event handling.

This is one concrete implementation of QueueAPI. Future implementations could use
ZeroMQ, Redis, Etcd, etc. for distributed queues.
"""

from __future__ import annotations

import queue
import traceback
from typing import Any, Callable, TypeVar

from areal.api.event_api import EventContext
from areal.api.filter_api import Filter, FilterContext
from areal.api.queue_api import QueueAPI
from areal.api.queue_event_handler import QueueEventContext, QueueEventHandler

# Type variable for queue items
T = TypeVar("T")


class LocalQueue(QueueAPI[T]):
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

    def __init__(
        self,
        maxsize: int = 0,
        config: Any | None = None,
        logger: Any | None = None,
    ):
        """Initialize filterable queue with event support.

        Queue manages its own FilterContext internally. Filters receive
        FilterContext during admission checks, not the full EventContext.

        Parameters
        ----------
        maxsize : int, optional
            Maximum queue size (0 = unlimited). Default is 0.
        config : Any | None, optional
            Configuration object for FilterContext. Default is None.
        logger : Any | None, optional
            Logger instance for FilterContext. Default is None.
        """
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._filters: list = []
        self._event_handlers: list = []
        self._config = config
        self._logger = logger
        # Queue creates its own FilterContext
        self._filter_context: FilterContext | None = self._create_filter_context()

    def _create_filter_context(self) -> FilterContext | None:
        """Create FilterContext for filters.

        Returns
        -------
        FilterContext | None
            FilterContext if config available, None otherwise
        """
        if self._config is None:
            return None
        return FilterContext(config=self._config, logger=self._logger)

    def register_filter(self, filter_obj: Filter) -> None:
        """Register a filter for admission control.

        Filters are checked in registration order when put_nowait() is called.
        If any filter rejects the item, the item is silently dropped.

        Parameters
        ----------
        filter_obj : Filter
            Filter implementing should_accept(item, context) -> bool
        """
        self._filters.append(filter_obj)

    def set_logger(self, logger: Any) -> None:
        """Update logger and recreate FilterContext.

        This is called during WorkflowExecutor.initialize() to set the logger
        after Queue construction.

        Parameters
        ----------
        logger : Any
            Logger instance
        """
        self._logger = logger
        self._filter_context = self._create_filter_context()

    @property
    def maxsize(self) -> int:
        """Return maximum queue size.

        Returns
        -------
        int
            Maximum queue size (0 means unlimited)
        """
        return self._queue.maxsize

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
        """Put item to queue after checking filters.

        Filters are checked if filters are registered and context is available.
        If any filter rejects the item, it is silently dropped (not added).

        Parameters
        ----------
        item : Any
            Item to put
        block : bool, optional
            Whether to block if queue full. Default is True.
        timeout : float | None, optional
            Timeout in seconds. Default is None.
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
        self._queue.put(item, block=block, timeout=timeout)

    def put_nowait(self, item: Any) -> None:
        """Put item to queue without blocking, after checking filters.

        This is equivalent to put(item, block=False).
        Filters are checked if filters are registered and context is available.
        If any filter rejects the item, it is silently dropped (not added).

        Parameters
        ----------
        item : Any
            Item to put

        Raises
        ------
        queue.Full
            If queue is full and item passes filters
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

    def register_event_handler(self, handler: QueueEventHandler) -> None:
        """Register a queue event handler.

        Parameters
        ----------
        handler : QueueEventHandler
            Handler implementing on_queue_event(context)
        """
        self._event_handlers.append(handler)

    def on_event(self, context: EventContext) -> None:
        """Handle global event by propagating to queue event handlers.

        This method is called by EventPropagator when a global event fires.
        It creates a QueueEventContext with metadata and calls registered
        queue event handlers.

        Parameters
        ----------
        context : EventContext
            Global event context
        """
        if not self._event_handlers:
            return

        # Create queue-specific context with metadata
        # Use context values (EventContext always has engine, config)
        # Prefer stored logger if available for consistency
        queue_context = QueueEventContext(
            event_type=context.event_type,
            engine=context.engine,
            config=context.config,
            logger=self._logger or context.logger,
            queue_metadata={
                "size": self.qsize(),
                "empty": self.empty(),
                "full": self.full(),
                # Provide access method for processing items
                "process_items": self.process_all_items,
            },
        )

        # Propagate to queue event handlers
        for handler in self._event_handlers:
            try:
                handler.on_queue_event(queue_context)
            except Exception as e:
                if self._logger:
                    self._logger.error(
                        f"Error in queue event handler {type(handler).__name__}: {e}"
                    )
                    traceback.print_exc()

    def process_all_items(
        self,
        processor: Callable[[Any, int], int],
        max_iterations: int = 3,
    ) -> int:
        """Process all items in the queue using a processor function.

        This method provides safe access to queue items for event handlers.
        It drains the queue, processes each item, and puts items back.

        Parameters
        ----------
        processor : Callable[[Any, int], int]
            Function that processes an item and returns count of changes made.
            Signature: processor(item, item_index) -> int
        max_iterations : int, optional
            Maximum number of drain-process-putback iterations. Default is 3.

        Returns
        -------
        int
            Total count of changes made by processor across all items
        """
        total_changes = 0

        try:
            for iteration in range(max_iterations):
                # Drain queue
                temp_items = []
                while True:
                    try:
                        item = self._queue.get_nowait()
                        temp_items.append(item)
                    except queue.Empty:
                        break

                if not temp_items:
                    break

                # Process items
                for idx, item in enumerate(temp_items):
                    try:
                        changes = processor(item, idx)
                        total_changes += changes
                    except Exception:
                        if self._logger:
                            self._logger.error(
                                f"Error processing queue item #{idx}:"
                            )
                        traceback.print_exc()

                # Put items back
                for item in temp_items:
                    try:
                        self._queue.put_nowait(item)
                    except queue.Full:
                        try:
                            self._queue.put(item, timeout=1.0)
                        except queue.Full:
                            if self._logger:
                                self._logger.error(
                                    "Queue full during putback, item dropped!"
                                )

                if self._logger:
                    self._logger.debug(
                        f"Queue process iteration {iteration + 1}: "
                        f"processed {len(temp_items)} items"
                    )

        except Exception:
            if self._logger:
                self._logger.error("Error in process_all_items:")
            traceback.print_exc()

        return total_changes
