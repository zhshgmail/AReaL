"""Event handlers for queue and cache.

These classes implement EventHandler protocol and wrap LocalQueue/LocalCache,
allowing queue/cache to respond to system events while maintaining single
responsibility principle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areal.core.event_system import EventContext
    from areal.core.local_cache import LocalCache
    from areal.core.local_queue import LocalQueue


class QueueEventHandler:
    """Event handler for LocalQueue (or any QueueAPI implementation).

    This class wraps a queue and implements EventHandler protocol,
    allowing the queue to respond to system events without violating SRP.

    The queue itself remains a simple data structure with filters.
    This handler adds event-driven behavior on top.

    Parameters
    ----------
    queue : LocalQueue (or any QueueAPI implementation)
        The queue to handle events for

    Examples
    --------
    >>> queue = LocalQueue(maxsize=100)
    >>> handler = QueueEventHandler(queue)
    >>> registry.register_handler(EventType.PRE_UPDATE, handler)
    """

    def __init__(self, queue: "LocalQueue"):
        """Initialize handler with queue.

        Parameters
        ----------
        queue : LocalQueue (or any QueueAPI implementation)
            Queue to handle events for
        """
        self.queue = queue

    def on_event(self, context: "EventContext") -> None:
        """Handle event for queue.

        Currently this is a placeholder for future queue-specific
        event handling (e.g., scanning for stale items, logging stats).

        The actual recompute logic is handled by ProximalRecomputer
        which accesses queue via context.data.

        Parameters
        ----------
        context : EventContext
            Event context with type, engine, config, logger
        """
        # Placeholder for future queue-specific event handling
        # For example:
        # - On PRE_UPDATE: scan queue and log statistics
        # - On POST_UPDATE: clear stale items
        # - On BEFORE_PAUSE: checkpoint queue state
        pass


class CacheEventHandler:
    """Event handler for LocalCache (or any CacheAPI implementation).

    This class wraps a cache and implements EventHandler protocol,
    allowing the cache to respond to system events without violating SRP.

    The cache itself remains a simple data structure with filters.
    This handler adds event-driven behavior on top.

    Parameters
    ----------
    cache : LocalCache (or any CacheAPI implementation)
        The cache to handle events for

    Examples
    --------
    >>> cache = LocalCache()
    >>> handler = CacheEventHandler(cache)
    >>> registry.register_handler(EventType.PRE_UPDATE, handler)
    """

    def __init__(self, cache: "LocalCache"):
        """Initialize handler with cache.

        Parameters
        ----------
        cache : LocalCache (or any CacheAPI implementation)
            Cache to handle events for
        """
        self.cache = cache

    def on_event(self, context: "EventContext") -> None:
        """Handle event for cache.

        Currently this is a placeholder for future cache-specific
        event handling (e.g., scanning for stale items, logging stats).

        The actual recompute logic is handled by ProximalRecomputer
        which accesses cache via context.data.

        Parameters
        ----------
        context : EventContext
            Event context with type, engine, config, logger
        """
        # Placeholder for future cache-specific event handling
        # For example:
        # - On PRE_UPDATE: scan cache and log statistics
        # - ON POST_UPDATE: clear stale items
        # - On BEFORE_PAUSE: checkpoint cache state
        pass
