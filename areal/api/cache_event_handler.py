"""Cache event handler protocol.

This module defines the interface for handling events within Cache implementations.
Cache-specific event handlers are registered ON the cache, not in the global
EventRegistry, maintaining encapsulation of cache internals.
"""

from __future__ import annotations

from typing import Any, Protocol


class CacheEventContext:
    """Context for cache-specific events.

    This context is created by the Cache implementation and passed to
    cache event handlers. It does NOT expose the cache's internal structure
    (e.g., list), only the information handlers need.

    Attributes
    ----------
    event_type : Any
        Type of event (e.g., PRE_UPDATE, POST_UPDATE)
    engine : Any
        Inference engine reference
    config : Any
        Configuration object
    logger : Any
        Logger instance
    cache_metadata : dict
        Cache-specific metadata (size, items count, etc.)
        Does NOT include direct cache access
    """

    def __init__(
        self,
        event_type: Any,
        engine: Any,
        config: Any,
        logger: Any,
        cache_metadata: dict[str, Any] | None = None,
    ):
        self.event_type = event_type
        self.engine = engine
        self.config = config
        self.logger = logger
        self.cache_metadata = cache_metadata or {}


class CacheEventHandler(Protocol):
    """Protocol for handlers of cache-specific events.

    Cache event handlers are registered ON the cache implementation itself,
    not in the global EventRegistry. This maintains encapsulation - handlers
    work with CacheEventContext (which doesn't expose cache internals) rather
    than accessing list directly.

    Examples
    --------
    >>> class CacheProximalRecomputer:
    ...     def on_cache_event(self, context: CacheEventContext):
    ...         # Work with context metadata, not direct cache access
    ...         if context.event_type == EventType.PRE_UPDATE:
    ...             items_to_recompute = context.cache_metadata['stale_items']
    ...             for item in items_to_recompute:
    ...                 self.recompute(item)
    """

    def on_cache_event(self, context: CacheEventContext) -> None:
        """Handle cache-specific event.

        Parameters
        ----------
        context : CacheEventContext
            Cache event context with metadata, not direct cache access
        """
        ...
