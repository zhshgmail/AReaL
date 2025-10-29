"""Cache event handler base class.

This module defines the interface for handling events within Cache implementations.
Cache-specific event handlers are registered ON the cache, not in the global
EventRegistry, maintaining encapsulation of cache internals.
"""

from __future__ import annotations

import abc
from typing import Any

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.event_api import EventContext, EventType


class CacheEventContext(EventContext):
    """Context for cache-specific events.

    This context extends EventContext with cache-specific metadata.
    It is created by the Cache implementation and passed to cache event handlers.
    It does NOT expose the cache's internal structure (e.g., list),
    only the information handlers need.

    Attributes
    ----------
    event_type : EventType
        Type of event (e.g., BEFORE_POLICY_UPDATE)
    engine : InferenceEngine
        Inference engine reference
    config : InferenceEngineConfig
        Configuration object
    logger : Any
        Logger instance
    data : dict[str, Any]
        Event-specific data (inherited from EventContext)
    cache_metadata : dict[str, Any]
        Cache-specific metadata (size, process_items function)
        Does NOT include direct cache access
    """

    def __init__(
        self,
        event_type: EventType,
        engine: InferenceEngine,
        config: InferenceEngineConfig,
        logger: Any,
        cache_metadata: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ):
        """Initialize cache event context.

        Parameters
        ----------
        event_type : EventType
            Type of event
        engine : InferenceEngine
            Inference engine reference
        config : InferenceEngineConfig
            Configuration object
        logger : Any
            Logger instance
        cache_metadata : dict[str, Any] | None, optional
            Cache-specific metadata. Default is None.
        data : dict[str, Any] | None, optional
            Event-specific data. Default is None.
        """
        super().__init__(event_type, engine, config, logger, data)
        self.cache_metadata = cache_metadata or {}


class CacheEventHandler(abc.ABC):
    """Abstract base class for handlers of cache-specific events.

    Cache event handlers are registered ON the cache implementation itself,
    not in the global EventRegistry. This maintains encapsulation - handlers
    work with CacheEventContext (which doesn't expose cache internals) rather
    than accessing list directly.

    Examples
    --------
    >>> class CacheProximalRecomputer(CacheEventHandler):
    ...     def on_cache_event(self, context: CacheEventContext):
    ...         # Work with context metadata, not direct cache access
    ...         if context.event_type == EventType.PRE_UPDATE:
    ...             items_to_recompute = context.cache_metadata['stale_items']
    ...             for item in items_to_recompute:
    ...                 self.recompute(item)
    """

    @abc.abstractmethod
    def on_cache_event(self, context: CacheEventContext) -> None:
        """Handle cache-specific event.

        Parameters
        ----------
        context : CacheEventContext
            Cache event context with metadata, not direct cache access
        """
        pass
