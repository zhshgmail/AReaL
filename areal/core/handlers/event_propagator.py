"""Event propagator for distributing global events to queue and cache.

This handler is registered in the global EventRegistry and propagates events
to queue and cache without exposing their internal structures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areal.api.event_api import EventContext

if TYPE_CHECKING:
    pass


class EventPropagator:
    """Propagates global events to queue and cache.

    This handler is registered in the global EventRegistry. When a global
    event fires (e.g., BEFORE_POLICY_UPDATE), this propagator calls the
    on_event() methods of the queue and cache, which in turn propagate
    to their registered event handlers.

    This maintains encapsulation - the propagator doesn't know about
    queue/cache internals or their handlers. It just calls on_event().

    Attributes
    ----------
    queue : Any
        Queue instance with on_event(context) method
    cache : Any
        Cache instance with on_event(context) method

    Examples
    --------
    >>> propagator = EventPropagator(queue, cache)
    >>> registry.register_handler(EventType.BEFORE_POLICY_UPDATE, propagator)
    >>>
    >>> # When event fires, propagator calls queue.on_event() and cache.on_event()
    >>> context = EventContext(EventType.BEFORE_POLICY_UPDATE, ...)
    >>> registry.fire_event(context)
    """

    def __init__(self, queue, cache):
        """Initialize event propagator.

        Parameters
        ----------
        queue : Any
            Queue instance with on_event(context) method
        cache : Any
            Cache instance with on_event(context) method
        """
        self.queue = queue
        self.cache = cache

    def on_event(self, context: EventContext) -> None:
        """Propagate global event to queue and cache.

        Parameters
        ----------
        context : EventContext
            Global event context
        """
        # Propagate to queue
        if hasattr(self.queue, "on_event"):
            try:
                self.queue.on_event(context)
            except Exception as e:
                if context.logger:
                    context.logger.error(f"Error propagating event to queue: {e}")

        # Propagate to cache
        if hasattr(self.cache, "on_event"):
            try:
                self.cache.on_event(context)
            except Exception as e:
                if context.logger:
                    context.logger.error(f"Error propagating event to cache: {e}")
