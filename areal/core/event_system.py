"""Event Registry for managing global event handlers.

This module provides EventRegistry for managing and firing global system events.
Event types, contexts, and handler protocols are defined in the api module.
"""

from __future__ import annotations

from areal.api.event_api import EventContext, EventHandler, EventType


class EventRegistry:
    """Registry for managing global event handlers.

    The registry maintains mappings from event types to handlers and provides
    synchronous event firing. When fire_event() returns, all registered handlers
    for that event have completed execution.

    This provides a clean extension point for adding behavior without modifying
    core workflow logic.

    Examples
    --------
    >>> registry = EventRegistry()
    >>> registry.register_handler(EventType.BEFORE_POLICY_UPDATE, propagator)
    >>>
    >>> # Fire event - all handlers execute synchronously
    >>> context = EventContext(EventType.BEFORE_POLICY_UPDATE, engine, config, logger)
    >>> registry.fire_event(context)
    >>> # When this returns, all handlers have completed
    """

    def __init__(self):
        """Initialize empty registry."""
        self._handlers: dict[EventType, list[EventHandler]] = {}

    def register_handler(
        self, event_type: EventType, handler: EventHandler
    ) -> None:
        """Register a handler for an event type.

        Multiple handlers can be registered for the same event type.
        They will execute in registration order.

        Parameters
        ----------
        event_type : EventType
            Type of event to handle
        handler : EventHandler
            Handler to register
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def fire_event(self, context: EventContext) -> None:
        """Fire an event and execute all registered handlers synchronously.

        Handlers execute in registration order. If a handler raises an exception,
        subsequent handlers for that event will not execute.

        Parameters
        ----------
        context : EventContext
            Event context with type, engine, config, logger, and data

        Raises
        ------
        Exception
            Propagates exceptions from handlers
        """
        handlers = self._handlers.get(context.event_type, [])

        if handlers and context.logger:
            context.logger.debug(
                f"Firing event {context.event_type.value} with {len(handlers)} handlers"
            )

        for handler in handlers:
            try:
                handler.on_event(context)
            except Exception as e:
                if context.logger:
                    context.logger.error(
                        f"Error in handler {type(handler).__name__} for event "
                        f"{context.event_type.value}: {e}"
                    )
                raise

    def get_handlers(self, event_type: EventType) -> list[EventHandler]:
        """Get all handlers registered for an event type.

        Parameters
        ----------
        event_type : EventType
            Event type to query

        Returns
        -------
        list[EventHandler]
            List of registered handlers (may be empty)
        """
        return self._handlers.get(event_type, []).copy()

    def clear_handlers(self, event_type: EventType | None = None) -> None:
        """Clear handlers for an event type, or all handlers if type is None.

        Parameters
        ----------
        event_type : EventType | None
            Event type to clear, or None to clear all
        """
        if event_type is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event_type, None)
