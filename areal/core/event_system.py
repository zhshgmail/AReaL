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

    EventRegistry stores references to engine, config, and logger to automatically
    construct EventContext when events are fired. This eliminates boilerplate in
    calling code - callers just specify the event type.

    This provides a clean extension point for adding behavior without modifying
    core workflow logic.

    Examples
    --------
    >>> registry = EventRegistry()
    >>> registry.register_handler(EventType.BEFORE_POLICY_UPDATE, propagator)
    >>>
    >>> # Bind registry to engine (usually done by factory)
    >>> registry.bind_to_engine(engine, config, logger)
    >>>
    >>> # Fire event - registry creates EventContext automatically
    >>> registry.fire_event(EventType.BEFORE_POLICY_UPDATE)
    >>> # When this returns, all handlers have completed
    """

    def __init__(self):
        """Initialize empty registry.

        Call bind_to_engine() before firing events to set engine/config/logger.
        """
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._engine = None
        self._config = None
        self._logger = None

    def bind_to_engine(self, engine, config, logger=None) -> None:
        """Bind registry to an engine for automatic EventContext creation.

        This should be called by the factory after creating the registry and
        before any events are fired.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine that will fire events
        config : InferenceEngineConfig
            Configuration object to pass to event handlers
        logger : logging.Logger | None, optional
            Logger for event system. Can be None initially and set later.
        """
        self._engine = engine
        self._config = config
        self._logger = logger

    def set_logger(self, logger) -> None:
        """Update the logger after registry is bound.

        This allows setting logger during initialization phase after
        the registry has been bound to the engine.

        Parameters
        ----------
        logger : logging.Logger
            Logger for event system
        """
        self._logger = logger

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

    def fire_event(self, event_type: EventType, data: dict | None = None) -> None:
        """Fire an event and execute all registered handlers synchronously.

        EventRegistry automatically constructs EventContext using stored engine,
        config, and logger references. This eliminates boilerplate in calling code.

        Handlers execute in registration order. If a handler raises an exception,
        subsequent handlers for that event will not execute.

        Parameters
        ----------
        event_type : EventType
            Type of event to fire
        data : dict | None, optional
            Optional event-specific data to pass to handlers. Default is None.

        Raises
        ------
        RuntimeError
            If registry is not bound to an engine (call bind_to_engine first)
        Exception
            Propagates exceptions from handlers
        """
        if self._engine is None or self._config is None:
            raise RuntimeError(
                "EventRegistry must be bound to an engine before firing events. "
                "Call bind_to_engine(engine, config, logger) first."
            )

        handlers = self._handlers.get(event_type, [])

        if handlers and self._logger:
            self._logger.debug(
                f"Firing event {event_type.value} with {len(handlers)} handlers"
            )

        # Construct EventContext automatically
        context = EventContext(
            event_type=event_type,
            engine=self._engine,
            config=self._config,
            logger=self._logger,
            data=data,
        )

        for handler in handlers:
            try:
                handler.on_event(context)
            except Exception as e:
                if self._logger:
                    self._logger.error(
                        f"Error in handler {type(handler).__name__} for event "
                        f"{event_type.value}: {e}"
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
