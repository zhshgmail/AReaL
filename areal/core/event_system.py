"""Event-driven system for queue/cache operations.

This module provides an event-driven architecture for extending queue and cache
behavior without tight coupling to specific algorithms:

1. **Filters**: Admission control when items added to queue/cache
2. **Event Handlers**: React to events (e.g., PolicyUpdated, BeforePause)
3. **Event Registry**: Manages handlers and fires events synchronously

This design minimizes changes to workflow and provides clear separation of concerns.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from areal.api.cli_args import InferenceEngineConfig
    from areal.api.engine_api import InferenceEngine


class EventType(Enum):
    """Event types that can be fired in the system."""

    # Fired before policy/model weights are updated
    BEFORE_POLICY_UPDATE = "before_policy_update"

    # Fired after policy/model weights are updated
    AFTER_POLICY_UPDATE = "after_policy_update"

    # Fired before workflow executor pauses
    BEFORE_PAUSE = "before_pause"

    # Fired after workflow executor pauses
    AFTER_PAUSE = "after_pause"

    # Fired before workflow executor resumes
    BEFORE_RESUME = "before_resume"

    # Fired after workflow executor resumes
    AFTER_RESUME = "after_resume"


class EventContext:
    """Context passed to event handlers.

    Provides access to engine state, configuration, and event-specific data.

    Attributes
    ----------
    event_type : EventType
        The type of event being fired
    engine : InferenceEngine
        The inference engine instance
    config : InferenceEngineConfig
        Training configuration
    logger : Any
        Logger instance for diagnostics
    data : dict
        Event-specific data (e.g., old_version, new_version for policy update)
    """

    def __init__(
        self,
        event_type: EventType,
        engine: "InferenceEngine",
        config: "InferenceEngineConfig",
        logger: Any,
        data: dict[str, Any] | None = None,
    ):
        self.event_type = event_type
        self.engine = engine
        self.config = config
        self.logger = logger
        self.data = data or {}


class QueueFilter(Protocol):
    """Filter for admission control when items added to queue/cache.

    Filters decide whether an item should be accepted into the queue or cache.
    They can reject items by returning False or raising an exception.

    Use cases:
    - Reject over-stale samples
    - Enforce capacity limits
    - Validate data format

    Examples
    --------
    >>> class StalenessFilter(QueueFilter):
    ...     def should_accept(self, item, context):
    ...         staleness = self._calculate_staleness(item, context)
    ...         if staleness > self.max_staleness:
    ...             context.logger.warning(f"Rejecting stale item: {staleness}")
    ...             return False
    ...         return True
    """

    def should_accept(self, item: Any, context: EventContext) -> bool:
        """Check if item should be accepted into queue/cache.

        Parameters
        ----------
        item : Any
            Item to be added (typically TensorDict sample)
        context : EventContext
            Context with engine, config, logger

        Returns
        -------
        bool
            True if item should be accepted, False to reject

        Raises
        ------
        Exception
            Can raise exception to reject item with error message
        """
        ...


class EventHandler(Protocol):
    """Handler for system events.

    Event handlers react to events like policy updates, pause/resume, etc.
    They execute synchronously when events are fired.

    Use cases:
    - Recompute proximal_t before policy update
    - Scan and filter queue/cache on events
    - Log metrics or trigger side effects

    Examples
    --------
    >>> class ProximalRecomputer(EventHandler):
    ...     def on_event(self, context):
    ...         if context.event_type == EventType.BEFORE_POLICY_UPDATE:
    ...             self._recompute_all_samples(context)
    """

    def on_event(self, context: EventContext) -> None:
        """Handle event.

        This method is called synchronously when the event is fired.
        When EventRegistry.fire_event() returns, all handlers have completed.

        Parameters
        ----------
        context : EventContext
            Context with event type, engine, config, logger, and event-specific data
        """
        ...


class EventRegistry:
    """Registry for managing event handlers.

    The registry maintains mappings from event types to handlers and provides
    synchronous event firing. When fire_event() returns, all registered handlers
    for that event have completed execution.

    This provides a clean extension point for adding behavior without modifying
    core workflow logic.

    Examples
    --------
    >>> registry = EventRegistry()
    >>> registry.register_handler(EventType.BEFORE_POLICY_UPDATE, recomputer)
    >>> registry.register_handler(EventType.BEFORE_PAUSE, validator)
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

        if handlers:
            context.logger.debug(
                f"Firing event {context.event_type.value} with {len(handlers)} handlers"
            )

        for handler in handlers:
            try:
                handler.on_event(context)
            except Exception as e:
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
