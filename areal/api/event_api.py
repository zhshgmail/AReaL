"""Event system API protocols and types.

This module defines the core interfaces for the event-driven architecture:
- EventType: Enum of system events
- EventContext: Context passed to global event handlers
- EventHandler: Protocol for global event handlers
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
    """Context passed to global event handlers.

    This is used for global events fired by EventRegistry.
    Queue-specific and cache-specific events use QueueEventContext
    and CacheEventContext respectively.

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


class EventHandler(Protocol):
    """Protocol for global event handlers.

    Global event handlers are registered in EventRegistry and respond to
    system-wide events (PRE_UPDATE, POST_UPDATE, etc.).

    For queue-specific or cache-specific event handling, use QueueEventHandler
    or CacheEventHandler protocols instead.

    Examples
    --------
    >>> class EventPropagator:
    ...     def __init__(self, queue, cache):
    ...         self.queue = queue
    ...         self.cache = cache
    ...
    ...     def on_event(self, context: EventContext):
    ...         # Propagate to queue and cache
    ...         self.queue.on_event(context)
    ...         self.cache.on_event(context)
    """

    def on_event(self, context: EventContext) -> None:
        """Handle global event.

        This method is called synchronously when the event is fired.
        When EventRegistry.fire_event() returns, all handlers have completed.

        Parameters
        ----------
        context : EventContext
            Context with event type, engine, config, logger, and event-specific data
        """
        ...
