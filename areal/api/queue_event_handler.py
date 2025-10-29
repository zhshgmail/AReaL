"""Queue event handler base class.

This module defines the interface for handling events within Queue implementations.
Queue-specific event handlers are registered ON the queue, not in the global
EventRegistry, maintaining encapsulation of queue internals.
"""

from __future__ import annotations

import abc
from typing import Any

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.event_api import EventContext, EventType


class QueueEventContext(EventContext):
    """Context for queue-specific events.

    This context extends EventContext with queue-specific metadata.
    It is created by the Queue implementation and passed to queue event handlers.
    It does NOT expose the queue's internal structure (e.g., queue.Queue),
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
    queue_metadata : dict[str, Any]
        Queue-specific metadata (size, empty, full status, process_items function)
        Does NOT include direct queue access
    """

    def __init__(
        self,
        event_type: EventType,
        engine: InferenceEngine,
        config: InferenceEngineConfig,
        logger: Any,
        queue_metadata: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ):
        """Initialize queue event context.

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
        queue_metadata : dict[str, Any] | None, optional
            Queue-specific metadata. Default is None.
        data : dict[str, Any] | None, optional
            Event-specific data. Default is None.
        """
        super().__init__(event_type, engine, config, logger, data)
        self.queue_metadata = queue_metadata or {}


class QueueEventHandler(abc.ABC):
    """Abstract base class for handlers of queue-specific events.

    Queue event handlers are registered ON the queue implementation itself,
    not in the global EventRegistry. This maintains encapsulation - handlers
    work with QueueEventContext (which doesn't expose queue internals) rather
    than accessing queue.Queue directly.

    Examples
    --------
    >>> class QueueProximalRecomputer(QueueEventHandler):
    ...     def on_queue_event(self, context: QueueEventContext):
    ...         # Work with context metadata, not direct queue access
    ...         if context.event_type == EventType.PRE_UPDATE:
    ...             items_to_recompute = context.queue_metadata['stale_items']
    ...             for item in items_to_recompute:
    ...                 self.recompute(item)
    """

    @abc.abstractmethod
    def on_queue_event(self, context: QueueEventContext) -> None:
        """Handle queue-specific event.

        Parameters
        ----------
        context : QueueEventContext
            Queue event context with metadata, not direct queue access
        """
        pass
