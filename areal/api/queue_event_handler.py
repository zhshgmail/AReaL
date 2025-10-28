"""Queue event handler protocol.

This module defines the interface for handling events within Queue implementations.
Queue-specific event handlers are registered ON the queue, not in the global
EventRegistry, maintaining encapsulation of queue internals.
"""

from __future__ import annotations

from typing import Any, Protocol


class QueueEventContext:
    """Context for queue-specific events.

    This context is created by the Queue implementation and passed to
    queue event handlers. It does NOT expose the queue's internal structure
    (e.g., queue.Queue), only the information handlers need.

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
    queue_metadata : dict
        Queue-specific metadata (size, items count, etc.)
        Does NOT include direct queue access
    """

    def __init__(
        self,
        event_type: Any,
        engine: Any,
        config: Any,
        logger: Any,
        queue_metadata: dict[str, Any] | None = None,
    ):
        self.event_type = event_type
        self.engine = engine
        self.config = config
        self.logger = logger
        self.queue_metadata = queue_metadata or {}


class QueueEventHandler(Protocol):
    """Protocol for handlers of queue-specific events.

    Queue event handlers are registered ON the queue implementation itself,
    not in the global EventRegistry. This maintains encapsulation - handlers
    work with QueueEventContext (which doesn't expose queue internals) rather
    than accessing queue.Queue directly.

    Examples
    --------
    >>> class QueueProximalRecomputer:
    ...     def on_queue_event(self, context: QueueEventContext):
    ...         # Work with context metadata, not direct queue access
    ...         if context.event_type == EventType.PRE_UPDATE:
    ...             items_to_recompute = context.queue_metadata['stale_items']
    ...             for item in items_to_recompute:
    ...                 self.recompute(item)
    """

    def on_queue_event(self, context: QueueEventContext) -> None:
        """Handle queue-specific event.

        Parameters
        ----------
        context : QueueEventContext
            Queue event context with metadata, not direct queue access
        """
        ...
