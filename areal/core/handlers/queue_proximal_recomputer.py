"""Queue-specific proximal logprob recomputation handler.

This handler responds to queue events and recomputes proximal_t for stale samples
in the queue before policy updates. It delegates actual recomputation logic to
ProximalRecomputeLogic and only handles queue-specific message structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areal.api.event_api import EventType
from areal.api.queue_event_handler import QueueEventContext
from areal.core.handlers.proximal_recompute_logic import ProximalRecomputeLogic

if TYPE_CHECKING:
    pass


class QueueProximalRecomputer:
    """Queue event handler for recomputing proximal_t before policy updates.

    This handler is registered ON the queue (LocalQueue) and responds to
    queue-specific events. It uses QueueEventContext (metadata only) to
    process queue items without directly accessing queue.Queue.

    The actual recomputation logic is delegated to ProximalRecomputeLogic.
    This class only handles queue-specific message structure.

    Examples
    --------
    >>> recomputer = QueueProximalRecomputer()
    >>> queue.register_event_handler(recomputer)
    >>>
    >>> # When EventPropagator fires, queue calls registered handlers
    >>> queue.on_event(global_context)
    """

    def on_queue_event(self, context: QueueEventContext) -> None:
        """Handle queue event by recomputing proximal_t for stale samples.

        Parameters
        ----------
        context : QueueEventContext
            Queue event context with metadata (not direct queue access)
        """
        # Only handle BEFORE_POLICY_UPDATE events
        if context.event_type != EventType.BEFORE_POLICY_UPDATE:
            return

        # Check if engine supports recompute
        if not hasattr(context.engine, "recompute_output_logprobs_sync"):
            if context.logger:
                context.logger.debug(
                    "[QueueProximalRecomputer] Engine does not support recompute, skipping"
                )
            return

        current_ver = context.engine.get_version()

        # Create recompute logic instance
        recompute_logic = ProximalRecomputeLogic(context.engine, context.logger)

        # Get process_items method from metadata
        process_items = context.queue_metadata.get("process_items")
        if process_items is None:
            if context.logger:
                context.logger.warning(
                    "[QueueProximalRecomputer] No process_items method in context"
                )
            return

        # Process all items in queue using the provided method
        total_recomputed = process_items(
            lambda item, idx: recompute_logic.recompute_sample(
                item, current_ver, f"queue#{idx}"
            )
        )

        if total_recomputed > 0 and context.logger:
            context.logger.info(
                f"[QueueProximalRecomputer] Recomputed {total_recomputed} tokens "
                f"in queue at version {current_ver}"
            )
