"""Cache-specific proximal logprob recomputation handler.

This handler responds to cache events and recomputes proximal_t for stale samples
in the cache before policy updates. It delegates actual recomputation logic to
ProximalRecomputeLogic and only handles cache-specific message structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areal.api.cache_event_handler import CacheEventContext
from areal.api.event_api import EventType
from areal.core.handlers.proximal_recompute_logic import ProximalRecomputeLogic

if TYPE_CHECKING:
    pass


class CacheProximalRecomputer:
    """Cache event handler for recomputing proximal_t before policy updates.

    This handler is registered ON the cache (LocalCache) and responds to
    cache-specific events. It uses CacheEventContext (metadata only) to
    process cache items without directly accessing the internal list.

    The actual recomputation logic is delegated to ProximalRecomputeLogic.
    This class only handles cache-specific message structure.

    Examples
    --------
    >>> recomputer = CacheProximalRecomputer()
    >>> cache.register_event_handler(recomputer)
    >>>
    >>> # When EventPropagator fires, cache calls registered handlers
    >>> cache.on_event(global_context)
    """

    def on_cache_event(self, context: CacheEventContext) -> None:
        """Handle cache event by recomputing proximal_t for stale samples.

        Parameters
        ----------
        context : CacheEventContext
            Cache event context with metadata (not direct cache access)
        """
        # Only handle BEFORE_POLICY_UPDATE events
        if context.event_type != EventType.BEFORE_POLICY_UPDATE:
            return

        # Check if engine supports recompute
        if not hasattr(context.engine, "recompute_output_logprobs_sync"):
            if context.logger:
                context.logger.debug(
                    "[CacheProximalRecomputer] Engine does not support recompute, skipping"
                )
            return

        current_ver = context.engine.get_version()

        # Create recompute logic instance
        recompute_logic = ProximalRecomputeLogic(context.engine, context.logger)

        # Get process_items method from metadata
        process_items = context.cache_metadata.get("process_items")
        if process_items is None:
            if context.logger:
                context.logger.warning(
                    "[CacheProximalRecomputer] No process_items method in context"
                )
            return

        # Process all items in cache using the provided method
        total_recomputed = process_items(
            lambda item, idx: recompute_logic.recompute_sample(
                item, current_ver, f"cache#{idx}"
            )
        )

        if total_recomputed > 0 and context.logger:
            context.logger.info(
                f"[CacheProximalRecomputer] Recomputed {total_recomputed} tokens "
                f"in cache at version {current_ver}"
            )
