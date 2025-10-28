"""Factory for creating WorkflowExecutor with event-driven architecture.

This factory is the centralized configuration (Spring @Configuration equivalent)
that assembles all components: Queue, Cache, Filters, Event Handlers, and
wires them together based on feature flags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areal.api.event_api import EventContext, EventType
from areal.core.async_task_runner import AsyncTaskRunner
from areal.core.event_system import EventRegistry
from areal.core.filters import StalenessFilter
from areal.core.handlers.cache_proximal_recomputer import CacheProximalRecomputer
from areal.core.handlers.event_propagator import EventPropagator
from areal.core.handlers.queue_proximal_recomputer import QueueProximalRecomputer
from areal.core.local_cache import LocalCache
from areal.core.local_queue import LocalQueue
from areal.core.staleness_manager import StalenessManager
from areal.core.workflow_executor import WorkflowExecutor

if TYPE_CHECKING:
    from areal.api.cli_args import InferenceEngineConfig
    from areal.api.engine_api import InferenceEngine


def create_workflow_executor_with_events(
    config: "InferenceEngineConfig",
    inference_engine: "InferenceEngine",
    staleness_manager: StalenessManager | None = None,
) -> tuple[WorkflowExecutor, EventRegistry]:
    """Factory to create WorkflowExecutor with event-driven architecture.

    This function configures filters and event handlers based on feature flags:

    **Segment-wise PPO mode** (``enable_segment_wise_ppo=True``):
        - Filter: StalenessFilter (rejects over-stale samples at admission)
        - Handlers: QueueProximalRecomputer and CacheProximalRecomputer
          (recompute proximal_t on BEFORE_POLICY_UPDATE)
        - Propagator: EventPropagator (distributes global events to queue/cache)

    **Standard PPO mode** (``enable_segment_wise_ppo=False``):
        - No filters or handlers (backward compatible)

    Architecture:
        1. EventRegistry fires global event (BEFORE_POLICY_UPDATE)
        2. EventPropagator receives event and calls queue.on_event() and cache.on_event()
        3. LocalQueue/LocalCache create QueueEventContext/CacheEventContext with metadata
        4. Queue/Cache-specific handlers (QueueProximalRecomputer, CacheProximalRecomputer)
           receive events and process items without accessing internal structures

    Parameters
    ----------
    config : InferenceEngineConfig
        Training configuration with feature flags
    inference_engine : InferenceEngine
        Inference engine instance
    staleness_manager : StalenessManager | None, optional
        Optional staleness manager

    Returns
    -------
    tuple[WorkflowExecutor, EventRegistry]
        Configured executor and event registry

    Examples
    --------
    >>> config = InferenceEngineConfig(enable_segment_wise_ppo=True)
    >>> engine = RemoteSGLangEngine(config)
    >>> executor, registry = create_workflow_executor_with_events(config, engine)
    >>>
    >>> # Fire event before policy update
    >>> context = EventContext(EventType.BEFORE_POLICY_UPDATE, engine, config, logger)
    >>> registry.fire_event(context)  # Propagates to queue/cache handlers

    See Also
    --------
    EventRegistry : Manages global event handlers
    EventPropagator : Propagates events to queue/cache
    QueueProximalRecomputer : Queue-specific recomputation handler
    CacheProximalRecomputer : Cache-specific recomputation handler
    Filter : Protocol for admission control
    EventHandler : Protocol for global event handling
    """
    enable_sdp = getattr(config, "enable_segment_wise_ppo", False)

    # Create event registry
    registry = EventRegistry()

    # Determine queue size
    max_concurrent_rollouts = config.max_concurrent_rollouts or config.consumer_batch_size
    qsize = config.queue_size or max_concurrent_rollouts * 16

    if enable_sdp:
        # Segment-wise decoupled PPO mode
        # This is the Spring @Configuration - assemble all components here

        # 1. Create filter context (needed by filters)
        # Logger will be set during WorkflowExecutor.initialize()
        filter_context = EventContext(
            event_type=EventType.BEFORE_PAUSE,  # Placeholder
            engine=inference_engine,
            config=config,
            logger=None,  # Will be updated during initialize
        )

        # 2. Create LocalQueue and LocalCache with engine, config, logger for event support
        output_queue = LocalQueue(
            maxsize=qsize,
            filter_context=filter_context,
            engine=inference_engine,
            config=config,
            logger=None,  # Will be set during initialize
        )
        result_cache = LocalCache(
            filter_context=filter_context,
            engine=inference_engine,
            config=config,
            logger=None,  # Will be set during initialize
        )

        # 3. Create filter
        staleness_filter = StalenessFilter(
            max_staleness=config.max_head_offpolicyness
        )

        # 4. Register filters ON Queue and Cache
        output_queue.register_filter(staleness_filter)
        result_cache.register_filter(staleness_filter)

        # 5. Create queue-specific and cache-specific event handlers
        queue_recomputer = QueueProximalRecomputer()
        cache_recomputer = CacheProximalRecomputer()

        # 6. Register event handlers ON Queue and Cache
        output_queue.register_event_handler(queue_recomputer)
        result_cache.register_event_handler(cache_recomputer)

        # 7. Create EventPropagator for global event propagation
        event_propagator = EventPropagator(output_queue, result_cache)

        # 8. Register EventPropagator in global EventRegistry
        registry.register_handler(EventType.BEFORE_POLICY_UPDATE, event_propagator)

        # 9. Create AsyncTaskRunner with LocalQueue and LocalCache
        # AsyncTaskRunner now REQUIRES QueueAPI and CacheAPI (no defaults)
        runner = AsyncTaskRunner(
            max_queue_size=qsize,
            output_queue=output_queue,
            result_cache=result_cache,
            enable_tracing=config.enable_rollout_tracing,
        )

        # 10. Create WorkflowExecutor with configured runner
        executor = WorkflowExecutor(
            config=config,
            inference_engine=inference_engine,
            staleness_manager=staleness_manager,
            runner=runner,  # Pass configured runner
        )

        # 11. Connect engine with event registry so it can fire events
        inference_engine.event_registry = registry

    else:
        # Standard PPO mode - no filters or handlers
        # Just create plain executor with default queue/cache

        executor = WorkflowExecutor(
            config=config,
            inference_engine=inference_engine,
            staleness_manager=staleness_manager,
        )

        inference_engine.event_registry = None

    return executor, registry
