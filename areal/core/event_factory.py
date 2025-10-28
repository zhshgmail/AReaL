"""Factory for creating WorkflowExecutor with event-driven architecture.

This factory is the centralized configuration (Spring @Configuration equivalent)
that assembles all components: Queue, Cache, Filters, Event Handlers, and
wires them together based on feature flags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areal.core.async_task_runner import AsyncTaskRunner
from areal.core.event_system import EventContext, EventRegistry, EventType
from areal.core.filterable_cache import FilterableCache
from areal.core.filterable_queue import FilterableQueue
from areal.core.filters import StalenessFilter
from areal.core.handlers import ProximalRecomputer
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
        - Handler: ProximalRecomputer (recomputes proximal_t on BEFORE_POLICY_UPDATE)

    **Standard PPO mode** (``enable_segment_wise_ppo=False``):
        - No filters or handlers (backward compatible)

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
    >>> context = EventContext(
    ...     EventType.BEFORE_POLICY_UPDATE, engine, config, logger,
    ...     data={'queue': queue, 'cache': cache}
    ... )
    >>> registry.fire_event(context)  # ProximalRecomputer runs

    See Also
    --------
    EventRegistry : Manages event handlers
    QueueFilter : Protocol for admission control
    EventHandler : Protocol for event handling
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

        # 2. Create FilterableQueue and FilterableCache with context
        output_queue = FilterableQueue(maxsize=qsize, filter_context=filter_context)
        result_cache = FilterableCache(filter_context=filter_context)

        # 3. Create filter
        staleness_filter = StalenessFilter(
            max_staleness=config.max_head_offpolicyness
        )

        # 4. Register filters ON Queue and Cache
        output_queue.register_filter(staleness_filter)
        result_cache.register_filter(staleness_filter)

        # 5. Create AsyncTaskRunner with FilterableQueue and FilterableCache
        runner = AsyncTaskRunner(
            max_queue_size=qsize,
            enable_tracing=config.enable_rollout_tracing,
            output_queue=output_queue,
            result_cache=result_cache,
        )

        # 6. Create WorkflowExecutor with configured runner
        executor = WorkflowExecutor(
            config=config,
            inference_engine=inference_engine,
            staleness_manager=staleness_manager,
            runner=runner,  # Pass configured runner
        )

        # 7. Store filter context for later use (will be updated with logger in initialize())
        executor._filter_context = filter_context

        # 8. Create event handler for policy updates
        proximal_recomputer = ProximalRecomputer()

        # 9. Register event handler
        registry.register_handler(
            EventType.BEFORE_POLICY_UPDATE, proximal_recomputer
        )

        # 10. Connect engine with event registry so it can fire events
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
