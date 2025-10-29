"""Factory for creating WorkflowExecutor with proper dependency injection.

This module provides two factories following Spring @Configuration pattern:

1. **create_workflow_executor_with_events()** - Generic factory
   - ALWAYS creates AsyncTaskRunner with explicit Queue/Cache instances
   - Segment-wise PPO enabled: Creates runner WITH filters/handlers
   - Segment-wise PPO disabled: Creates runner WITHOUT filters/handlers
   - Takes optional EventRegistry for custom event logic injection
   - Both cases explicitly show what components are created

2. **create_workflow_executor()** - Business-specific factory
   - Creates EventRegistry with business-specific handlers (proximal recomputers)
   - Calls generic factory with the registry
   - Recommended for production use with segment-wise decoupled PPO

Design Principles:
- WorkflowExecutor NEVER knows about concrete Queue/Cache types (LocalQueue/LocalCache)
- Factory has FULL control over what gets injected (proper dependency injection)
- Both enabled/disabled cases are EXPLICIT about what's created (@Configuration pattern)
- Clear separation of concerns: WorkflowExecutor uses interfaces, Factory creates instances
"""

from __future__ import annotations

from typing import Any, List, Protocol

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
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


class InferenceEngineProtocol(Protocol):
    """Protocol defining the interface required by workflow factory.

    This protocol specifies the minimal interface that an inference engine must
    implement to work with the workflow factory. It uses structural subtyping
    (duck typing) rather than inheritance, allowing RemoteInfEngine and SGLangEngine
    to work with the factory without inheriting from InferenceEngine.

    Required attributes and methods:
    - event_registry: Optional event registry for event-driven features
    - get_version(): Returns current policy version
    - recompute_output_logprobs_sync(): Recomputes logprobs under current policy
    """

    event_registry: EventRegistry | None

    def get_version(self) -> int:
        """Get the current policy version."""
        ...

    def recompute_output_logprobs_sync(
        self,
        input_ids: List[int],
        start_index: int,
        image_data: List[Any] | None = None,
    ) -> List[float]:
        """Recompute logprobs for output tokens under current policy.

        Parameters
        ----------
        input_ids : List[int]
            Full sequence including prompt and outputs
        start_index : int
            Index to start computing logprobs from
        image_data : List[Any] | None, optional
            Optional image data for VLM models

        Returns
        -------
        List[float]
            Logprobs for tokens after start_index
        """
        ...


def create_workflow_executor_with_events(
    config: InferenceEngineConfig,
    inference_engine: InferenceEngineProtocol,
    staleness_manager: StalenessManager | None = None,
    registry: EventRegistry | None = None,
) -> tuple[WorkflowExecutor, EventRegistry | None]:
    """Generic factory to create WorkflowExecutor with proper dependency injection.

    This factory ALWAYS creates AsyncTaskRunner with explicit Queue/Cache instances,
    following Spring @Configuration pattern. The factory has full control over what
    components are created and injected into WorkflowExecutor.

    **Registry Provided** (``registry is not None``):
        - Creates: input_queue (plain), output_queue (with filters), result_cache (with filters)
        - Adds: StalenessFilter to output_queue and result_cache (admission control)
        - Adds: Event handlers (QueueProximalRecomputer, CacheProximalRecomputer)
        - Creates: AsyncTaskRunner with configured queues/cache
        - Injects: Runner into WorkflowExecutor (DI)
        - Sets: inference_engine.event_registry = registry

    **No Registry** (``registry is None``):
        - Creates: input_queue, output_queue, result_cache (all plain, NO filters)
        - NO filters, NO event handlers (plain behavior)
        - Creates: AsyncTaskRunner with plain queues/cache
        - Injects: Runner into WorkflowExecutor (DI)
        - Sets: inference_engine.event_registry = None

    Both cases are EXPLICIT about what's created - this is the @Configuration pattern.
    WorkflowExecutor NEVER knows about concrete types (LocalQueue/LocalCache).

    The generic factory does NOT check business feature flags (like enable_segment_wise_ppo).
    It only checks if registry is provided. The business-specific factory (create_workflow_executor)
    decides whether to pass a registry based on business logic.

    Architecture when registry provided:
        1. EventRegistry fires global event (BEFORE_POLICY_UPDATE)
        2. EventPropagator receives event and calls queue.on_event() and cache.on_event()
        3. LocalQueue/LocalCache create QueueEventContext/CacheEventContext with metadata
        4. Queue/Cache-specific handlers receive events and process items

    Parameters
    ----------
    config : InferenceEngineConfig
        Training configuration with feature flags
    inference_engine : InferenceEngine
        Inference engine instance
    staleness_manager : StalenessManager | None, optional
        Optional staleness manager
    registry : EventRegistry | None, optional
        Optional event registry. If provided, event handlers will be created and
        registered. If None, only filters will be configured (no events).
        Default is None.

    Returns
    -------
    tuple[WorkflowExecutor, EventRegistry | None]
        Configured executor and event registry (None if not provided)

    Examples
    --------
    Create with custom registry for custom event logic:

    >>> registry = EventRegistry()
    >>> registry.register_handler(EventType.BEFORE_POLICY_UPDATE, MyCustomHandler())
    >>> executor, _ = create_workflow_executor_with_events(config, engine, registry=registry)

    Create without events (filters only):

    >>> executor, _ = create_workflow_executor_with_events(config, engine, registry=None)

    See Also
    --------
    EventRegistry : Manages global event handlers
    EventPropagator : Propagates events to queue/cache
    Filter : Protocol for admission control
    EventHandler : Protocol for global event handling
    """
    # Determine queue size
    max_concurrent_rollouts = config.max_concurrent_rollouts or config.consumer_batch_size
    qsize = config.queue_size or max_concurrent_rollouts * 16

    # Generic factory logic: check if registry is provided (not business feature flag)
    # Registry presence determines whether filters/handlers are needed
    if registry is not None:
        # Registry provided: Create with filters AND event handlers
        # Filters control admission, handlers process events

        # 1. Create input_queue, output_queue, and result_cache
        # All queues/caches use LocalQueue/LocalCache implementations
        # Queues/Caches manage their own FilterContext internally
        # Logger will be set during WorkflowExecutor.initialize()

        # Input queue - no filters or event handlers (used for task submission only)
        input_queue = LocalQueue(
            maxsize=qsize,
            config=None,  # No config needed for input queue
            logger=None,
        )

        # Output queue - with staleness filter and optional recompute handler
        output_queue = LocalQueue(
            maxsize=qsize,
            config=config,
            logger=None,  # Will be set during initialize
        )

        # Result cache - with staleness filter and optional recompute handler
        result_cache = LocalCache(
            config=config,
            logger=None,  # Will be set during initialize
        )

        # 2. Create filter with engine dependency (dependency injection)
        staleness_filter = StalenessFilter(
            max_staleness=config.max_head_offpolicyness,
            engine=inference_engine,
        )

        # 3. Register filters ON output_queue and result_cache (not input_queue)
        output_queue.register_filter(staleness_filter)
        result_cache.register_filter(staleness_filter)

        # 4. Create and register event handlers (registry is guaranteed non-None here)
        # Create event handlers with engine dependency (dependency injection)
        queue_recomputer = QueueProximalRecomputer(engine=inference_engine)
        cache_recomputer = CacheProximalRecomputer(engine=inference_engine)

        # Register event handlers ON output_queue and result_cache (not input_queue)
        output_queue.register_event_handler(queue_recomputer)
        result_cache.register_event_handler(cache_recomputer)

        # Create EventPropagator for global event propagation
        event_propagator = EventPropagator(output_queue, result_cache)

        # Register EventPropagator in global EventRegistry
        registry.register_handler(EventType.BEFORE_POLICY_UPDATE, event_propagator)

        # Bind registry to engine for automatic EventContext creation
        # Registry will use these to construct EventContext when events are fired
        registry.bind_to_engine(
            engine=inference_engine,
            config=config,
            logger=None,  # Logger will be set during WorkflowExecutor.initialize()
        )

        # Connect engine with event registry so it can fire events
        inference_engine.event_registry = registry

        # 5. Create AsyncTaskRunner with all three QueueAPI/CacheAPI instances
        # AsyncTaskRunner now REQUIRES input_queue, output_queue, and result_cache
        runner = AsyncTaskRunner(
            max_queue_size=qsize,
            input_queue=input_queue,
            output_queue=output_queue,
            result_cache=result_cache,
            enable_tracing=config.enable_rollout_tracing,
        )

        # 6. Create WorkflowExecutor with configured runner
        executor = WorkflowExecutor(
            config=config,
            inference_engine=inference_engine,
            runner=runner,  # Pass configured runner with filters/handlers
            staleness_manager=staleness_manager,
        )

    else:
        # No registry: Create plain queues/cache WITHOUT filters/handlers
        # This follows @Configuration pattern: both cases explicitly show what's created

        # 1. Create plain LocalQueue and LocalCache WITHOUT any config
        # No filters, no event handlers - just plain queues/cache
        input_queue = LocalQueue(maxsize=qsize)
        output_queue = LocalQueue(maxsize=qsize)
        result_cache = LocalCache()

        # 2. Create AsyncTaskRunner with plain queues/cache
        runner = AsyncTaskRunner(
            max_queue_size=qsize,
            input_queue=input_queue,
            output_queue=output_queue,
            result_cache=result_cache,
            enable_tracing=config.enable_rollout_tracing,
        )

        # 3. Create WorkflowExecutor with plain runner (no filters/handlers)
        executor = WorkflowExecutor(
            config=config,
            inference_engine=inference_engine,
            runner=runner,  # Pass plain runner WITHOUT filters/handlers
            staleness_manager=staleness_manager,
        )

        # Disable event system (no registry provided)
        inference_engine.event_registry = None

    return executor, registry


def create_workflow_executor(
    config: InferenceEngineConfig,
    inference_engine: InferenceEngineProtocol,
    staleness_manager: StalenessManager | None = None,
) -> WorkflowExecutor:
    """Business-specific factory to create WorkflowExecutor with event-driven architecture.

    This function creates an EventRegistry with business-specific event handlers
    (QueueProximalRecomputer, CacheProximalRecomputer) and calls the generic factory
    create_workflow_executor_with_events() with the configured registry.

    This is the recommended factory for production use with segment-wise decoupled PPO,
    as it provides the complete event-driven architecture with proximal recomputation.

    **Segment-wise PPO mode** (``enable_segment_wise_ppo=True``):
        - Creates EventRegistry
        - Filter: StalenessFilter (rejects over-stale samples at admission)
        - Handlers: QueueProximalRecomputer and CacheProximalRecomputer
          (recompute proximal_t on BEFORE_POLICY_UPDATE)
        - Propagator: EventPropagator (distributes global events to queue/cache)

    **Standard PPO mode** (``enable_segment_wise_ppo=False``):
        - No registry, filters, or handlers (backward compatible behavior)

    Parameters
    ----------
    config : InferenceEngineConfig
        Training configuration with feature flags
    inference_engine : InferenceEngine
        Inference engine instance for generation and recomputation
    staleness_manager : StalenessManager | None, optional
        Optional staleness manager. If None, one will be created during
        executor initialization. Default is None.

    Returns
    -------
    WorkflowExecutor
        Configured workflow executor with event-driven architecture

    Examples
    --------
    Create executor with segment-wise PPO enabled:

    >>> config = InferenceEngineConfig(enable_segment_wise_ppo=True)
    >>> engine = RemoteSGLangEngine(config)
    >>> executor = create_workflow_executor(config, engine)
    >>> # Executor has EventRegistry with business-specific handlers

    Create executor with standard PPO:

    >>> config = InferenceEngineConfig(enable_segment_wise_ppo=False)
    >>> engine = RemoteSGLangEngine(config)
    >>> executor = create_workflow_executor(config, engine)
    >>> # Executor has no filters or handlers (backward compatible)

    See Also
    --------
    create_workflow_executor_with_events : Generic factory with optional registry injection
    WorkflowExecutor : Main executor class
    EventRegistry : Manages event handlers
    """
    enable_sdp = getattr(config, "enable_segment_wise_ppo", False)

    # Create EventRegistry with business logic if segment-wise PPO enabled
    registry = EventRegistry() if enable_sdp else None

    # Delegate to generic factory with the registry
    executor, _registry = create_workflow_executor_with_events(
        config=config,
        inference_engine=inference_engine,
        staleness_manager=staleness_manager,
        registry=registry,
    )
    return executor
