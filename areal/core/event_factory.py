"""Factory for creating WorkflowExecutor with event-driven architecture.

This factory configures filters and event handlers based on feature flags,
providing clean extension points without modifying core workflow logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areal.core.event_system import EventRegistry, EventType
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

    # Create workflow executor
    executor = WorkflowExecutor(
        config=config,
        inference_engine=inference_engine,
        staleness_manager=staleness_manager,
    )

    if enable_sdp:
        # Segment-wise decoupled PPO mode

        # Create filter for admission control
        staleness_filter = StalenessFilter(
            max_staleness=config.max_head_offpolicyness
        )

        # Create handler for policy update events
        proximal_recomputer = ProximalRecomputer()

        # Register handler
        registry.register_handler(
            EventType.BEFORE_POLICY_UPDATE, proximal_recomputer
        )

        # Store filter and registry on executor for access
        executor._staleness_filter = staleness_filter
        executor._event_registry = registry

        # Connect engine with event registry so it can fire events
        inference_engine.event_registry = registry

        # Log configuration
        if hasattr(executor, "logger") and executor.logger:
            executor.logger.debug(
                "Configured for segment-wise PPO: "
                "StalenessFilter + ProximalRecomputer on BEFORE_POLICY_UPDATE"
            )
    else:
        # Standard PPO mode - no filters or handlers
        executor._staleness_filter = None
        executor._event_registry = registry
        inference_engine.event_registry = None

        if hasattr(executor, "logger") and executor.logger:
            executor.logger.debug("Configured for standard PPO (no filters/handlers)")

    return executor, registry
