"""Factory for creating WorkflowExecutor with event-driven architecture.

This module provides a backward-compatible factory function that delegates to
the event_factory. This maintains compatibility with code that calls
create_workflow_executor() while using the new event-driven architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areal.core.event_factory import create_workflow_executor_with_events
from areal.core.staleness_manager import StalenessManager

if TYPE_CHECKING:
    from areal.api.cli_args import InferenceEngineConfig
    from areal.api.engine_api import InferenceEngine

    from areal.core.workflow_executor import WorkflowExecutor


def create_workflow_executor(
    config: "InferenceEngineConfig",
    inference_engine: "InferenceEngine",
    staleness_manager: StalenessManager | None = None,
) -> "WorkflowExecutor":
    """Factory to create WorkflowExecutor with event-driven architecture.

    This function delegates to create_workflow_executor_with_events() and
    returns only the executor for backward compatibility.

    The factory configures filters and event handlers based on whether
    segment-wise decoupled PPO is enabled:

    **Segment-wise PPO mode** (``enable_segment_wise_ppo=True``):
        - Filter: StalenessFilter (rejects over-stale samples at admission)
        - Handler: ProximalRecomputer (recomputes proximal_t on PRE_UPDATE event)

    **Standard PPO mode** (``enable_segment_wise_ppo=False``):
        - No filters or handlers (backward compatible behavior)

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
    >>> # Executor has StalenessFilter and ProximalRecomputer configured

    Create executor with standard PPO:

    >>> config = InferenceEngineConfig(enable_segment_wise_ppo=False)
    >>> engine = RemoteSGLangEngine(config)
    >>> executor = create_workflow_executor(config, engine)
    >>> # Executor has no filters or handlers (backward compatible)

    See Also
    --------
    create_workflow_executor_with_events : Event factory that returns (executor, registry)
    WorkflowExecutor : Main executor class
    EventRegistry : Manages event handlers
    """
    # Delegate to event factory and return only the executor
    executor, _registry = create_workflow_executor_with_events(
        config=config,
        inference_engine=inference_engine,
        staleness_manager=staleness_manager,
    )
    return executor
