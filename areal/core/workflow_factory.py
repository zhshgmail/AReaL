"""Factory for creating WorkflowExecutor with appropriate transformers.

This module provides a factory function that configures WorkflowExecutor
based on feature flags, implementing the dependency injection pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areal.core.staleness_manager import StalenessManager
from areal.core.transformers import ProximalRecomputer, StalenessFilter
from areal.core.workflow_executor import WorkflowExecutor

if TYPE_CHECKING:
    from areal.api.cli_args import InferenceEngineConfig
    from areal.api.engine_api import InferenceEngine


def create_workflow_executor(
    config: "InferenceEngineConfig",
    inference_engine: "InferenceEngine",
    staleness_manager: StalenessManager | None = None,
) -> WorkflowExecutor:
    """Factory to create WorkflowExecutor with appropriate transformers.

    This function implements the Spring @Configuration pattern - all component
    assembly happens here based on feature flags. This is the SINGLE SOURCE OF
    TRUTH for creating workflow executors.

    The factory configures different transformer chains based on whether
    segment-wise decoupled PPO is enabled:

    **Segment-wise PPO mode** (``enable_segment_wise_ppo=True``):
        - Pre-pause: ProximalRecomputer (recompute proximal_t for v-1 samples)
        - Pre-wait: StalenessFilter (remove over-stale samples)

    **Standard PPO mode** (``enable_segment_wise_ppo=False``):
        - No transformers (backward compatible behavior)

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
        Configured workflow executor with appropriate transformers

    Examples
    --------
    Create executor with segment-wise PPO enabled:

    >>> config = InferenceEngineConfig(enable_segment_wise_ppo=True)
    >>> engine = RemoteSGLangEngine(config)
    >>> executor = create_workflow_executor(config, engine)
    >>> # Executor has ProximalRecomputer and StalenessFilter

    Create executor with standard PPO:

    >>> config = InferenceEngineConfig(enable_segment_wise_ppo=False)
    >>> engine = RemoteSGLangEngine(config)
    >>> executor = create_workflow_executor(config, engine)
    >>> # Executor has no transformers (backward compatible)

    See Also
    --------
    WorkflowExecutor : Main executor class
    ProximalRecomputer : Transformer for recomputing proximal_t
    StalenessFilter : Transformer for filtering stale samples
    """
    enable_sdp = getattr(config, "enable_segment_wise_ppo", False)

    if enable_sdp:
        # Segment-wise decoupled PPO mode
        # Apply recompute before weight updates and filter stale samples
        pre_pause_transformers = [
            ProximalRecomputer(),  # Recompute proximal_t for v-1 samples
        ]

        pre_wait_transformers = [
            StalenessFilter(
                max_staleness=config.max_head_offpolicyness
            ),  # Filter over-stale samples
        ]

        return WorkflowExecutor(
            config=config,
            inference_engine=inference_engine,
            staleness_manager=staleness_manager,
            pre_pause_transformers=pre_pause_transformers,
            pre_wait_transformers=pre_wait_transformers,
        )
    else:
        # Standard PPO mode (backward compatible)
        # No transformers needed
        return WorkflowExecutor(
            config=config,
            inference_engine=inference_engine,
            staleness_manager=staleness_manager,
            pre_pause_transformers=[],
            pre_wait_transformers=[],
        )
