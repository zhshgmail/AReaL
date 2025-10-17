"""Components for workflow execution - staleness control and recomputation.

This module provides the pluggable components used by WorkflowExecutor:
- StalenessControlStrategy: Controls which samples are considered stale
- ProximalRecomputer: Recomputes proximal_t values for segment-wise PPO

These components are assembled by create_workflow_components() based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from areal.api.cli_args import InferenceEngineConfig
    from areal.api.engine_api import InferenceEngine

# Re-export components
from areal.api.proximal_recomputer import RECOMPUTE_VERSION_KEY, ProximalRecomputer
from areal.api.staleness_control import (
    _ensure_recompute_key,
    SegmentWisePPOStrategy,
    StalenessControlStrategy,
    StandardPPOStrategy,
)

__all__ = [
    "RECOMPUTE_VERSION_KEY",
    "ProximalRecomputer",
    "StalenessControlStrategy",
    "StandardPPOStrategy",
    "SegmentWisePPOStrategy",
    "_ensure_recompute_key",
    "create_workflow_components",
]


def create_workflow_components(
    config: "InferenceEngineConfig",
    inference_engine: "InferenceEngine",
    logger: Any,
) -> Tuple[StalenessControlStrategy, ProximalRecomputer | None]:
    """Factory function to create workflow components based on configuration.

    This function implements the Spring @Configuration pattern - all component
    assembly happens here based on feature flags. This is the SINGLE SOURCE OF
    TRUTH for creating components.

    Args:
        config: Training configuration
        inference_engine: Inference engine instance
        logger: Logger instance

    Returns:
        Tuple of (staleness_strategy, proximal_recomputer)
        - staleness_strategy: Always returned (StandardPPO or SegmentWisePPO)
        - proximal_recomputer: Only returned if segment-wise PPO is enabled, else None

    Example:
        >>> strategy, recomputer = create_workflow_components(config, engine, logger)
        >>> executor = WorkflowExecutor(config, engine, strategy, recomputer)
    """
    enable_sdp = getattr(config, "enable_segment_wise_ppo", False)

    if enable_sdp:
        # Segment-wise decoupled PPO mode
        strategy = SegmentWisePPOStrategy(config)
        recomputer = ProximalRecomputer(inference_engine, logger)
        logger.debug("Configured for segment-wise PPO: full staleness control + recomputation")
        return (strategy, recomputer)
    else:
        # Standard PPO mode (backward compatible)
        strategy = StandardPPOStrategy(config)
        recomputer = None
        logger.debug("Configured for standard PPO: no staleness filtering")
        return (strategy, recomputer)
