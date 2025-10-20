"""Factory for creating and assembling workflow components.

This module provides centralized component assembly based on configuration,
following the Dependency Injection pattern. All business logic lives in
separate strategy/component classes, and this factory wires them together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from areal.api.cache_api import LocalRolloutCache, RolloutCache
from areal.api.proximal_recomputer import ProximalRecomputer
from areal.api.queue_api import LocalRolloutQueue, RolloutQueue
from areal.api.staleness_control import (
    SegmentWisePPOStrategy,
    StalenessControlStrategy,
    StandardPPOStrategy,
)
from areal.core.filtered_capacity_modifier import FilteredSamplesCapacityModifier

if TYPE_CHECKING:
    from areal.api.cli_args import InferenceEngineConfig
    from areal.api.engine_api import InferenceEngine
    from areal.api.workflow_api import WorkflowExecutor
    from areal.core.staleness_manager import StalenessManager


def create_queue(config: "InferenceEngineConfig") -> RolloutQueue:
    """Create rollout queue based on configuration.

    Args:
        config: Engine configuration

    Returns:
        Queue instance (currently LocalRolloutQueue, future: Ray/Redis)
    """
    # Calculate queue size
    max_concurrent_rollouts = (
        config.max_concurrent_rollouts or config.consumer_batch_size
    )
    qsize = config.queue_size or max_concurrent_rollouts * 16

    # Future: Could return RayRolloutQueue, RedisRolloutQueue based on config
    return LocalRolloutQueue(maxsize=qsize)


def create_cache(config: "InferenceEngineConfig") -> RolloutCache:
    """Create rollout cache based on configuration.

    Args:
        config: Engine configuration

    Returns:
        Cache instance (currently LocalRolloutCache, future: Ray/Redis)
    """
    # Future: Could return RayRolloutCache, RedisRolloutCache based on config
    return LocalRolloutCache()


def create_staleness_strategy(
    config: "InferenceEngineConfig",
) -> StalenessControlStrategy:
    """Create staleness control strategy based on configuration.

    Args:
        config: Engine configuration

    Returns:
        Staleness control strategy instance
    """
    if config.enable_segment_wise_ppo:
        return SegmentWisePPOStrategy(config)
    else:
        return StandardPPOStrategy(config)


def create_proximal_recomputer(
    inference_engine: "InferenceEngine",
    logger: Any,
    config: "InferenceEngineConfig",
) -> ProximalRecomputer | None:
    """Create proximal recomputer if needed.

    Args:
        inference_engine: Inference engine instance
        logger: Logger instance
        config: Engine configuration

    Returns:
        ProximalRecomputer instance if segment-wise PPO enabled, None otherwise
    """
    if config.enable_segment_wise_ppo:
        return ProximalRecomputer(inference_engine, logger)
    return None


def create_filtered_capacity_modifier(
    config: "InferenceEngineConfig",
) -> FilteredSamplesCapacityModifier | None:
    """Create filtered capacity modifier if needed.

    Args:
        config: Engine configuration

    Returns:
        Modifier instance if segment-wise PPO enabled, None otherwise
    """
    if config.enable_segment_wise_ppo:
        return FilteredSamplesCapacityModifier()
    return None


def register_capacity_modifiers(
    staleness_manager: "StalenessManager",
    filtered_capacity_modifier: FilteredSamplesCapacityModifier | None,
) -> None:
    """Register capacity modifiers with StalenessManager.

    Args:
        staleness_manager: Manager to register modifiers with
        filtered_capacity_modifier: Filtered capacity modifier (if any)
    """
    if filtered_capacity_modifier is not None:
        staleness_manager.register_capacity_modifier(filtered_capacity_modifier)


def create_workflow_executor(
    inference_engine: "InferenceEngine",
    staleness_manager: "StalenessManager",
    config: "InferenceEngineConfig",
    logger: Any,
) -> "WorkflowExecutor":
    """Create and configure WorkflowExecutor with all dependencies.

    This is the main factory method that assembles all components based on
    configuration and injects them into WorkflowExecutor.

    Design principle: "Extends over changes"
    - All feature-specific business logic lives in strategies/components
    - WorkflowExecutor is generic and depends on abstractions
    - Easy to disable feature via config (backward compatibility)

    Args:
        inference_engine: Inference engine instance
        staleness_manager: Staleness manager instance
        config: Engine configuration
        logger: Logger instance

    Returns:
        Fully configured WorkflowExecutor instance

    Example:
        >>> executor = create_workflow_executor(
        ...     inference_engine=engine,
        ...     staleness_manager=manager,
        ...     config=config,
        ...     logger=logger
        ... )
        >>> executor.start()
    """
    # Create queue and cache
    output_queue = create_queue(config)
    result_cache = create_cache(config)

    # Create staleness control strategy
    staleness_strategy = create_staleness_strategy(config)

    # Create proximal recomputer (if needed)
    proximal_recomputer = create_proximal_recomputer(inference_engine, logger, config)

    # Create and register capacity modifier (if needed)
    filtered_capacity_modifier = create_filtered_capacity_modifier(config)
    register_capacity_modifiers(staleness_manager, filtered_capacity_modifier)

    # Import here to avoid circular dependency
    from areal.api.workflow_api import WorkflowExecutor

    # Create WorkflowExecutor with injected dependencies
    executor = WorkflowExecutor(
        inference_engine=inference_engine,
        staleness_manager=staleness_manager,
        output_queue=output_queue,
        result_cache=result_cache,
        staleness_strategy=staleness_strategy,
        proximal_recomputer=proximal_recomputer,
        filtered_capacity_modifier=filtered_capacity_modifier,
        config=config,
        logger=logger,
    )

    return executor
