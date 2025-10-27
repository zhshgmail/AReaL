"""Base protocol and context for queue/cache transformers.

This module provides the foundation for the filter/transformer pattern used in
segment-wise decoupled PPO and other queue/cache processing tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Protocol

if TYPE_CHECKING:
    from areal.api.cli_args import InferenceEngineConfig
    from areal.api.engine_api import InferenceEngine


@dataclass
class TransformerContext:
    """Shared context for all transformers.

    This provides transformers with access to engine, configuration, and logging
    without tight coupling. The context can be updated dynamically as the system
    state changes.

    Attributes
    ----------
    engine : InferenceEngine
        The inference engine for accessing version, recompute methods, etc.
    config : InferenceEngineConfig
        Training configuration with feature flags and hyperparameters
    logger : Any
        Logger instance for diagnostics and debugging
    current_version : int | None, optional
        Current model version (can be updated dynamically)
    """

    engine: "InferenceEngine"
    config: "InferenceEngineConfig"
    logger: Any
    current_version: int | None = None


class QueueTransformer(Protocol):
    """Protocol for queue/cache transformers.

    Transformers are stateless, composable operations that can:
    - **Transform**: Modify items in-place (e.g., recompute proximal_t)
    - **Filter**: Remove items from the collection (e.g., drop stale samples)
    - **Validate**: Check items and log warnings (e.g., format validation)

    Transformers are applied at specific points in the workflow:
    - Pre-pause: Before weight updates (e.g., recompute proximal_t)
    - Pre-wait: Before returning samples to trainer (e.g., filter stale samples)

    Examples
    --------
    Transform operation (modifies in-place):

    >>> class ProximalRecomputer(QueueTransformer):
    ...     def apply(self, items, context):
    ...         for item in items:
    ...             self._recompute(item, context)
    ...         return items  # Same list, modified

    Filter operation (returns subset):

    >>> class StalenessFilter(QueueTransformer):
    ...     def apply(self, items, context):
    ...         return [item for item in items if not self._is_stale(item)]
    """

    def apply(
        self, items: List[Any], context: TransformerContext
    ) -> List[Any]:
        """Apply transformation/filtering to items.

        This method should be idempotent when possible - applying twice should
        have the same effect as applying once (for transformers, not filters).

        Parameters
        ----------
        items : List[Any]
            Items to process (typically TensorDict samples)
        context : TransformerContext
            Shared context with engine, config, logger

        Returns
        -------
        List[Any]
            Processed items. May be:
            - Same list with items modified in-place (transformer)
            - New list with subset of items (filter)
            - Same list unmodified (validator/no-op)
        """
        ...
