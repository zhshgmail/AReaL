"""Filter API for admission control.

This module defines the Filter interface and FilterContext for queue/cache admission control.
Filters are separate from event handlers and have their own context.
"""

from __future__ import annotations

import abc
from typing import Any

from areal.api.cli_args import InferenceEngineConfig


class FilterContext:
    """Context passed to filters for admission control decisions.

    This is separate from EventContext because filtering and event handling
    are different concerns. Filters make admission decisions, while event
    handlers respond to system events.

    Attributes
    ----------
    config : InferenceEngineConfig
        Training configuration
    logger : Any | None
        Logger instance for diagnostics
    """

    def __init__(
        self,
        config: InferenceEngineConfig,
        logger: Any | None = None,
    ):
        """Initialize filter context.

        Parameters
        ----------
        config : InferenceEngineConfig
            Training configuration
        logger : Any | None, optional
            Logger instance. Default is None.
        """
        self.config = config
        self.logger = logger


class Filter(abc.ABC):
    """Abstract base class for admission control filters.

    Filters decide whether an item should be accepted when added to
    a queue or cache. They can be registered on any QueueAPI or CacheAPI
    implementation.

    Filters own their dependencies (e.g., engine reference for StalenessFilter)
    and receive only generic FilterContext during admission checks.

    Use cases:
    - Reject over-stale samples (StalenessFilter)
    - Enforce capacity limits based on metadata
    - Validate data format
    - Apply custom admission policies

    Examples
    --------
    >>> class StalenessFilter(Filter):
    ...     def __init__(self, max_staleness: int, engine: InferenceEngine):
    ...         self.max_staleness = max_staleness
    ...         self.engine = engine  # Store dependency
    ...
    ...     def should_accept(self, item: Any, context: FilterContext) -> bool:
    ...         current_ver = self.engine.get_version()  # Use stored reference
    ...         staleness = self._calculate_staleness(item, current_ver, context.config)
    ...         if staleness > self.max_staleness:
    ...             context.logger.warning(f"Rejecting stale item: {staleness}")
    ...             return False
    ...         return True
    """

    @abc.abstractmethod
    def should_accept(self, item: Any, context: FilterContext) -> bool:
        """Check if item should be accepted.

        Parameters
        ----------
        item : Any
            Item to be added (typically TensorDict sample)
        context : FilterContext
            Context with config and logger

        Returns
        -------
        bool
            True if item should be accepted, False to reject

        Raises
        ------
        Exception
            Can raise exception to reject item with error message
        """
        ...
