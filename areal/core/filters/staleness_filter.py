"""Staleness filter for queue admission control.

This filter rejects samples that are too stale when they are added to the
queue or cache, preventing over-stale data from entering the training pipeline.
"""

from __future__ import annotations

import traceback

import torch
from tensordict import TensorDict

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.filter_api import Filter, FilterContext

# Key for tracking recompute version
RECOMPUTE_VERSION_KEY = "_recompute_version"


class StalenessFilter(Filter):
    """Filter that rejects over-stale samples at queue/cache admission.

    This filter owns its dependency (engine reference) and only receives
    generic FilterContext during admission checks. This follows proper
    dependency injection principles.

    Staleness calculation:
    - For non-recomputed samples: staleness = current_ver - max(token_versions)
    - For recomputed samples: staleness = current_ver - min(token_versions)
    - A sample is rejected if: staleness > max_staleness

    This provides admission control at the queue/cache level, decoupled from
    workflow logic.

    Parameters
    ----------
    max_staleness : int
        Maximum allowed staleness (typically max_head_offpolicyness from config)
    engine : InferenceEngine
        Engine reference for getting current version (dependency injection)

    Examples
    --------
    >>> engine = RemoteSGLangEngine(config)
    >>> filter = StalenessFilter(max_staleness=2, engine=engine)
    >>> context = FilterContext(config=config, logger=logger)
    >>> if filter.should_accept(sample, context):
    ...     queue.put(sample)
    """

    def __init__(self, max_staleness: int, engine: InferenceEngine):
        """Initialize staleness filter with dependencies.

        Parameters
        ----------
        max_staleness : int
            Maximum allowed staleness
        engine : InferenceEngine
            Engine reference for getting current version (stored as dependency)
        """
        self.max_staleness = max_staleness
        self.engine = engine  # Store dependency

    def should_accept(self, item: TensorDict, context: FilterContext) -> bool:
        """Check if sample should be accepted based on staleness.

        Parameters
        ----------
        item : TensorDict
            Sample to check
        context : FilterContext
            Context with config and logger (no engine - we use self.engine)

        Returns
        -------
        bool
            True if sample should be accepted, False to reject
        """
        try:
            # Use stored engine reference (dependency injection)
            current_ver = self.engine.get_version()
            staleness, allow_staleness, max_version = self._calculate_staleness(
                item, current_ver, context.config
            )

            if staleness > allow_staleness:
                if context.logger:
                    context.logger.debug(
                        f"[StalenessFilter] Rejecting sample: staleness={staleness}, "
                        f"allow={allow_staleness}, max_ver={max_version}"
                    )
                return False

            return True

        except Exception:
            # On error, accept sample to be safe
            if context.logger:
                context.logger.error("[StalenessFilter] Error checking staleness:")
            traceback.print_exc()
            return True

    def _calculate_staleness(
        self,
        td: TensorDict,
        current_ver: int,
        config: InferenceEngineConfig,
    ) -> tuple[int, int, int]:
        """Calculate staleness metrics for a sample.

        Parameters
        ----------
        td : TensorDict
            Sample to analyze
        current_ver : int
            Current model version
        config : InferenceEngineConfig
            Training configuration

        Returns
        -------
        tuple[int, int, int]
            (staleness, allow_staleness, max_version)
        """
        versions = td.get("versions", None)
        loss_mask = td.get("loss_mask", None)
        recompute_version_tensor = td.get(RECOMPUTE_VERSION_KEY, None)

        if versions is None or loss_mask is None:
            return (0, self.max_staleness, -1)

        # Extract version and mask lists
        ver = versions[0].tolist() if torch.is_tensor(versions) else list(versions[0])
        lm = loss_mask[0].tolist() if torch.is_tensor(loss_mask) else list(loss_mask[0])

        # Get recompute version
        recompute_version = -1
        if recompute_version_tensor is not None:
            if torch.is_tensor(recompute_version_tensor):
                recompute_version = int(recompute_version_tensor[0, 0].item())
            else:
                recompute_version = int(recompute_version_tensor[0][0])

        # Find output token versions
        output_positions = [idx for idx, mask in enumerate(lm) if mask and idx < len(ver)]
        output_versions = [ver[idx] for idx in output_positions if ver[idx] >= 0]

        if not output_versions:
            return (0, self.max_staleness, -1)

        max_version = max(output_versions)
        min_version = min(output_versions)
        recomputed = recompute_version >= 0

        # Calculate staleness based on whether sample was recomputed
        if recomputed:
            # For recomputed samples, check head staleness (oldest token)
            staleness = current_ver - min_version
            # Allow more staleness for recomputed samples
            allow_staleness = max(
                self.max_staleness,
                int(getattr(config, "max_head_offpolicyness", self.max_staleness)),
            )
        else:
            # For non-recomputed samples, check tail staleness (newest token)
            staleness = current_ver - max_version
            allow_staleness = self.max_staleness

        return (staleness, allow_staleness, max_version)
