"""Staleness filter for queue admission control.

This filter rejects samples that are too stale when they are added to the
queue or cache, preventing over-stale data from entering the training pipeline.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from areal.api.event_api import EventContext
from areal.api.filter_api import Filter

if TYPE_CHECKING:
    from areal.api.cli_args import InferenceEngineConfig

# Key for tracking recompute version
RECOMPUTE_VERSION_KEY = "_recompute_version"


class StalenessFilter(Filter):
    """Filter that rejects over-stale samples at queue/cache admission.

    This is a Filter that checks staleness when items are added to
    queue or cache. It rejects samples exceeding the staleness threshold.

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

    Examples
    --------
    >>> filter = StalenessFilter(max_staleness=2)
    >>> context = EventContext(event_type, engine, config, logger)
    >>> if filter.should_accept(sample, context):
    ...     queue.put(sample)
    """

    def __init__(self, max_staleness: int):
        """Initialize staleness filter.

        Parameters
        ----------
        max_staleness : int
            Maximum allowed staleness
        """
        self.max_staleness = max_staleness

    def should_accept(self, item: TensorDict, context: EventContext) -> bool:
        """Check if sample should be accepted based on staleness.

        Parameters
        ----------
        item : TensorDict
            Sample to check
        context : EventContext
            Context with engine, config, logger

        Returns
        -------
        bool
            True if sample should be accepted, False to reject
        """
        try:
            current_ver = context.engine.get_version()
            staleness, allow_staleness, max_version = self._calculate_staleness(
                item, current_ver, context.config
            )

            if staleness > allow_staleness:
                context.logger.debug(
                    f"[StalenessFilter] Rejecting sample: staleness={staleness}, "
                    f"allow={allow_staleness}, max_ver={max_version}"
                )
                return False

            return True

        except Exception:
            # On error, accept sample to be safe
            context.logger.error("[StalenessFilter] Error checking staleness:")
            traceback.print_exc()
            return True

    def _calculate_staleness(
        self,
        td: TensorDict,
        current_ver: int,
        config: "InferenceEngineConfig",
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
