"""Staleness filter transformer for segment-wise decoupled PPO.

This transformer removes samples that exceed the maximum allowed staleness
threshold, ensuring training stability with asynchronous updates.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, List

import torch
from tensordict import TensorDict

from areal.core.queue_transformer import QueueTransformer, TransformerContext

if TYPE_CHECKING:
    from areal.api.cli_args import InferenceEngineConfig

# Key for tracking recompute version
RECOMPUTE_VERSION_KEY = "_recompute_version"


class StalenessFilter(QueueTransformer):
    """Filter out samples exceeding staleness threshold.

    This transformer removes samples whose tokens are too far behind the current
    policy version. It implements the staleness control logic for segment-wise
    decoupled PPO.

    Staleness calculation:
    - For non-recomputed samples: staleness = current_ver - max(token_versions)
    - For recomputed samples: staleness = current_ver - min(token_versions)
    - A sample is dropped if: staleness > max_staleness

    This is a **filter** - it returns a new list containing only non-stale samples.
    """

    def __init__(self, max_staleness: int):
        """Initialize staleness filter.

        Parameters
        ----------
        max_staleness : int
            Maximum allowed staleness (typically max_head_offpolicyness from config)
        """
        self.max_staleness = max_staleness

    def apply(
        self, items: List[TensorDict], context: TransformerContext
    ) -> List[TensorDict]:
        """Filter out stale samples.

        Parameters
        ----------
        items : List[TensorDict]
            Samples to filter
        context : TransformerContext
            Context with engine, config, logger

        Returns
        -------
        List[TensorDict]
            New list containing only non-stale samples
        """
        current_ver = context.engine.get_version()
        filtered = []
        dropped = 0
        dropped_details = []

        for idx, td in enumerate(items):
            try:
                if self._is_sample_too_stale(td, current_ver, context.config):
                    dropped += 1
                    # Collect details for logging
                    staleness_info = self._get_staleness_info(td, current_ver, context.config)
                    dropped_details.append(f"sample#{idx}:{staleness_info}")
                else:
                    filtered.append(td)
            except Exception:
                context.logger.error(f"[StalenessFilter] Error checking sample #{idx}:")
                traceback.print_exc()
                # Keep sample on error to be safe
                filtered.append(td)

        if dropped > 0:
            context.logger.warning(
                f"[StalenessFilter] Dropped {dropped}/{len(items)} over-stale samples "
                f"at version {current_ver}. Details: {', '.join(dropped_details[:5])}"
                + (f" and {len(dropped_details) - 5} more..." if len(dropped_details) > 5 else "")
            )

        return filtered

    def _is_sample_too_stale(
        self,
        td: TensorDict,
        current_ver: int,
        config: "InferenceEngineConfig",
    ) -> bool:
        """Check if a sample exceeds staleness threshold.

        Parameters
        ----------
        td : TensorDict
            Sample to check
        current_ver : int
            Current model version
        config : InferenceEngineConfig
            Configuration with max_head_offpolicyness

        Returns
        -------
        bool
            True if sample should be dropped
        """
        try:
            staleness, allow_staleness, _ = self._calculate_staleness(
                td, current_ver, config
            )
            return staleness > allow_staleness
        except Exception:
            # On error, don't drop sample
            return False

    def _get_staleness_info(
        self,
        td: TensorDict,
        current_ver: int,
        config: "InferenceEngineConfig",
    ) -> str:
        """Get staleness info for logging."""
        try:
            staleness, allow_staleness, max_version = self._calculate_staleness(
                td, current_ver, config
            )
            return f"staleness={staleness},allow={allow_staleness},max_ver={max_version}"
        except Exception:
            return "error"

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
