"""Shared proximal logprob recomputation logic.

This module provides the core recomputation logic that is shared between
QueueProximalRecomputer and CacheProximalRecomputer. It handles the actual
recomputation of proximal_t values without knowledge of queue/cache structure.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine

# Key for tracking recompute version
RECOMPUTE_VERSION_KEY = "_recompute_version"


def ensure_recompute_key(td: TensorDict) -> None:
    """Ensure the recompute tracking key exists with batch-aligned shape."""
    if not isinstance(td, TensorDict):
        return
    if RECOMPUTE_VERSION_KEY in td.keys():
        return
    if "versions" not in td.keys():
        return
    versions = td.get("versions")
    default_value = torch.full_like(versions[:, :1], -1, dtype=torch.int64)
    td.set(RECOMPUTE_VERSION_KEY, default_value)


class ProximalRecomputeLogic:
    """Shared logic for recomputing proximal_t values.

    This class encapsulates the core recomputation algorithm, independent of
    whether the sample comes from a queue or cache. Queue/Cache-specific
    handlers delegate to this class for the actual recomputation.

    Parameters
    ----------
    engine : InferenceEngine
        Engine for recomputing logprobs
    logger : Any
        Logger instance for diagnostics
    """

    def __init__(self, engine: "InferenceEngine", logger: Any):
        self.engine = engine
        self.logger = logger

    def recompute_sample(
        self,
        td: TensorDict,
        current_ver: int,
        sample_id: str = "",
    ) -> int:
        """Recompute proximal_t for a single sample.

        Parameters
        ----------
        td : TensorDict
            Sample to recompute
        current_ver : int
            Current model version
        sample_id : str
            Identifier for logging

        Returns
        -------
        int
            Number of tokens recomputed
        """
        try:
            # Extract required fields
            input_ids = td.get("input_ids", None)
            versions = td.get("versions", None)
            loss_mask = td.get("loss_mask", None)
            prox = td.get("proximal_logprobs_t", None)

            if (
                input_ids is None
                or versions is None
                or loss_mask is None
                or prox is None
            ):
                return 0

            # Convert to lists
            ids = input_ids[0].tolist()
            ver = versions[0].tolist()
            lm = loss_mask[0].tolist()

            # Get valid length
            attn_mask = td.get("attention_mask", None)
            valid_len = len(ids)

            try:
                if attn_mask is not None:
                    mask_row = attn_mask[0]
                    if torch.is_tensor(mask_row):
                        valid_len = min(valid_len, int(mask_row.sum().item()))
                    else:
                        valid_len = min(valid_len, int(sum(mask_row)))
            except Exception:
                traceback.print_exc()

            valid_len = min(valid_len, len(ver), len(lm))
            if valid_len <= 0:
                return 0

            # Find output positions
            lm_valid = lm[:valid_len]
            output_positions = [idx for idx, mask in enumerate(lm_valid) if mask]
            out_len = len(output_positions)
            if out_len == 0:
                return 0

            # Find positions needing recompute (version == current_ver - 1)
            need_positions = [
                (pos_idx, seq_idx)
                for pos_idx, seq_idx in enumerate(output_positions)
                if seq_idx < len(ver) and ver[seq_idx] == current_ver - 1
            ]

            if not need_positions:
                return 0

            # Calculate start_index
            first_output_idx = output_positions[0]
            start_index = max(0, first_output_idx - 1)

            # Recompute logprobs
            latest_out_logp = self.engine.recompute_output_logprobs_sync(
                input_ids=ids,
                start_index=start_index,
            )

            # Validate length
            max_required_offset = output_positions[-1] - start_index - 1
            if max_required_offset >= len(latest_out_logp):
                self.logger.warning(
                    f"[ProximalRecomputeLogic] {sample_id}: length mismatch, "
                    f"required idx {max_required_offset} but got {len(latest_out_logp)} logprobs"
                )
                return 0

            # Patch proximal_logprobs_t
            patched_here = 0
            for pos_idx, seq_idx in need_positions:
                rel_offset = seq_idx - start_index - 1
                if rel_offset < 0 or rel_offset >= len(latest_out_logp):
                    self.logger.warning(
                        f"[ProximalRecomputeLogic] {sample_id}: rel_offset={rel_offset} "
                        f"out of range for logprobs len={len(latest_out_logp)}"
                    )
                    continue
                prox[0, seq_idx] = float(latest_out_logp[rel_offset])
                patched_here += 1

            if patched_here == 0:
                return 0

            # Mark as recomputed
            ensure_recompute_key(td)
            patched_value = torch.full_like(
                versions[:, :1], int(current_ver), dtype=torch.int64
            )
            td.set(RECOMPUTE_VERSION_KEY, patched_value)

            return patched_here

        except Exception:
            self.logger.error(f"[ProximalRecomputeLogic] Error recomputing {sample_id}:")
            traceback.print_exc()
            return 0
