"""Proximal logprob recomputation transformer for segment-wise decoupled PPO.

This transformer recomputes proximal_t values for stale samples before model
weight updates, ensuring correct importance weight calculations.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any, List

import torch
from tensordict import TensorDict

from areal.core.queue_transformer import QueueTransformer, TransformerContext

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


class ProximalRecomputer(QueueTransformer):
    """Recompute proximal_t for samples with version v-1 tokens.

    This transformer updates the proximal_logprobs_t field for tokens that were
    generated at version v-1. It should be applied RIGHT BEFORE weight updates
    to ensure all samples get their proximal_t recomputed with the latest policy.

    The recompute operation:
    1. Finds tokens with version == current_version - 1
    2. Calls engine.recompute_output_logprobs_sync() to get new logprobs
    3. Patches proximal_logprobs_t[i] with new values
    4. Marks sample with _recompute_version = current_version

    This is a **transformer** - it modifies items in-place and returns the same list.
    """

    def apply(
        self, items: List[TensorDict], context: TransformerContext
    ) -> List[TensorDict]:
        """Recompute proximal_t for v-1 samples.

        Parameters
        ----------
        items : List[TensorDict]
            Samples to process (modified in-place)
        context : TransformerContext
            Context with engine, config, logger

        Returns
        -------
        List[TensorDict]
            Same list with items modified in-place
        """
        current_ver = context.engine.get_version()
        total_recomputed = 0

        # Check if engine supports recompute
        if not hasattr(context.engine, "recompute_output_logprobs_sync"):
            context.logger.debug(
                "[ProximalRecomputer] Engine does not support recompute_output_logprobs_sync, skipping"
            )
            return items

        try:
            for idx, td in enumerate(items):
                n_recomputed = self._recompute_sample(
                    td, current_ver, context.engine, context.logger, f"item#{idx}"
                )
                total_recomputed += n_recomputed
        except Exception:
            context.logger.error("[ProximalRecomputer] Error during recompute:")
            traceback.print_exc()

        if total_recomputed > 0:
            context.logger.info(
                f"[ProximalRecomputer] Recomputed {total_recomputed} tokens "
                f"across {len(items)} samples at version {current_ver}"
            )

        return items  # Same list, modified in-place

    def _recompute_sample(
        self,
        td: TensorDict,
        current_ver: int,
        engine: "InferenceEngine",
        logger: Any,
        sample_id: str = "",
    ) -> int:
        """Recompute proximal_t for a single sample.

        Parameters
        ----------
        td : TensorDict
            Sample to recompute
        current_ver : int
            Current model version
        engine : InferenceEngine
            Engine for recomputation
        logger : Any
            Logger instance
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

            # Convert to lists for processing
            ids = input_ids[0].tolist()
            ver = versions[0].tolist()
            lm = loss_mask[0].tolist()

            # Get valid length from attention mask if available
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

            # Find output positions (where loss_mask is True)
            lm_valid = lm[:valid_len]
            output_positions = [idx for idx, mask in enumerate(lm_valid) if mask]
            out_len = len(output_positions)
            if out_len == 0:
                return 0

            # Find positions that need recomputation (version == current_ver - 1)
            need_positions = [
                (pos_idx, seq_idx)
                for pos_idx, seq_idx in enumerate(output_positions)
                if seq_idx < len(ver) and ver[seq_idx] == current_ver - 1
            ]

            if not need_positions:
                return 0

            # Log version histogram for debugging
            try:
                seg = [ver[pos] for pos in output_positions if pos < len(ver)]
                hist = {}
                for v in seg:
                    hist[v] = hist.get(v, 0) + 1
                hist_items = sorted(hist.items())
                logger.debug(
                    f"[ProximalRecomputer] {sample_id}: version_hist={dict(hist_items)}, "
                    f"need_recompute={len(need_positions)}/{len(output_positions)}"
                )
            except Exception:
                traceback.print_exc()

            # Calculate start_index for recompute (one position before first output)
            first_output_idx = output_positions[0]
            start_index = max(0, first_output_idx - 1)

            # Call engine to recompute logprobs
            latest_out_logp = engine.recompute_output_logprobs_sync(
                input_ids=ids,
                start_index=start_index,
            )

            # Validate returned logprobs length
            max_required_offset = output_positions[-1] - start_index - 1
            if max_required_offset >= len(latest_out_logp):
                logger.warning(
                    f"[ProximalRecomputer] {sample_id}: length mismatch, required idx {max_required_offset} "
                    f"but got {len(latest_out_logp)} logprobs"
                )
                return 0

            # Patch proximal_logprobs_t for v-1 tokens
            patched_here = 0
            for pos_idx, seq_idx in need_positions:
                rel_offset = seq_idx - start_index - 1
                if rel_offset < 0 or rel_offset >= len(latest_out_logp):
                    logger.warning(
                        f"[ProximalRecomputer] {sample_id}: rel_offset={rel_offset} out_of_range "
                        f"for logprobs len={len(latest_out_logp)}"
                    )
                    continue
                prox[0, seq_idx] = float(latest_out_logp[rel_offset])
                patched_here += 1

            if patched_here == 0:
                return 0

            # Mark sample as recomputed
            ensure_recompute_key(td)
            patched_value = torch.full_like(
                versions[:, :1], int(current_ver), dtype=torch.int64
            )
            td.set(RECOMPUTE_VERSION_KEY, patched_value)

            return patched_here

        except Exception:
            logger.error(f"[ProximalRecomputer] Error recomputing sample {sample_id}:")
            traceback.print_exc()
            return 0
