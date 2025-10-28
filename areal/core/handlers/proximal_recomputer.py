"""Proximal logprob recomputation event handler.

This handler recomputes proximal_t for stale samples in response to
BEFORE_POLICY_UPDATE events, ensuring correct importance weight calculations
for segment-wise decoupled PPO.
"""

from __future__ import annotations

import queue
import traceback
from typing import TYPE_CHECKING, Any, List

import torch
from tensordict import TensorDict

from areal.api.event_api import EventContext, EventHandler, EventType

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


class ProximalRecomputer(EventHandler):
    """Event handler for recomputing proximal_t before policy updates.

    This handler responds to BEFORE_POLICY_UPDATE events by scanning the
    queue and cache for samples with version v-1 tokens and recomputing
    their proximal_t values under the current policy.

    This is the core component of segment-wise decoupled PPO, ensuring that
    π_proximal_t = π_{v+1} for v-1 samples before the policy advances to v+1.

    The handler accesses the queue and cache through the event context data:
    - context.data['queue']: The output queue to scan
    - context.data['cache']: The result cache to scan

    Examples
    --------
    >>> recomputer = ProximalRecomputer()
    >>> registry.register_handler(EventType.BEFORE_POLICY_UPDATE, recomputer)
    >>>
    >>> # When policy update happens
    >>> context = EventContext(
    ...     EventType.BEFORE_POLICY_UPDATE,
    ...     engine, config, logger,
    ...     data={'queue': output_queue, 'cache': result_cache}
    ... )
    >>> registry.fire_event(context)  # Recomputer executes synchronously
    """

    def on_event(self, context: EventContext) -> None:
        """Handle BEFORE_POLICY_UPDATE event by recomputing proximal_t.

        Parameters
        ----------
        context : EventContext
            Event context with queue and cache in context.data
        """
        # Only handle BEFORE_POLICY_UPDATE events
        if context.event_type != EventType.BEFORE_POLICY_UPDATE:
            return

        # Check if engine supports recompute
        if not hasattr(context.engine, "recompute_output_logprobs_sync"):
            context.logger.debug(
                "[ProximalRecomputer] Engine does not support recompute, skipping"
            )
            return

        # Get queue and cache from event data
        output_queue = context.data.get("queue")
        result_cache = context.data.get("cache")

        if output_queue is None and result_cache is None:
            context.logger.warning(
                "[ProximalRecomputer] No queue or cache provided in event context"
            )
            return

        current_ver = context.engine.get_version()
        total_recomputed = 0

        # Recompute cache samples
        if result_cache is not None:
            cache_recomputed = self._recompute_cache(
                result_cache, current_ver, context.engine, context.logger
            )
            total_recomputed += cache_recomputed

        # Recompute queue samples
        if output_queue is not None:
            queue_recomputed = self._recompute_queue(
                output_queue, current_ver, context.engine, context.logger
            )
            total_recomputed += queue_recomputed

        if total_recomputed > 0:
            context.logger.info(
                f"[ProximalRecomputer] Recomputed {total_recomputed} tokens "
                f"at version {current_ver} before policy update"
            )

    def _recompute_cache(
        self,
        result_cache: List[TensorDict],
        current_ver: int,
        engine: "InferenceEngine",
        logger: Any,
    ) -> int:
        """Recompute proximal_t for samples in result cache.

        Parameters
        ----------
        result_cache : List[TensorDict]
            Cache containing samples
        current_ver : int
            Current model version
        engine : InferenceEngine
            Engine for recomputation
        logger : Any
            Logger instance

        Returns
        -------
        int
            Number of tokens recomputed
        """
        total_patched = 0
        try:
            for idx, td in enumerate(result_cache):
                patched = self._recompute_sample(
                    td, current_ver, engine, logger, f"cache#{idx}"
                )
                total_patched += patched
        except Exception:
            logger.error("[ProximalRecomputer] Error recomputing cache:")
            traceback.print_exc()
        return total_patched

    def _recompute_queue(
        self,
        output_queue: queue.Queue,
        current_ver: int,
        engine: "InferenceEngine",
        logger: Any,
    ) -> int:
        """Recompute proximal_t for samples in output queue.

        Uses drain-process-putback strategy to avoid blocking background thread.

        Parameters
        ----------
        output_queue : queue.Queue
            Queue containing pending outputs
        current_ver : int
            Current model version
        engine : InferenceEngine
            Engine for recomputation
        logger : Any
            Logger instance

        Returns
        -------
        int
            Number of tokens recomputed
        """
        total_patched = 0
        max_iterations = 3

        try:
            for iteration in range(max_iterations):
                # Drain queue
                temp_samples = []
                while True:
                    try:
                        sample = output_queue.get_nowait()
                        temp_samples.append(sample)
                    except queue.Empty:
                        break

                if not temp_samples:
                    break

                # Process samples
                for idx, td in enumerate(temp_samples):
                    try:
                        patched = self._recompute_sample(
                            td, current_ver, engine, logger, f"queue#{idx}"
                        )
                        total_patched += patched
                    except Exception:
                        traceback.print_exc()

                # Put samples back
                for sample in temp_samples:
                    try:
                        output_queue.put_nowait(sample)
                    except queue.Full:
                        try:
                            output_queue.put(sample, timeout=1.0)
                        except queue.Full:
                            logger.error(
                                "[ProximalRecomputer] Queue full during putback, sample dropped!"
                            )

                logger.debug(
                    f"[ProximalRecomputer] Iteration {iteration + 1}: "
                    f"processed {len(temp_samples)} queue samples"
                )
        except Exception:
            logger.error("[ProximalRecomputer] Error recomputing queue:")
            traceback.print_exc()

        return total_patched

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
            latest_out_logp = engine.recompute_output_logprobs_sync(
                input_ids=ids,
                start_index=start_index,
            )

            # Validate length
            max_required_offset = output_positions[-1] - start_index - 1
            if max_required_offset >= len(latest_out_logp):
                logger.warning(
                    f"[ProximalRecomputer] {sample_id}: length mismatch, "
                    f"required idx {max_required_offset} but got {len(latest_out_logp)} logprobs"
                )
                return 0

            # Patch proximal_logprobs_t
            patched_here = 0
            for pos_idx, seq_idx in need_positions:
                rel_offset = seq_idx - start_index - 1
                if rel_offset < 0 or rel_offset >= len(latest_out_logp):
                    logger.warning(
                        f"[ProximalRecomputer] {sample_id}: rel_offset={rel_offset} "
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
            logger.error(f"[ProximalRecomputer] Error recomputing {sample_id}:")
            traceback.print_exc()
            return 0
