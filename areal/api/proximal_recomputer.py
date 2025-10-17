"""Proximal logprob recomputation for segment-wise decoupled PPO.

This module contains the logic for recomputing proximal_t values for stale samples
before model weight updates. This is a critical component of segment-wise PPO to
ensure correct importance weight calculations.
"""

from __future__ import annotations

import queue
import traceback
from typing import TYPE_CHECKING, Any, List

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine

RECOMPUTE_VERSION_KEY = "_recompute_version"


class ProximalRecomputer:
    """Handles recomputation of proximal_t for segment-wise decoupled PPO.

    This component is responsible for updating the proximal_logprobs_t field
    in samples that have version v-1 tokens. It should be called RIGHT BEFORE
    weight updates to ensure all in-progress rollouts are still at current version.
    """

    def __init__(self, inference_engine: "InferenceEngine", logger: Any):
        """Initialize the recomputer.

        Args:
            inference_engine: Inference engine for logprob recomputation
            logger: Logger instance for diagnostics
        """
        self.inference_engine = inference_engine
        self.logger = logger

    def recompute_all(
        self, output_queue: queue.Queue, result_cache: List[TensorDict]
    ) -> None:
        """Recompute proximal_t for all v-1 samples before weight update.

        This should be called RIGHT BEFORE update_weights() to ensure:
        1. All in-progress rollouts are still at current version
        2. All v-1 samples (in queue or cache) get recomputed
        3. No samples miss their recompute window

        Processes BOTH output_queue and result_cache.

        Args:
            output_queue: Queue containing pending rollout outputs
            result_cache: Cache containing samples waiting to be returned
        """
        current_ver = self.inference_engine.get_version()

        # 1. Recompute samples in result_cache
        cache_recomputed = self._recompute_cache_proximal_t(result_cache, current_ver)

        # 2. Recompute samples in output_queue
        queue_recomputed = self._recompute_queue_proximal_t(output_queue, current_ver)

        total = cache_recomputed + queue_recomputed
        if total > 0:
            self.logger.info(
                f"[Recompute] Total recomputed: {total} "
                f"(cache: {cache_recomputed}, queue: {queue_recomputed}) at version {current_ver}"
            )

    def _recompute_cache_proximal_t(
        self, result_cache: List[TensorDict], current_ver: int
    ) -> int:
        """Recompute proximal_t for samples in result_cache.

        Args:
            result_cache: Cache containing samples
            current_ver: Current model version

        Returns:
            Number of tokens recomputed
        """
        total_patched = 0
        try:
            if hasattr(self.inference_engine, "recompute_output_logprobs_sync"):
                for idx, td in enumerate(result_cache):
                    patched = self._recompute_sample_proximal_t(
                        td, current_ver, f"cache#{idx}"
                    )
                    total_patched += patched
        except Exception:
            traceback.print_exc()
        return total_patched

    def _recompute_queue_proximal_t(
        self, output_queue: queue.Queue, current_ver: int
    ) -> int:
        """Recompute proximal_t for samples in output_queue.

        Uses drain-process-putback strategy to avoid blocking background thread.
        Multiple iterations ensure eventual consistency even with concurrent puts.

        Args:
            output_queue: Queue containing pending outputs
            current_ver: Current model version

        Returns:
            Number of tokens recomputed
        """
        if not hasattr(self.inference_engine, "recompute_output_logprobs_sync"):
            return 0

        total_patched = 0
        max_iterations = 3

        try:
            for iteration in range(max_iterations):
                # Drain queue into temporary list (queue.get_nowait is thread-safe)
                temp_samples = []
                while True:
                    try:
                        sample = output_queue.get_nowait()
                        temp_samples.append(sample)
                    except queue.Empty:
                        break

                if not temp_samples:
                    break  # Queue empty, done

                # Process samples (no lock needed - working on local list)
                for idx, td in enumerate(temp_samples):
                    try:
                        patched = self._recompute_sample_proximal_t(
                            td, current_ver, f"queue#{idx}"
                        )
                        total_patched += patched
                    except Exception:
                        traceback.print_exc()
                        # Keep sample even if recompute fails

                # Put samples back into queue (queue.put_nowait is thread-safe)
                for sample in temp_samples:
                    try:
                        output_queue.put_nowait(sample)
                    except queue.Full:
                        # Queue full, use blocking put with timeout
                        try:
                            output_queue.put(sample, timeout=1.0)
                        except queue.Full:
                            self.logger.error(
                                "[Recompute] Queue full during put-back, sample dropped!"
                            )

                self.logger.debug(
                    f"[Recompute] Iteration {iteration + 1}: processed {len(temp_samples)} samples from queue"
                )
        except Exception:
            traceback.print_exc()

        return total_patched

    def _recompute_sample_proximal_t(
        self, td: TensorDict, current_ver: int, sample_id: str = ""
    ) -> int:
        """Recompute proximal_t for a single sample.

        Args:
            td: TensorDict sample
            current_ver: Current model version
            sample_id: Identifier for logging

        Returns:
            Number of tokens recomputed
        """
        try:
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

            ids = input_ids[0].tolist()
            ver = versions[0].tolist()
            lm = loss_mask[0].tolist()
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

            lm_valid = lm[:valid_len]
            output_positions = [idx for idx, mask in enumerate(lm_valid) if mask]
            out_len = len(output_positions)
            if out_len == 0:
                return 0

            first_output_idx = output_positions[0]
            start_index = max(0, first_output_idx - 1)
            need_positions = [
                (pos_idx, seq_idx)
                for pos_idx, seq_idx in enumerate(output_positions)
                if ver[seq_idx] == current_ver - 1
            ]

            if not need_positions:
                return 0

            # Log version histogram for debugging
            try:
                seg = [ver[pos] for pos in output_positions]
                hist = {}
                for v in seg:
                    hist[v] = hist.get(v, 0) + 1
                hist_items = sorted(hist.items())
                self.logger.debug(
                    f"[Recompute] {sample_id}: version_hist={dict(hist_items)}"
                )
            except Exception:
                traceback.print_exc()

            latest_out_logp = self.inference_engine.recompute_output_logprobs_sync(
                input_ids=ids,
                start_index=start_index,
            )

            patched_here = 0
            max_required_offset = output_positions[-1] - start_index - 1
            if max_required_offset >= len(latest_out_logp):
                self.logger.warning(
                    f"[Recompute] {sample_id}: length mismatch, required idx {max_required_offset} "
                    f"but got {len(latest_out_logp)} logprobs"
                )
                return 0

            for pos_idx, seq_idx in need_positions:
                rel_offset = seq_idx - start_index - 1
                if rel_offset < 0 or rel_offset >= len(latest_out_logp):
                    self.logger.warning(
                        f"[Recompute] {sample_id}: rel_offset={rel_offset} out_of_range "
                        f"for logprobs len={len(latest_out_logp)}"
                    )
                    continue
                prox[0, seq_idx] = float(latest_out_logp[rel_offset])
                patched_here += 1

            if patched_here == 0:
                return 0

            patched_value = torch.full_like(
                versions[:, :1], int(current_ver), dtype=torch.int64
            )
            td.set(RECOMPUTE_VERSION_KEY, patched_value)
            return patched_here
        except Exception:
            traceback.print_exc()
            return 0
