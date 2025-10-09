import asyncio
import itertools
import queue
import threading
import time
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import aiohttp
import torch
import torch.distributed as dist
import uvloop
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import RolloutStat
from areal.utils.data import concat_padded_tensors
from areal.utils.http import get_default_connector
from realhf.base import logging

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine

logger = logging.getLogger("areal.workflow_api")


RECOMPUTE_VERSION_KEY = "_recompute_version"


def _extract_version(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value)
        tensor = tensor.reshape(-1)
        if tensor.numel() == 0:
            return None
        return int(tensor[0].item())
    except Exception:
        try:
            return int(value)
        except Exception:
            return None


ROLLOUT_POLL_WAIT_TIME = 0.05


def _ensure_recompute_key(td: TensorDict) -> None:
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


class RolloutWorkflow:

    async def arun_episode(
        self, engine: "InferenceEngine", data: Dict[str, Any]
    ) -> TensorDict | None:
        """Run a single episode of the workflow.

        `None` implies that this trajectory is rejected and will not be used for training.

        See concrete example implementations under the `areal/workflow` directory.
        """
        raise NotImplementedError()


class WorkflowExecutor:

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: "InferenceEngine",
    ):
        config.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config
        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()

        self.inference_engine = inference_engine

        qsize = config.queue_size or config.max_concurrent_rollouts * 16
        self.input_queue = queue.Queue(maxsize=qsize)
        self.output_queue = queue.Queue(maxsize=qsize)
        self.result_cache: List[TensorDict] = []

        self.rollout_stat = RolloutStat()

        self.session = None
        # Count of dropped over-stale samples at output filtering
        self.drop_count = 0
        # Track last version purged from output_queue
        self._last_purged_ver = -1
        # Track recent effective samples returned to the trainer
        self._effective_history = deque(maxlen=16)
        self._effective_sum = 0.0

    def _calculate_staleness(
        self,
        versions: List[int],
        loss_mask: List[int],
        current_ver: int,
        recompute_version: int = -1,
    ) -> tuple[int, int, int]:
        """Calculate staleness metrics for a sample.

        Args:
            versions: Version list from sample
            loss_mask: Loss mask list from sample
            current_ver: Current model version
            recompute_version: Recompute version if sample was recomputed

        Returns:
            Tuple of (staleness, allow_staleness, max_version)
        """
        output_positions = [idx for idx, mask in enumerate(loss_mask) if mask]
        output_versions = [versions[idx] for idx in output_positions if versions[idx] >= 0]

        if not output_versions:
            return (0, 1, -1)

        max_version = max(output_versions)
        min_version = min(output_versions)
        recomputed = recompute_version >= 0

        tail_staleness = current_ver - max_version
        head_staleness = current_ver - min_version if recomputed else None
        staleness = head_staleness if recomputed else tail_staleness

        allow_staleness = 1
        if recomputed and self.config.max_head_offpolicyness is not None:
            allow_staleness = max(allow_staleness, int(self.config.max_head_offpolicyness))

        return (staleness, allow_staleness, max_version)

    def _is_sample_too_stale(self, td: TensorDict, current_ver: int) -> bool:
        """Check if a sample exceeds staleness threshold.

        Args:
            td: TensorDict sample to check
            current_ver: Current model version

        Returns:
            True if sample should be dropped due to staleness
        """
        try:
            versions = td.get("versions", None)
            loss_mask = td.get("loss_mask", None)
            if versions is None or loss_mask is None:
                return False  # Keep samples without version info

            ver = versions[0].tolist()
            lm_row = loss_mask[0]
            if torch.is_tensor(lm_row):
                lm = lm_row.tolist()
            else:
                lm = list(lm_row)

            patched_ver = _extract_version(td.get(RECOMPUTE_VERSION_KEY, None))
            staleness, allow_staleness, _ = self._calculate_staleness(
                ver, lm, current_ver, patched_ver or -1
            )

            return staleness > allow_staleness
        except Exception:
            traceback.print_exc()
            return False  # Keep on error

    def _purge_stale_samples_from_queue(self, current_ver: int) -> None:
        """Drain output queue and drop stale samples when version increases.

        Args:
            current_ver: Current model version
        """
        if current_ver <= self._last_purged_ver:
            return  # Only purge when version increases

        drained = 0
        picked_prev = 0
        dropped = 0
        kept = 0
        v0_picked = 0
        v0_kept = 0
        v0_dropped = 0
        put_back_buf: List[TensorDict] = []

        # Drain entire queue
        while True:
            try:
                traj = self.output_queue.get_nowait()
            except queue.Empty:
                break

            drained += 1
            try:
                if self._is_sample_too_stale(traj, current_ver):
                    dropped += 1
                    # Track v0 statistics
                    versions = traj.get("versions", None)
                    if versions is not None:
                        ver = versions[0].tolist()
                        loss_mask = traj.get("loss_mask", None)
                        if loss_mask is not None:
                            lm = loss_mask[0].tolist() if torch.is_tensor(loss_mask[0]) else list(loss_mask[0])
                            output_positions = [idx for idx, mask in enumerate(lm) if mask]
                            if output_positions:
                                max_version = max([ver[idx] for idx in output_positions if ver[idx] >= 0], default=-1)
                                if max_version == 0:
                                    v0_dropped += 1
                else:
                    put_back_buf.append(traj)
                    kept += 1
                    # Track statistics for kept samples
                    versions = traj.get("versions", None)
                    if versions is not None:
                        ver = versions[0].tolist()
                        loss_mask = traj.get("loss_mask", None)
                        if loss_mask is not None:
                            lm = loss_mask[0].tolist() if torch.is_tensor(loss_mask[0]) else list(loss_mask[0])
                            output_positions = [idx for idx, mask in enumerate(lm) if mask]
                            output_versions = [ver[idx] for idx in output_positions if ver[idx] >= 0]
                            if output_versions:
                                max_version = max(output_versions)
                                if max_version == 0:
                                    v0_kept += 1
                                if (current_ver - 1) in output_versions:
                                    picked_prev += 1
                                    if max_version == 0:
                                        v0_picked += 1
            except Exception:
                put_back_buf.append(traj)  # Keep on error
                traceback.print_exc()

        # FIX BUG #5: Put items back with timeout to handle full queue
        for item in put_back_buf:
            try:
                _ensure_recompute_key(item)
                self.output_queue.put(item, timeout=1.0)
            except queue.Full:
                logger.error(
                    f"Output queue remains full after version purge. "
                    f"Queue size: {self.output_queue.qsize()}/{self.output_queue.maxsize}. "
                    f"Please increase queue_size."
                )
                raise RuntimeError(
                    "Output queue full when putting back after version purge. Please increase queue_size."
                )

        logger.info(
            f"[QueuePurge] ver_switch to {current_ver}: drained={drained} "
            f"picked_prev={picked_prev} dropped={dropped} kept={kept} "
            f"cache_size={len(self.result_cache)} "
            f"v0(picked/dropped/kept)={v0_picked}/{v0_dropped}/{v0_kept}"
        )
        self._last_purged_ver = current_ver

    def _collect_samples_from_queue(
        self,
        count: int,
        timeout: float,
        start_time: float,
        should_accept: Callable | None = None,
    ) -> int:
        """Collect samples from output queue into result_cache until reaching count.

        Args:
            count: Target number of samples in cache
            timeout: Overall timeout in seconds
            start_time: Start time from perf_counter()
            should_accept: Optional filter function

        Returns:
            Number of samples collected in this call
        """
        collected = 0
        while (
            len(self.result_cache) < count
            and not self.exiting.is_set()
            and time.perf_counter() - start_time < timeout
        ):
            try:
                result = self.output_queue.get(timeout=ROLLOUT_POLL_WAIT_TIME)
                if result is not None and (should_accept is None or should_accept(result)):
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Accept rollout result. cache_size/count = {len(self.result_cache)}/{count}"
                        )
                    result = result.clone()
                    self.result_cache.append(result)
                    collected += 1
                elif result is not None:
                    # FIX BUG #2: Decrement rollout_stat when should_accept rejects
                    if self.config.enable_rollout_tracing:
                        logger.info(f"Rollout is rejected by should_accept filter.")
                    with self.lock:
                        self.rollout_stat.accepted -= 1
            except queue.Empty:
                pass

        return collected

    def _filter_stale_from_cache(self, current_ver: int) -> int:
        """Remove stale samples from result_cache.

        Args:
            current_ver: Current model version

        Returns:
            Number of samples dropped
        """
        filtered: List[TensorDict] = []
        dropped_cache = 0

        for td in self.result_cache:
            if self._is_sample_too_stale(td, current_ver):
                dropped_cache += 1
            else:
                filtered.append(td)

        self.result_cache = filtered

        if dropped_cache:
            logger.info(f"[CacheFilter] dropped_cache={dropped_cache} size={len(self.result_cache)}")

        return dropped_cache

    def recompute_all_proximal_t(self) -> None:
        """Recompute proximal_t for all v-1 samples before weight update.

        This should be called RIGHT BEFORE update_weights() to ensure:
        1. All in-progress rollouts are still at current version
        2. All v-1 samples (in queue or cache) get recomputed
        3. No samples miss their recompute window

        Processes BOTH output_queue and result_cache.
        """
        current_ver = self.inference_engine.get_version()

        # 1. Recompute samples in result_cache
        cache_recomputed = self._recompute_cache_proximal_t(current_ver)

        # 2. Recompute samples in output_queue
        queue_recomputed = self._recompute_queue_proximal_t(current_ver)

        total = cache_recomputed + queue_recomputed
        if total > 0:
            logger.info(
                f"[Recompute] Total recomputed: {total} "
                f"(cache: {cache_recomputed}, queue: {queue_recomputed}) at version {current_ver}"
            )

    def _recompute_cache_proximal_t(self, current_ver: int) -> int:
        """Recompute proximal_t for samples in result_cache.

        Args:
            current_ver: Current model version

        Returns:
            Number of tokens recomputed
        """
        total_patched = 0
        try:
            if hasattr(self.inference_engine, "recompute_output_logprobs_sync"):
                for idx, td in enumerate(self.result_cache):
                    patched = self._recompute_sample_proximal_t(td, current_ver, f"cache#{idx}")
                    total_patched += patched
        except Exception:
            traceback.print_exc()
        return total_patched

    def _recompute_queue_proximal_t(self, current_ver: int) -> int:
        """Recompute proximal_t for samples in output_queue.

        Uses drain-process-putback strategy to avoid blocking background thread.
        Multiple iterations ensure eventual consistency even with concurrent puts.

        Args:
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
                        sample = self.output_queue.get_nowait()
                        temp_samples.append(sample)
                    except queue.Empty:
                        break

                if not temp_samples:
                    break  # Queue empty, done

                # Process samples (no lock needed - working on local list)
                for idx, td in enumerate(temp_samples):
                    try:
                        patched = self._recompute_sample_proximal_t(td, current_ver, f"queue#{idx}")
                        total_patched += patched
                    except Exception:
                        traceback.print_exc()
                        # Keep sample even if recompute fails

                # Put samples back into queue (queue.put_nowait is thread-safe)
                for sample in temp_samples:
                    try:
                        self.output_queue.put_nowait(sample)
                    except queue.Full:
                        # Queue full, use blocking put with timeout
                        try:
                            self.output_queue.put(sample, timeout=1.0)
                        except queue.Full:
                            logger.error("[Recompute] Queue full during put-back, sample dropped!")

                logger.debug(
                    f"[Recompute] Iteration {iteration + 1}: processed {len(temp_samples)} samples from queue"
                )
        except Exception:
            traceback.print_exc()

        return total_patched

    def _recompute_sample_proximal_t(self, td: TensorDict, current_ver: int, sample_id: str = "") -> int:
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
                logger.debug(
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
                logger.warning(
                    f"[Recompute] {sample_id}: length mismatch, required idx {max_required_offset} "
                    f"but got {len(latest_out_logp)} logprobs"
                )
                return 0

            for pos_idx, seq_idx in need_positions:
                rel_offset = seq_idx - start_index - 1
                if rel_offset < 0 or rel_offset >= len(latest_out_logp):
                    logger.warning(
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

    def _recompute_stale_logprobs(self, current_ver: int) -> None:
        """DEPRECATED: Use recompute_all_proximal_t() instead.

        This method is kept for backward compatibility but should not be used.
        The new approach is to call recompute_all_proximal_t() before weight updates.

        Args:
            current_ver: Current model version
        """
        try:
            total_candidates = 0
            total_patched = 0
            if hasattr(self.inference_engine, "recompute_output_logprobs_sync"):
                for idx, td in enumerate(self.result_cache):
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
                            continue

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
                            continue

                        lm_valid = lm[:valid_len]
                        output_positions = [idx for idx, mask in enumerate(lm_valid) if mask]
                        out_len = len(output_positions)
                        if out_len == 0:
                            continue

                        first_output_idx = output_positions[0]
                        start_index = max(0, first_output_idx - 1)
                        need_positions = [
                            (pos_idx, seq_idx)
                            for pos_idx, seq_idx in enumerate(output_positions)
                            if ver[seq_idx] == current_ver - 1
                        ]

                        if not need_positions:
                            continue

                        total_candidates += 1

                        # Log version histogram for debugging
                        try:
                            seg = [ver[pos] for pos in output_positions]
                            hist = {}
                            for v in seg:
                                hist[v] = hist.get(v, 0) + 1
                            hist_items = sorted(hist.items())
                            logger.info(
                                f"[Recompute] sample#{idx}: version_hist={dict(hist_items)}"
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
                            logger.warning(
                                f"[Recompute] sample#{idx}: length mismatch, required idx {max_required_offset} "
                                f"but got {len(latest_out_logp)} logprobs"
                            )
                            continue

                        for pos_idx, seq_idx in need_positions:
                            rel_offset = seq_idx - start_index - 1
                            if rel_offset < 0 or rel_offset >= len(latest_out_logp):
                                logger.warning(
                                    f"[Recompute] sample#{idx}: rel_offset={rel_offset} out_of_range "
                                    f"for logprobs len={len(latest_out_logp)}"
                                )
                                continue
                            prox[0, seq_idx] = float(latest_out_logp[rel_offset])
                            patched_here += 1

                        if patched_here == 0:
                            continue

                        patched_value = torch.full_like(
                            versions[:, :1], int(current_ver), dtype=torch.int64
                        )
                        td.set(RECOMPUTE_VERSION_KEY, patched_value)
                        total_patched += patched_here
                    except Exception:
                        traceback.print_exc()
                        continue
        except Exception:
            traceback.print_exc()

    def initialize(self):
        self.rollout_tasks: Dict[str, asyncio.Task] = {}
        self.rollout_thread = threading.Thread(
            target=self._rollout_thread, daemon=True
        )  # set daemon=True to automatically exit when error occurs
        self.rollout_thread.start()

    def destroy(self):
        self.exiting.set()
        self.rollout_thread.join()

    def get_capacity(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        with self.lock:
            max_concurrent_rollouts = max(
                1, self.config.max_concurrent_rollouts // world_size
            )
            capacity = max_concurrent_rollouts - len(self.rollout_tasks)
            # Staleness control
            version = self.inference_engine.get_version()
            ofp = self.config.max_head_offpolicyness
            if self._effective_history:
                effective_recent = self._effective_sum / len(self._effective_history)
            else:
                effective_recent = float(self.rollout_stat.accepted)
            sample_cnt = effective_recent + self.rollout_stat.running
            consumer_bs = max(1, self.config.consumer_batch_size // world_size)
            budget = (ofp + version + 1) * consumer_bs - sample_cnt
            budget_int = int(budget)
            capacity = min(capacity, max(0, budget_int))
        return capacity

    def _record_effective_samples(self, count: int) -> None:
        if count <= 0:
            return
        with self.lock:
            if len(self._effective_history) == self._effective_history.maxlen:
                self._effective_sum -= self._effective_history.popleft()
            self._effective_history.append(count)
            self._effective_sum += count

    def _rollout_thread(self):
        """Thread that runs the rollout loop."""
        try:
            uvloop.run(self._rollout_thread_async())
        except Exception:
            traceback.print_exc()

    async def _rollout_thread_async(self):
        if self.session is None:
            # NOTE: Lazily initialize aiohttp.ClientSession since it needs to be initialized
            # inside asyncio loop in WorkflowExecutor
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.config.request_timeout,
                    sock_connect=self.config.request_timeout,
                    connect=self.config.request_timeout,
                ),
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            )

        rollout_tasks = self.rollout_tasks
        rid = 0
        try:
            while not self.exiting.is_set():
                # Check capacity
                capacity = self.get_capacity()
                # Create new rollout task
                self.lock.acquire()
                while (
                    capacity > 0
                    and not self.paused.is_set()
                    and self.input_queue.qsize() > 0
                ):
                    data, workflow = self.input_queue.get_nowait()
                    logger.debug(f"Get data from puller: {data}")
                    task = asyncio.create_task(
                        workflow.arun_episode(self.inference_engine, data),
                        name=str(rid),
                    )
                    rollout_tasks[str(rid)] = task
                    self.rollout_stat.submitted += 1
                    self.rollout_stat.running += 1
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Submit rollout rid {rid}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}."
                        )
                    capacity -= 1
                    rid += 1
                tasks = list(rollout_tasks.values())
                self.lock.release()

                # Wait for rollout completion
                done = []
                if tasks:
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=ROLLOUT_POLL_WAIT_TIME,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                # Collect done results
                for task in done:
                    traj = await task
                    traj: TensorDict
                    task_rid = task.get_name()
                    with self.lock:
                        rollout_tasks.pop(task_rid)
                        self.rollout_stat.accepted += 1
                        self.rollout_stat.running -= 1
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Finish rollout {task_rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
                            )
                    # Plan A: conditionally filter over-stale samples before enqueueing output
                    try:
                        drop = False
                        # high-water threshold and per-iteration drop budget
                        MAX_DROP_PER_LOOP = 64
                        qsize = self.output_queue.qsize()
                        qmax = self.output_queue.maxsize or 1
                        high_water = qsize >= int(0.8 * qmax)
                        # cache low-water: keep training flowing
                        cache_low_water = len(self.result_cache) < max(1, self.config.consumer_batch_size // 2)
                        # local drop counter
                        local_dropped = 0
                        try:
                            current_ver = self.inference_engine.get_version()
                            versions = traj.get("versions", None)
                            loss_mask = traj.get("loss_mask", None)
                            if versions is not None:
                                ver = versions[0].tolist()
                                valid_versions = [v for v in ver if v >= 0]
                                if valid_versions:
                                    max_version = max(valid_versions)
                                    if high_water and (not cache_low_water) and (current_ver - max_version) >= 2:
                                        drop = True
                        except Exception:
                            traceback.print_exc()
                        if drop:
                            with self.lock:
                                self.rollout_stat.accepted -= 1
                                self.drop_count += 1
                                dc = self.drop_count
                            logger.warning(
                                f"[OutputFilter] drop over-stale sample before enqueue; total_dropped={dc}"
                            )
                            continue
                        self.output_queue.put_nowait(traj)
                    except queue.Full:
                        raise RuntimeError(
                            "Output queue full. Please increase queue_size."
                        )

                await asyncio.sleep(1)
        except Exception:
            traceback.print_exc()
        finally:
            # Cancel remaining tasks
            with self.lock:
                for task in rollout_tasks.values():
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
            await self.session.close()

    def submit(
        self,
        data: Dict[str, Any],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
    ) -> None:
        try:
            if workflow is None:
                workflow = workflow_builder()
            self.input_queue.put_nowait((data, workflow))
        except queue.Full:
            raise RuntimeError("Input queue full. Please increase queue_size.")

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        """Wait for and return a batch of rollout samples.

        This method orchestrates the complete sample collection process:
        1. Purges stale samples from queue when version increases
        2. Collects samples from queue into cache
        3. Filters stale samples from cache
        4. Re-collects if needed after filtering (BUG #1 fix)
        5. Returns requested number of samples

        NOTE: Recompute logic has been moved out of wait() and should be called
        separately via recompute_all_proximal_t() BEFORE weight updates.

        All 6 identified bugs have been fixed in this refactored version.

        Args:
            count: Number of samples to return
            timeout: Maximum time to wait in seconds (default: 7 days)
            should_accept: Optional filter function for samples

        Returns:
            Concatenated batch of TensorDict samples

        Raises:
            TimeoutError: If timeout exceeded before collecting enough samples
            RuntimeError: If executor is exiting
        """
        tik = time.perf_counter()
        timeout = timeout or float(7 * 24 * 3600)

        # FIX BUG #3 & #4: Read version once at start for consistency across all operations
        current_ver = self.inference_engine.get_version()

        # FIX BUG #6: Check timeout before expensive operations
        if time.perf_counter() - tik >= timeout:
            raise TimeoutError(
                f"Timed out before purge, only have {len(self.result_cache)} samples in cache."
            )

        # Step 1: Purge stale samples from queue (uses consistent current_ver)
        self._purge_stale_samples_from_queue(current_ver)

        # Step 2: Collect samples from queue into cache
        self._collect_samples_from_queue(count, timeout, tik, should_accept)

        # Check if we have enough after initial collection
        if self.exiting.is_set():
            raise RuntimeError("Rollout engine is exiting, cannot wait for results.")
        if len(self.result_cache) < count:
            raise TimeoutError(
                f"Timed out waiting for {count} rollouts, only received {len(self.result_cache)}."
            )

        if self.config.enable_rollout_tracing:
            logger.info(
                f"Rollout results are ready! cache_size/count = {len(self.result_cache)}/{count}"
            )

        # Step 3: Recompute removed - now done before weight update via recompute_all_proximal_t()

        # Step 4: Filter stale samples from cache (uses same current_ver)
        dropped = self._filter_stale_from_cache(current_ver)

        # FIX BUG #1: After cache filtering, if we don't have enough samples, collect more
        # This is the simplified fix - just reuse _collect_samples_from_queue!
        if len(self.result_cache) < count:
            remaining_timeout = timeout - (time.perf_counter() - tik)
            if remaining_timeout <= 0:
                raise TimeoutError(
                    f"Timed out after cache filtering: need {count}, have {len(self.result_cache)}. "
                    f"{dropped} samples were dropped as too stale."
                )

            logger.info(
                f"[CacheFilter] After filtering, need {count - len(self.result_cache)} more samples. "
                f"Current cache: {len(self.result_cache)}, requested: {count}"
            )

            # Collect remaining samples (reuses same collection logic!)
            self._collect_samples_from_queue(count, timeout, tik, should_accept)

            # Final check after re-collection
            if len(self.result_cache) < count:
                if self.exiting.is_set():
                    raise RuntimeError("Rollout engine is exiting, cannot wait for results.")
                raise TimeoutError(
                    f"Timed out after re-collection: need {count}, have {len(self.result_cache)}."
                )

        # Step 5: Return results
        results, self.result_cache = (
            self.result_cache[:count],
            self.result_cache[count:],
        )
        for td in results:
            _ensure_recompute_key(td)
        self._record_effective_samples(len(results))
        return concat_padded_tensors(results)

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        for item in data:
            self.submit(item, workflow, workflow_builder)
        return self.wait(count=len(data))

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ):
        if not hasattr(self, "data_generator"):
            self.data_generator = itertools.cycle(dataloader)
        assert dataloader.batch_size is not None
        while True:
            # Submit at least two batches to allow maximum overlap
            if (
                self.get_capacity() + dataloader.batch_size > 0
                and self.input_queue.qsize() + dataloader.batch_size
                < self.input_queue.maxsize
            ):
                data = next(self.data_generator)
                for item in data:
                    self.submit(
                        item,
                        workflow=workflow,
                        workflow_builder=workflow_builder,
                    )
            try:
                return self.wait(
                    dataloader.batch_size, timeout=1, should_accept=should_accept
                )
            except TimeoutError:
                pass

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()
