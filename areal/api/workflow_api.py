from __future__ import annotations  # noqa

import asyncio
import queue
import random
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import aiohttp
import torch
import torch.distributed as dist
import uvloop
from megatron.core import parallel_state as mpu
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import RolloutStat
from areal.api.workflow_components import (
    ProximalRecomputer,
    StalenessControlStrategy,
    StandardPPOStrategy,
    create_workflow_components,
)
from areal.experimental.openai.types import CompletionWithTokenLogpReward
from areal.utils import logging
from areal.utils.data import concat_padded_tensors, cycle_dataloader

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine


# Re-export from components for backward compatibility
from areal.api.workflow_components import RECOMPUTE_VERSION_KEY, _ensure_recompute_key

ROLLOUT_POLL_WAIT_TIME = 0.05


@dataclass
class _TimedResult:
    t: int
    data: Dict[str, Any]


@dataclass
class _RolloutTaskInput:
    data: Dict[str, Any]
    workflow: RolloutWorkflow
    should_accept: Callable | None = None


@dataclass
class _RolloutTask:
    create_time: int
    task: asyncio.Task
    task_input: _RolloutTaskInput


class RolloutWorkflow:

    async def arun_episode(
        self, engine: "InferenceEngine", data: Dict[str, Any]
    ) -> Dict[str, Any] | None | Dict[str, CompletionWithTokenLogpReward]:
        """Run a single episode of the workflow.

        Note
        ----
        Returning `None` implies that this trajectory is rejected and will not be used for training.

        See concrete example implementations under the `areal/workflow` directory.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine to use for generating responses
        data : Dict[str, Any]
            Input data for the workflow episode

        Returns
        -------
        Dict[str, Any] | None | Dict[str, CompletionWithTokenLogpReward]
            The trajectory result, None if rejected, or a dictionary of completion results
        """
        raise NotImplementedError()


def check_trajectory_format(
    data: Dict[str, Any] | None | Dict[str, CompletionWithTokenLogpReward],
    batch_size: int | None = None,
    expected_keys: set | None = None,
    logger: Any = None,
) -> bool:
    """Check the format of trajectory data returned by workflow.arun_episode.

    This function validates trajectory data to ensure it conforms to one of three expected formats:

    1. **None**: Indicates a rejected trajectory that will not be used for training
    2. **Dict[str, CompletionWithTokenLogpReward]**: Completion results from the workflow
    3. **Dict[str, torch.Tensor]**: Tensor format with specific shape and key requirements

    For tensor format validation, the function ensures:

    - Required keys ``input_ids`` and ``attention_mask`` are present
    - All tensors have consistent batch size and sequence length dimensions
    - Tensor shapes follow the pattern ``[batch_size, max_seqlen]``
    - Keys are consistent across different episodes when ``expected_keys`` is provided

    Special handling is provided for:

    - **multi_modal_input**: Expected to be a non-empty list of dictionaries containing ``pixel_values``
    - **Non-tensor data**: Logged for informational purposes

    Parameters
    ----------
    data : Dict[str, Any] | None | Dict[str, CompletionWithTokenLogpReward]
        The trajectory data to validate. Can be:

        - ``None`` for rejected trajectories
        - Dictionary mapping strings to ``CompletionWithTokenLogpReward`` objects
        - Dictionary mapping strings to PyTorch tensors or other data types

    batch_size : int | None, optional
        Expected batch size for tensor validation. If ``None``, batch size is inferred
        from the first dimension of ``input_ids``. Default is ``None``.

    expected_keys : set | None, optional
        Set of expected keys for consistency checking across multiple episodes.
        If provided, validates that the current trajectory contains all expected keys.
        Default is ``None``.

    logger : Any, optional
        Logger instance for warning and info messages. If ``None``, creates a default
        logger named "Workflow API". Default is ``None``.

    Returns
    -------
    bool
        ``True`` if the trajectory format is valid, ``False`` otherwise.

    Raises
    ------
    ValueError
        If the trajectory format is invalid. Error messages provide detailed information
        about the specific validation failure, including:

        - Missing required keys
        - Incorrect tensor dimensions
        - Inconsistent batch sizes or sequence lengths
        - Invalid multi-modal input format
        - Key inconsistencies across episodes

    Examples
    --------
    Basic usage with tensor data:

    >>> import torch
    >>> data = {
    ...     'input_ids': torch.randint(0, 1000, (2, 10)),
    ...     'attention_mask': torch.ones(2, 10)
    ... }
    >>> check_trajectory_format(data, batch_size=2)
    True

    Validation with expected keys:

    >>> expected = {'input_ids', 'attention_mask', 'labels'}
    >>> data_with_labels = {
    ...     'input_ids': torch.randint(0, 1000, (2, 10)),
    ...     'attention_mask': torch.ones(2, 10),
    ...     'labels': torch.randint(0, 1000, (2, 10))
    ... }
    >>> check_trajectory_format(data_with_labels, expected_keys=expected)
    True

    Rejected trajectory:

    >>> check_trajectory_format(None)
    True

    See Also
    --------
    RolloutWorkflow.arun_episode : Method that returns trajectory data to be validated
    WorkflowExecutor : Class that uses this function when ``check_trajectory_format`` is enabled
    """
    if logger is None:
        logger = logging.getLogger("Workflow API")
    if data is None:
        return True

    if not isinstance(data, dict):
        raise ValueError(f"Expected data to be None or dict, got {type(data)}")

    if len(data) == 0:
        raise ValueError("Data dict cannot be empty")

    # Check if all values are CompletionWithTokenLogpReward
    if all(isinstance(v, CompletionWithTokenLogpReward) for v in data.values()):
        return True

    # Check required keys
    # At least require `input_ids` and `attention_mask`
    required_keys = {"input_ids", "attention_mask"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in tensor data: {missing_keys}")

    # Check tensor shapes
    input_ids = data["input_ids"]
    if input_ids.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors with shape [batch_size, max_seqlen], got {input_ids.dim()}D"
        )

    inferred_batch_size, max_seqlen = input_ids.shape

    if batch_size is not None and inferred_batch_size != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {inferred_batch_size}")

    # Check all tensors have consistent shape
    for key, value in data.items():
        if torch.is_tensor(value):
            if value.shape[0] != inferred_batch_size:
                logger.warning(
                    f"The first dim of tensor `{key}` is {value.shape[0]}, "
                    f"rather than the batch size of input_ids ({inferred_batch_size})."
                )
            if value.ndim >= 2 and value.shape[1] != max_seqlen:
                logger.warning(
                    f"The second dim of tensor `{key}` is {value.shape[1]}, "
                    f"rather than the max seqlen of input_ids ({max_seqlen})."
                )
        elif key == "multi_modal_input":
            if (
                not isinstance(value, list)
                or len(value) == 0
                or any(not isinstance(v, dict) for v in value)
            ):
                raise ValueError(
                    "multi_modal_input should be a non-empty list of dicts"
                )
            if not all("pixel_values" in v for v in value):
                raise ValueError(
                    "multi_modal_input should at least contain the `pixel_values` field."
                )
        else:
            logger.info(f"Encounter non-tensor data with key `{key}`: {value}")

    # Check key consistency if expected_keys is provided
    if expected_keys is not None:
        missing_keys = expected_keys - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"Inconsistent keys compared to expected: "
                f"expected {expected_keys}, but {missing_keys} are missing."
            )

    return True


class WorkflowExecutor:
    """Orchestrates rollout workflow execution with pluggable components.

    This class now follows dependency injection pattern - components are injected
    rather than created internally. Use create_workflow_executor() factory for
    standard instantiation.
    """

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: "InferenceEngine",
        staleness_strategy: StalenessControlStrategy | None = None,
        proximal_recomputer: ProximalRecomputer | None = None,
    ):
        """Initialize WorkflowExecutor with injected components.

        Args:
            config: Training configuration
            inference_engine: Inference engine for rollouts
            staleness_strategy: Strategy for staleness control (injected)
            proximal_recomputer: Component for recomputing proximal_t (injected, optional)

        Note:
            For standard usage, use create_workflow_executor() factory instead of
            calling this constructor directly.
        """
        self.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config
        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()

        self.inference_engine = inference_engine

        qsize = config.queue_size or self.max_concurrent_rollouts * 16
        self.input_queue = queue.Queue(maxsize=qsize)
        self.output_queue = queue.Queue(maxsize=qsize)
        self.result_cache: List[_TimedResult] = []

        self.rollout_stat = RolloutStat()

        # For trajectory format checking
        self._expected_trajectory_keys: set | None = None

        self.session = None
        # Count of dropped over-stale samples at output filtering
        self.drop_count = 0
        # Track last version purged from output_queue
        self._last_purged_ver = -1
        # Track recent effective samples returned to the trainer
        self._effective_history = deque(maxlen=16)
        self._effective_sum = 0.0

        # Hook/callback lists for pause and resume lifecycle events
        self.pre_pause_hooks: List[Callable] = []
        self.post_pause_hooks: List[Callable] = []
        self.pre_resume_hooks: List[Callable] = []
        self.post_resume_hooks: List[Callable] = []

        # DEPENDENCY INJECTION: Components are provided from outside
        # Default to StandardPPOStrategy if not provided (backward compatible)
        self.staleness_strategy = staleness_strategy or StandardPPOStrategy(config)
        self.proximal_recomputer = proximal_recomputer

        # GUARDRAIL: Warn if segment-wise PPO is enabled but components not properly injected
        enable_sdp = getattr(config, "enable_segment_wise_ppo", False)
        if enable_sdp and (staleness_strategy is None or proximal_recomputer is None):
            import warnings
            warnings.warn(
                "WorkflowExecutor created with enable_segment_wise_ppo=True but without proper "
                "component injection. This likely means the factory was not used. "
                "Please use create_workflow_executor(config, engine) instead of direct instantiation. "
                "Segment-wise PPO features may not work correctly!",
                UserWarning,
                stacklevel=2
            )

        # Auto-register recompute hook if recomputer is provided
        if self.proximal_recomputer is not None:
            self.pre_pause_hooks.append(
                lambda: self.proximal_recomputer.recompute_all(
                    self.output_queue, self.result_cache
                )
            )

    def _purge_stale_samples_from_queue(self, current_ver: int) -> None:
        """Drain output queue and drop stale samples when version increases.

        Args:
            current_ver: Current model version
        """
        # DELEGATE to strategy - strategy contains the actual logic
        self._last_purged_ver = self.staleness_strategy.purge_stale_samples_from_queue(
            self.output_queue,
            current_ver,
            self._last_purged_ver,
            self.inference_engine,
            self.result_cache,
            self.config,
            self.logger,
        )

    def _purge_stale_samples_from_queue_OLD(self, current_ver: int) -> None:
        """OLD IMPLEMENTATION - DEPRECATED, kept for reference.

        This logic has been moved into the strategy classes.
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
                self.logger.error(
                    f"Output queue remains full after version purge. "
                    f"Queue size: {self.output_queue.qsize()}/{self.output_queue.maxsize}. "
                    f"Please increase queue_size."
                )
                raise RuntimeError(
                    "Output queue full when putting back after version purge. Please increase queue_size."
                )

        self.logger.info(
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
                        self.logger.info(
                            f"Accept rollout result. cache_size/count = {len(self.result_cache)}/{count}"
                        )
                    # Note: result is a dict, not TensorDict, so no clone() method
                    # Each result from the queue is already a separate object
                    self.result_cache.append(result)
                    collected += 1
                elif result is not None:
                    # FIX BUG #2: Decrement rollout_stat when should_accept rejects
                    if self.config.enable_rollout_tracing:
                        self.logger.info(f"Rollout is rejected by should_accept filter.")
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
        # DELEGATE to strategy - strategy contains the actual logic
        # Note: strategy modifies result_cache in place
        return self.staleness_strategy.filter_stale_from_cache(
            self.result_cache,
            current_ver,
            self.config,
            self.logger,
        )

    def recompute_all_proximal_t(self) -> None:
        """Recompute proximal_t for all v-1 samples before weight update.

        This method is kept for backward compatibility. It delegates to the
        ProximalRecomputer component if available, otherwise does nothing.

        This should be called RIGHT BEFORE update_weights() to ensure:
        1. All in-progress rollouts are still at current version
        2. All v-1 samples (in queue or cache) get recomputed
        3. No samples miss their recompute window

        Processes BOTH output_queue and result_cache.
        """
        # DELEGATE to recomputer if available
        if self.proximal_recomputer is not None:
            self.proximal_recomputer.recompute_all(self.output_queue, self.result_cache)
        else:
            # No recomputer available - this is normal for standard PPO mode
            pass

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
                            self.logger.info(
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
                            self.logger.warning(
                                f"[Recompute] sample#{idx}: length mismatch, required idx {max_required_offset} "
                                f"but got {len(latest_out_logp)} logprobs"
                            )
                            continue

                        for pos_idx, seq_idx in need_positions:
                            rel_offset = seq_idx - start_index - 1
                            if rel_offset < 0 or rel_offset >= len(latest_out_logp):
                                self.logger.warning(
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

    def initialize(self, logger=None, train_data_parallel_size: int | None = None):
        if logger is None:
            logger = logging.getLogger("WorkflowExecutor")
        self.logger = logger

        if train_data_parallel_size is not None:
            self.dp_world_size = train_data_parallel_size
        else:
            if dist.is_initialized():
                if not mpu.is_initialized():
                    self.dp_world_size = dist.get_world_size()
                else:
                    self.dp_world_size = mpu.get_data_parallel_world_size()
            else:
                self.dp_world_size = 1

        self.rollout_tasks: Dict[str, _RolloutTask] = {}
        self.rollout_thread = threading.Thread(
            target=self._rollout_thread, daemon=True
        )  # set daemon=True to automatically exit when error occurs
        self.rollout_thread.start()

    def destroy(self):
        self.exiting.set()
        self.rollout_thread.join()

    def get_capacity(self):
        with self.lock:
            max_concurrent_rollouts = max(
                1, self.max_concurrent_rollouts // self.dp_world_size
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
            consumer_bs = max(1, self.config.consumer_batch_size // self.dp_world_size)
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
                    x = self.input_queue.get_nowait()
                    x: _RolloutTaskInput
                    self.logger.debug(f"Get data from puller: {x.data}")
                    task = asyncio.create_task(
                        x.workflow.arun_episode(self.inference_engine, x.data),
                        name=str(rid),
                    )
                    rollout_tasks[str(rid)] = _RolloutTask(
                        create_time=time.monotonic_ns(), task=task, task_input=x
                    )
                    self.rollout_stat.submitted += 1
                    self.rollout_stat.running += 1
                    if self.config.enable_rollout_tracing:
                        self.logger.info(
                            f"Submit rollout rid {rid}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}."
                        )
                    capacity -= 1
                    rid += 1
                tasks = [x.task for x in rollout_tasks.values()]
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

                    # Trajectory format checking, directly raise an error when the format is wrong.
                    if self.config.check_trajectory_format and traj is not None:
                        check_trajectory_format(
                            traj, expected_keys=self._expected_trajectory_keys
                        )
                        # Track expected keys for consistency checking
                        if isinstance(traj, dict) and "input_ids" in traj:
                            if self._expected_trajectory_keys is None:
                                self._expected_trajectory_keys = set(traj.keys())
                                self.logger.info(
                                    f"Trajectory format check: tracking keys {self._expected_trajectory_keys}"
                                )

                    if isinstance(traj, dict) and all(
                        isinstance(v, CompletionWithTokenLogpReward)
                        for v in traj.values()
                    ):
                        traj = concat_padded_tensors(
                            [v.to_tensor_dict() for v in traj.values()]
                        )
                    assert traj is None or isinstance(traj, dict), traj
                    task_rid = task.get_name()
                    with self.lock:
                        task_obj = rollout_tasks.pop(task_rid)
                        self.rollout_stat.accepted += 1
                        self.rollout_stat.running -= 1
                        if self.config.enable_rollout_tracing:
                            self.logger.info(
                                f"Finish rollout {task_rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
                            )
                    # Plan A: conditionally filter over-stale samples before enqueueing output
                    if self.staleness_strategy.should_filter_before_enqueue():
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
                                self.logger.warning(
                                    f"[OutputFilter] drop over-stale sample before enqueue; total_dropped={dc}"
                                )
                                continue
                            self.output_queue.put_nowait(traj)
                        except queue.Full:
                            raise RuntimeError(
                                "Output queue full. Please increase queue_size."
                            )
                    else:
                        # Feature disabled, enqueue directly without filtering
                        try:
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
                for task_obj in rollout_tasks.values():
                    if not task_obj.task.done():
                        task_obj.task.cancel()
                        try:
                            await task_obj.task
                        except asyncio.CancelledError:
                            pass

    def submit(
        self,
        data: Dict[str, Any],
        workflow: "RolloutWorkflow" | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> None:
        """Submit a request to the workflow executor.

        See :meth:`~areal.api.engine_api.InferenceEngine.submit` for detailed documentation.
        """
        try:
            if workflow is None:
                workflow = workflow_builder()
            x = _RolloutTaskInput(
                data=data, workflow=workflow, should_accept=should_accept
            )
            self.input_queue.put_nowait(x)
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
            self.logger.info(
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

            self.logger.info(
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
        workflow: "RolloutWorkflow" | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> Dict[str, Any]:
        """Submit a batch of requests and wait for results.

        See :meth:`~areal.api.engine_api.InferenceEngine.rollout_batch` for detailed documentation.
        """
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
                workflow_builder=workflow_builder,
                should_accept=should_accept,
            )
        return self.wait(count=len(data))

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: "RolloutWorkflow" | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ):
        """Prepare a batch with controlled staleness.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for detailed documentation.
        """
        if not hasattr(self, "data_generator"):
            self.data_generator = cycle_dataloader(dataloader)
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
                        should_accept=should_accept,
                    )
            try:
                return self.wait(dataloader.batch_size, timeout=1)
            except TimeoutError:
                pass

    def register_pre_pause_hook(self, hook_fn: Callable):
        """Register a hook to be called before pausing rollout.

        Hooks are executed synchronously in registration order.
        Useful for tasks that need to run before weight updates (e.g., recompute).

        Parameters
        ----------
        hook_fn : Callable
            Function to call before pause. Should take no arguments.

        Example
        -------
        >>> def my_pre_pause_logic():
        ...     print("Preparing for pause...")
        >>> executor.register_pre_pause_hook(my_pre_pause_logic)
        """
        self.pre_pause_hooks.append(hook_fn)

    def register_post_pause_hook(self, hook_fn: Callable):
        """Register a hook to be called after pausing rollout.

        Parameters
        ----------
        hook_fn : Callable
            Function to call after pause. Should take no arguments.
        """
        self.post_pause_hooks.append(hook_fn)

    def register_pre_resume_hook(self, hook_fn: Callable):
        """Register a hook to be called before resuming rollout.

        Parameters
        ----------
        hook_fn : Callable
            Function to call before resume. Should take no arguments.
        """
        self.pre_resume_hooks.append(hook_fn)

    def register_post_resume_hook(self, hook_fn: Callable):
        """Register a hook to be called after resuming rollout.

        Parameters
        ----------
        hook_fn : Callable
            Function to call after resume. Should take no arguments.
        """
        self.post_resume_hooks.append(hook_fn)

    def pause(self):
        """Pause request submission for async rollout.

        Executes all registered pre-pause hooks before pausing, and post-pause hooks after.
        By default, if enable_segment_wise_ppo is enabled, recompute_all_proximal_t()
        is automatically registered as a pre-pause hook.

        See :meth:`~areal.api.engine_api.InferenceEngine.pause` for detailed documentation.
        """
        # Execute pre-pause hooks
        for hook in self.pre_pause_hooks:
            try:
                hook()
            except Exception as e:
                self.logger.warning(f"Pre-pause hook {hook.__name__} failed: {e}")
                traceback.print_exc()

        self.paused.set()

        # Execute post-pause hooks
        for hook in self.post_pause_hooks:
            try:
                hook()
            except Exception as e:
                self.logger.warning(f"Post-pause hook {hook.__name__} failed: {e}")
                traceback.print_exc()

    def resume(self):
        """Resume request submission for async rollout.

        Executes all registered pre-resume hooks before resuming, and post-resume hooks after.

        See :meth:`~areal.api.engine_api.InferenceEngine.resume` for detailed documentation.
        """
        # Execute pre-resume hooks
        for hook in self.pre_resume_hooks:
            try:
                hook()
            except Exception as e:
                self.logger.warning(f"Pre-resume hook {hook.__name__} failed: {e}")
                traceback.print_exc()

        self.paused.clear()

        # Execute post-resume hooks
        for hook in self.post_resume_hooks:
            try:
                hook()
            except Exception as e:
                self.logger.warning(f"Post-resume hook {hook.__name__} failed: {e}")
                traceback.print_exc()


# ============================================================================
# Factory Function for Creating Workflow Executor
# ============================================================================


def create_workflow_executor(
    config: InferenceEngineConfig,
    inference_engine: "InferenceEngine",
) -> WorkflowExecutor:
    """Factory function for creating WorkflowExecutor with proper component injection.

    This is the recommended way to create a WorkflowExecutor. It handles:
    1. Creating appropriate strategy based on config
    2. Creating recomputer if needed
    3. Injecting all components into executor
    4. Following Spring @Configuration pattern

    Args:
        config: Training configuration
        inference_engine: Inference engine for rollouts

    Returns:
        Fully configured WorkflowExecutor instance

    Example:
        >>> config = InferenceEngineConfig()
        >>> config.enable_segment_wise_ppo = True
        >>> engine = InferenceEngine(config)
        >>> executor = create_workflow_executor(config, engine)
        >>> executor.initialize()
    """
    # Create a temporary logger for component creation
    # (real logger will be set in initialize())
    temp_logger = logging.getLogger("WorkflowExecutor")

    # Use component factory to create strategy and recomputer
    strategy, recomputer = create_workflow_components(config, inference_engine, temp_logger)

    # Inject components into executor
    executor = WorkflowExecutor(
        config=config,
        inference_engine=inference_engine,
        staleness_strategy=strategy,
        proximal_recomputer=recomputer,
    )

    return executor
