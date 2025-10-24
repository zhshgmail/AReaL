from __future__ import annotations  # noqa

import queue
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.workflow_api import RolloutWorkflow
from areal.core.async_task_runner import AsyncTaskRunner, TaskQueueFullError
from areal.core.staleness_manager import StalenessManager
from areal.experimental.openai.types import CompletionWithTokenLogpReward
from areal.utils import logging
from areal.utils.data import concat_padded_tensors, cycle_dataloader

if TYPE_CHECKING:
    from areal.api.cache_api import RolloutCache
    from areal.api.engine_api import InferenceEngine
    from areal.api.queue_api import RolloutQueue
    from areal.api.staleness_control import StalenessControlStrategy
    from areal.core.filtered_capacity_modifier import FilteredSamplesCapacityModifier
    from areal.core.proximal_recomputer import ProximalRecomputer


def check_trajectory_format(
    data: dict[str, Any] | None | dict[str, CompletionWithTokenLogpReward],
    batch_size: int | None = None,
    expected_keys: set | None = None,
    logger: Any = None,
) -> bool:
    """Check the format of trajectory data returned by workflow.arun_episode.

    This function validates trajectory data to ensure it conforms to one of three
    expected formats:

    1. **None**: Indicates a rejected trajectory that will not be used for
       training
    2. **Dict[str, CompletionWithTokenLogpReward]**: Completion results from
       the workflow
    3. **Dict[str, torch.Tensor]**: Tensor format with specific shape and
       key requirements

    For tensor format validation, the function ensures:

    - Required keys ``input_ids`` and ``attention_mask`` are present
    - All tensors have consistent batch size and sequence length dimensions
    - Tensor shapes follow the pattern ``[batch_size, max_seqlen]``
    - Keys are consistent across different episodes when ``expected_keys`` is
      provided

    Special handling is provided for:

    - **multi_modal_input**: Expected to be a non-empty list of dictionaries
      containing ``pixel_values``
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
    RolloutWorkflow.arun_episode : Method that returns trajectory data
    WorkflowExecutor : Class that uses this function when
        ``check_trajectory_format`` is enabled
    """
    if logger is None:
        logger = logging.getLogger(__name__)
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
            f"Expected 2D tensors with shape [batch_size, max_seqlen], "
            f"got {input_ids.dim()}D"
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
                    "multi_modal_input should at least contain the "
                    "`pixel_values` field."
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


@dataclass
class _TimedResult:
    """Result with creation timestamp for cache management."""

    t: int
    data: dict[str, Any]


@dataclass
class _RolloutTaskInput:
    """Internal wrapper for rollout-specific task input."""

    data: dict[str, Any]
    workflow: RolloutWorkflow
    should_accept: Callable | None = None


class WorkflowExecutor:
    """Executor for asynchronous workflow-based rollout generation.

    This class orchestrates the execution of rollout workflows with
    AReaL-specific features including staleness management, trajectory
    validation, and result filtering. It uses a generic AsyncTaskRunner
    internally for task execution while adding domain-specific logic for
    RL training.

    The executor manages:
    - Integration with InferenceEngine for model generation
    - Staleness-aware capacity control via StalenessManager
    - Trajectory format validation
    - Result filtering via should_accept callbacks
    - CompletionWithTokenLogpReward processing

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine including queue sizes,
        concurrency limits, and validation settings.
    inference_engine : InferenceEngine
        The inference engine to use for generating completions.
    staleness_manager : StalenessManager | None, optional
        Manager for staleness-aware capacity control. If None, a default manager
        will be created during initialization. Default is None.

    See Also
    --------
    AsyncTaskRunner : Generic async task executor used internally
    StalenessManager : Manages capacity based on staleness constraints
    RolloutWorkflow : Interface for rollout episode execution
    """

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: InferenceEngine,
        staleness_manager: StalenessManager | None = None,
        output_queue: "RolloutQueue | None" = None,
        result_cache: "RolloutCache | None" = None,
        staleness_strategy: "StalenessControlStrategy | None" = None,
        proximal_recomputer: "ProximalRecomputer | None" = None,
        filtered_capacity_modifier: "FilteredSamplesCapacityModifier | None" = None,
        logger: Any = None,
    ):
        self.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.consumer_batch_size = config.consumer_batch_size

        self.config = config
        self.inference_engine = inference_engine

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self.staleness_manager = staleness_manager

        # Segment-wise PPO components (injected via factory or None for standard PPO)
        self.staleness_strategy = staleness_strategy
        self.proximal_recomputer = proximal_recomputer
        self.filtered_capacity_modifier = filtered_capacity_modifier

        # Output queue and result cache (optional, for segment-wise PPO)
        # If provided, we use them; if None, we use internal list
        self.output_queue = output_queue
        self.result_cache = result_cache

        # Track last purged version for queue purging (segment-wise PPO)
        self._last_purged_version = -1

        # Metrics tracking for observability
        self._metrics_filtered_total = 0
        self._metrics_accepted_total = 0
        self._metrics_rejected_total = 0

        # Logger (will be set in initialize if None)
        self.logger = logger

        # Create the generic async task runner
        qsize = config.queue_size or self.max_concurrent_rollouts * 16
        self.runner = AsyncTaskRunner[dict[str, Any] | None](
            max_queue_size=qsize,
            enable_tracing=config.enable_rollout_tracing,
        )

        # For trajectory format checking
        self._expected_trajectory_keys: set | None = None

        # Cache for tracking inputs and accepted/rejected results
        # These are used when output_queue/result_cache are None (standard PPO)
        self._pending_results: list[dict[str, Any]] = []
        self._pending_inputs: list[_RolloutTaskInput] = []

    def initialize(self, logger=None, train_data_parallel_size: int | None = None):
        """Initialize the workflow executor and start the async task runner.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for debugging and tracing. If None, creates a
            default logger.
        train_data_parallel_size : int | None, optional
            Data parallel world size for capacity scaling. If None, will be inferred
            from distributed state.
        """
        # Set logger if not already set in constructor
        if logger is not None:
            self.logger = logger
        elif self.logger is None:
            self.logger = logging.getLogger(__name__)

        # Initialize staleness manager if not provided
        if self.staleness_manager is None:
            if train_data_parallel_size is not None:
                dp_world_size = train_data_parallel_size
            else:
                if dist.is_initialized():
                    if not mpu.is_initialized():
                        dp_world_size = dist.get_world_size()
                    else:
                        dp_world_size = mpu.get_data_parallel_world_size()
                else:
                    dp_world_size = 1

            # Apply data parallel scaling
            max_concurrent_rollouts = max(
                1, self.max_concurrent_rollouts // dp_world_size
            )
            consumer_batch_size = max(1, self.consumer_batch_size // dp_world_size)

            self.staleness_manager = StalenessManager(
                max_concurrent_rollouts=max_concurrent_rollouts,
                consumer_batch_size=consumer_batch_size,
                max_staleness=self.config.max_head_offpolicyness,
                logger=self.logger,
            )

        # Initialize the generic async task runner
        self.runner.initialize(logger=logger)

    def destroy(self):
        """Shutdown the workflow executor and clean up resources."""
        self.runner.destroy()

    def get_capacity(self):
        """Get current available capacity for new rollouts.

        Returns
        -------
        int
            Number of new rollout slots available based on staleness constraints.
        """
        version = self.inference_engine.get_version()
        capacity = self.staleness_manager.get_capacity(version)
        return capacity

    def _create_workflow_task(
        self, task_input: _RolloutTaskInput
    ) -> Callable[[], Awaitable[dict[str, Any] | None]]:
        """Wrapper to create an async function that will be executed by AsyncTaskRunner.

        This is a synchronous function that returns an async function, which allows
        us to capture the task_input context.

        Parameters
        ----------
        task_input : _RolloutTaskInput
            The rollout task input containing workflow, data, and filter callback.

        Returns
        -------
        Callable
            An async function that executes the workflow and applies
            filtering/validation.
        """

        async def _execute_workflow():
            """Execute workflow.arun_episode and apply AReaL-specific logic."""
            # Execute the workflow
            traj = await task_input.workflow.arun_episode(
                self.inference_engine, task_input.data
            )

            # Trajectory format checking
            if self.config.check_trajectory_format and traj is not None:
                check_trajectory_format(
                    traj,
                    expected_keys=self._expected_trajectory_keys,
                    logger=self.logger,
                )
                # Track expected keys for consistency checking
                if isinstance(traj, dict) and "input_ids" in traj:
                    if self._expected_trajectory_keys is None:
                        self._expected_trajectory_keys = set(traj.keys())
                        self.logger.info(
                            f"Trajectory format check: tracking keys "
                            f"{self._expected_trajectory_keys}"
                        )

            # Convert CompletionWithTokenLogpReward to tensor dict if needed
            if isinstance(traj, dict) and all(
                isinstance(v, CompletionWithTokenLogpReward) for v in traj.values()
            ):
                traj = concat_padded_tensors(
                    [v.to_tensor_dict() for v in traj.values()]
                )

            assert traj is None or isinstance(traj, dict), traj

            # Apply should_accept filtering
            should_accept_traj = traj is not None and (
                task_input.should_accept is None or task_input.should_accept(traj)
            )

            # Staleness filtering and proximal recomputation (segment-wise PPO)
            if should_accept_traj and self.staleness_strategy is not None:
                current_ver = self.inference_engine.get_version()

                # Pre-enqueue staleness check
                from tensordict import TensorDict

                if not isinstance(traj, TensorDict):
                    traj = TensorDict(traj, batch_size=[])

                # Extract version info for logging
                sample_version = -1
                if "versions" in traj.keys():
                    versions = traj.get("versions")
                    if versions is not None:
                        sample_version = versions[0, 0].item() if versions.numel() > 0 else -1

                should_enqueue = not self.staleness_strategy.is_sample_too_stale(
                    traj, current_ver, self.config
                )

                if should_enqueue:
                    # Add proximal logprobs if recomputer exists
                    if self.proximal_recomputer:
                        traj = self.proximal_recomputer.add_proximal_logprobs(
                            traj, current_ver
                        )
                    # Log acceptance
                    self._metrics_accepted_total += 1
                    self.logger.debug(
                        f"[StalenessFilter] Sample accepted: version={sample_version}, "
                        f"current_ver={current_ver}, staleness={current_ver - sample_version}, "
                        f"total_accepted={self._metrics_accepted_total}"
                    )
                else:
                    # Filtered due to staleness - mark as rejected
                    self._metrics_filtered_total += 1
                    self.logger.debug(
                        f"[StalenessFilter] Sample filtered due to staleness: version={sample_version}, "
                        f"current_ver={current_ver}, staleness={current_ver - sample_version}, "
                        f"total_filtered={self._metrics_filtered_total}"
                    )
                    if self.filtered_capacity_modifier:
                        self.filtered_capacity_modifier.on_samples_filtered(
                            1, current_ver
                        )
                    should_accept_traj = False

            # Notify staleness manager
            if should_accept_traj:
                self.staleness_manager.on_rollout_accepted()
                if self.config.enable_rollout_tracing:
                    stat = self.staleness_manager.get_stats()
                    self.logger.info(
                        f"Finish and accept rollout. "
                        f"Submit: {stat.submitted}, "
                        f"running: {stat.running}, "
                        f"accepted: {stat.accepted}."
                    )
                return traj
            else:
                self.staleness_manager.on_rollout_rejected()
                if self.config.enable_rollout_tracing:
                    stat = self.staleness_manager.get_stats()
                    self.logger.info(
                        f"Finish but reject rollout. "
                        f"Submit: {stat.submitted}, "
                        f"running: {stat.running}, "
                        f"accepted: {stat.accepted}."
                    )
                return None

        return _execute_workflow

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> None:
        """Submit a request to the workflow executor.

        See :meth:`~areal.api.engine_api.InferenceEngine.submit` for detailed
        documentation.
        """
        # Create workflow if builder provided
        if workflow is None:
            workflow = workflow_builder()
        self._pending_inputs.append(
            _RolloutTaskInput(
                data=data,
                workflow=workflow,
                should_accept=should_accept,
            )
        )

    def _commit_one_to_runner(self):
        task_input = self._pending_inputs.pop(0)
        # Create the async workflow execution function and submit to runner
        workflow_fn = self._create_workflow_task(task_input)
        try:
            self.runner.submit(workflow_fn)
        except TaskQueueFullError:
            # Convert RuntimeError from AsyncTaskRunner to queue.Full for
            # backward compatibility
            raise queue.Full("Input queue full. Please increase queue_size.")

        # Notify staleness manager of submission only after successful submission
        self.staleness_manager.on_rollout_submitted()
        if self.config.enable_rollout_tracing:
            stat = self.staleness_manager.get_stats()
            self.logger.info(
                f"Submit rollout. "
                f"Submit: {stat.submitted}, "
                f"running: {stat.running}, "
                f"accepted: {stat.accepted}."
            )

    def wait(self, count: int, timeout: float | None = None) -> dict[str, Any]:
        """Wait for workflow results.

        See :meth:`~areal.api.engine_api.InferenceEngine.wait` for detailed
        documentation.
        """
        start_time = time.perf_counter()
        timeout = timeout or float(7 * 24 * 3600)
        current_ver = self.inference_engine.get_version()

        # Step 1: Purge stale samples from queue (segment-wise PPO only)
        if self.staleness_strategy and self.output_queue:
            self._last_purged_version = self.staleness_strategy.purge_stale_samples_from_queue(
                output_queue=self.output_queue,
                current_ver=current_ver,
                last_purged_ver=self._last_purged_version,
                inference_engine=self.inference_engine,
                result_cache=self.result_cache,
                config=self.config,
                logger=self.logger,
            )

        # Keep requesting results from runner until we have enough accepted
        # (non-None) results. Use short timeout (1 second) for each wait call
        # to allow periodic checking. This matches original behavior where
        # wait() would poll and allow prepare_batch() to continue
        while True:
            # Submit pending inputs
            # Check capacity before submitting
            capacity = self.get_capacity()
            # Submit pending tasks
            for _ in range(capacity):
                if len(self._pending_inputs) == 0:
                    break
                self._commit_one_to_runner()

            # Step 2: Drain queue to cache (segment-wise PPO)
            if self.output_queue:
                while True:
                    try:
                        timed_result = self.output_queue.get_nowait()
                        # Add to result cache
                        if self.result_cache and hasattr(self.result_cache, 'add'):
                            self.result_cache.add(timed_result)
                        elif self.result_cache:
                            # Fallback for list-like cache
                            self.result_cache.append(timed_result)
                    except queue.Empty:
                        break

            # Step 3: Filter stale from cache (segment-wise PPO)
            if self.staleness_strategy and self.result_cache:
                dropped_cache = self.staleness_strategy.filter_stale_from_cache(
                    result_cache=self.result_cache,
                    current_ver=current_ver,
                    config=self.config,
                    logger=self.logger,
                )
                if dropped_cache > 0 and self.filtered_capacity_modifier:
                    self.filtered_capacity_modifier.on_samples_filtered(
                        dropped_cache, current_ver
                    )

            # Check if we have enough results
            # For segment-wise PPO (with result_cache), check cache size
            cache_size = None
            if self.result_cache:
                cache_size = (
                    self.result_cache.size()
                    if hasattr(self.result_cache, 'size')
                    else len(self.result_cache)
                )
                if cache_size >= count:
                    break
            # For standard PPO (without result_cache), check pending results
            elif len(self._pending_results) >= count:
                break

            elapsed = time.perf_counter() - start_time
            remaining_timeout = timeout - elapsed

            if remaining_timeout <= 0:
                actual_count = (
                    cache_size if cache_size is not None
                    else len(self._pending_results)
                )
                raise TimeoutError(
                    f"Timed out waiting for {count} rollouts, only received "
                    f"{actual_count}."
                )

            # Try to get at least the number we still need, but request at least 1
            # Note: runner.wait() might return fewer due to rejections (None results)
            if cache_size is not None:
                needed = max(1, count - cache_size)
            else:
                needed = max(1, count - len(self._pending_results))

            try:
                # Use short timeout to allow periodic returns (matches original
                # polling behavior)
                batch = self.runner.wait(
                    count=needed, timeout=min(0.1, remaining_timeout)
                )

                # Filter out None results (rejected trajectories)
                # runner.wait() returns List[T] where T can be None for
                # rejected rollouts
                accepted = [result for result in batch if result is not None]

                # For segment-wise PPO, add to output queue as TimedResult
                if self.output_queue:
                    for result in accepted:
                        timed_result = _TimedResult(
                            t=time.monotonic_ns(),
                            data=result
                        )
                        try:
                            self.output_queue.put_nowait(timed_result)
                        except queue.Full:
                            raise RuntimeError(
                                "Output queue full. Please increase queue_size."
                            )
                # For standard PPO, add directly to pending results
                else:
                    self._pending_results.extend(accepted)
            except TimeoutError:
                pass

        if self.config.enable_rollout_tracing:
            self.logger.info("Rollout results are ready!")

        # Log staleness filtering metrics (segment-wise PPO)
        if self.staleness_strategy is not None and self._metrics_filtered_total > 0:
            self.logger.info(
                f"[StalenessMetrics] Total samples: "
                f"filtered={self._metrics_filtered_total}, "
                f"accepted={self._metrics_accepted_total}, "
                f"filter_rate={self._metrics_filtered_total / (self._metrics_filtered_total + self._metrics_accepted_total):.2%}"
            )

        # Extract requested number of results
        # For segment-wise PPO (with result_cache)
        if self.result_cache:
            # Take results from cache
            if hasattr(self.result_cache, 'take_first_n'):
                # Using RolloutCache abstraction
                results = self.result_cache.take_first_n(count)
            else:
                # Using list
                self.result_cache.sort(key=lambda x: x.t)
                results, remaining = (
                    self.result_cache[:count],
                    self.result_cache[count:],
                )
                # Update result_cache reference if it's a list
                if isinstance(self.result_cache, list):
                    self.result_cache[:] = remaining
                else:
                    self.result_cache = remaining

            random.shuffle(results)
            return concat_padded_tensors([r.data for r in results])
        # For standard PPO (without result_cache)
        else:
            results = self._pending_results[:count]
            self._pending_results = self._pending_results[count:]

            # Shuffle for randomness (helps with data diversity)
            random.shuffle(results)

            # Concatenate into batch tensor format
            return concat_padded_tensors(results)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of requests and wait for results.

        See :meth:`~areal.api.engine_api.InferenceEngine.rollout_batch` for
        detailed documentation.
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
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ):
        """Prepare a batch with controlled staleness.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for
        detailed documentation.
        """
        if not hasattr(self, "data_generator"):
            self.data_generator = cycle_dataloader(dataloader)
        assert dataloader.batch_size is not None
        while True:
            # Submit at least two batches to allow maximum overlap
            if (
                self.get_capacity() + dataloader.batch_size > 0
                and self.runner.get_input_queue_size() + dataloader.batch_size
                < self.runner.max_queue_size
            ):
                data = next(self.data_generator)
                for item in data:
                    try:
                        self.submit(
                            item,
                            workflow=workflow,
                            workflow_builder=workflow_builder,
                            should_accept=should_accept,
                        )
                    except queue.Full:
                        # Capacity exhausted during batch submission, stop and wait
                        break
            try:
                return self.wait(dataloader.batch_size, timeout=1)
            except TimeoutError:
                pass

    def pause(self):
        """Pause request submission for async rollout.

        See :meth:`~areal.api.engine_api.InferenceEngine.pause` for detailed
        documentation.
        """
        self.runner.pause()

    def resume(self):
        """Resume request submission for async rollout.

        See :meth:`~areal.api.engine_api.InferenceEngine.resume` for detailed
        documentation.
        """
        self.runner.resume()

    def is_paused(self):
        return self.runner.paused.is_set()
