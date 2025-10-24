# Integration Strategy: Segment-Wise PPO + AsyncTaskRunner

## Decision: Clean Integration Approach

After attempting git rebase and encountering complex conflicts in `workflow_executor.py` (5+ conflict regions, 834-line merged file), I'm taking a **clean integration approach**:

**Strategy**: Create new feature branch from latest main, then carefully port our segment-wise PPO changes.

**Rationale**:
1. Main's refactoring (async_task_runner.py extraction) is substantial (617 → 713 lines in workflow_executor.py)
2. Our changes are well-understood and documented
3. Manual integration ensures proper understanding of new architecture
4. Results in cleaner, more maintainable code
5. Easier to verify correctness

## Step-by-Step Integration Plan

### Phase 1: Preparation

#### 1.1 Backup Current State ✅
```bash
git branch backup/segment-wise-ppo-v2-before-integration
```

#### 1.2 Identify All Our Changes
Files modified for segment-wise PPO:
- **New files** (15 total):
  - `areal/api/cache_api.py`
  - `areal/api/queue_api.py`
  - `areal/api/staleness_control.py`
  - `areal/api/workflow_factory.py`
  - `areal/core/capacity_modifier.py`
  - `areal/core/filtered_capacity_modifier.py`
  - `areal/core/proximal_recomputer.py`
  - `areal/core/rollout_cache.py`
  - `areal/core/rollout_queue.py`
  - `areal/core/staleness_strategies.py`
  - 8 test files in `areal/tests/sdp/`
  - `areal/tests/conftest.py`
  - `areal/tests/test_model_utils.py`
  - `examples/math/gsm8k_grpo_segment_wise.yaml`

- **Modified files** (10 total):
  - `areal/api/cli_args.py` - config propagation
  - `areal/api/io_struct.py` - proximal_logprobs_t field
  - `areal/core/staleness_manager.py` - capacity modifiers
  - `areal/core/workflow_executor.py` - ⚠️ CONFLICT
  - `areal/core/remote_inf_engine.py` - use factory
  - `areal/engine/ppo/actor.py` - feature flag
  - `areal/utils/functional.py` - utilities
  - `areal/utils/model.py` - model utils
  - `areal/workflow/rlvr.py` - pass proximal_logprobs_t
  - `areal/workflow/vision_rlvr.py` - pass proximal_logprobs_t
  - `realhf/impl/model/interface/ppo_interface.py` - collect proximal_t
  - `realhf/impl/model/utils/ppo_functional.py` - use proximal_t

### Phase 2: Create New Feature Branch

```bash
git checkout -b feature/segment-wise-ppo-v3 origin/main
```

This starts fresh from latest main (04ab6018).

### Phase 3: Port Non-Conflicting Files

#### 3.1 Copy All New Files
These files don't exist in main, so no conflicts:
```bash
# API layer
git checkout feature/segment-wise-ppo-v2 -- areal/api/cache_api.py
git checkout feature/segment-wise-ppo-v2 -- areal/api/queue_api.py
git checkout feature/segment-wise-ppo-v2 -- areal/api/staleness_control.py
git checkout feature/segment-wise-ppo-v2 -- areal/api/workflow_factory.py

# Core layer
git checkout feature/segment-wise-ppo-v2 -- areal/core/capacity_modifier.py
git checkout feature/segment-wise-ppo-v2 -- areal/core/filtered_capacity_modifier.py
git checkout feature/segment-wise-ppo-v2 -- areal/core/proximal_recomputer.py
git checkout feature/segment-wise-ppo-v2 -- areal/core/rollout_cache.py
git checkout feature/segment-wise-ppo-v2 -- areal/core/rollout_queue.py
git checkout feature/segment-wise-ppo-v2 -- areal/core/staleness_strategies.py

# Tests
git checkout feature/segment-wise-ppo-v2 -- areal/tests/sdp/
git checkout feature/segment-wise-ppo-v2 -- areal/tests/conftest.py
git checkout feature/segment-wise-ppo-v2 -- areal/tests/test_model_utils.py

# Examples
git checkout feature/segment-wise-ppo-v2 -- examples/math/gsm8k_grpo_segment_wise.yaml
```

#### 3.2 Port Simple Modifications
Files with small, isolated changes:

**areal/api/cli_args.py**:
- Add `enable_segment_wise_ppo` field
- Add propagation logic

**areal/api/io_struct.py**:
- Add `proximal_logprobs_t` field to ModelResponse

**areal/core/staleness_manager.py**:
- Add capacity modifier registration
- Add `modify_capacity` call in `get_capacity()`

**areal/engine/ppo/actor.py**:
- Add feature flag integration

**areal/utils/functional.py**:
- Add utility functions

**areal/utils/model.py**:
- Add model utilities

**areal/workflow/rlvr.py**:
- Pass proximal_logprobs_t

**areal/workflow/vision_rlvr.py**:
- Pass proximal_logprobs_t

**realhf/impl/model/interface/ppo_interface.py**:
- Collect proximal_t

**realhf/impl/model/utils/ppo_functional.py**:
- Use proximal_t for importance weights

### Phase 4: Handle Conflicting Files

#### 4.1 areal/core/workflow_executor.py (CRITICAL)

Our modifications to integrate:

**A. Imports** (lines 24-30):
```python
if TYPE_CHECKING:
    from areal.api.cache_api import RolloutCache
    from areal.api.engine_api import InferenceEngine
    from areal.api.queue_api import RolloutQueue
    from areal.api.staleness_control import StalenessControlStrategy
    from areal.core.filtered_capacity_modifier import FilteredSamplesCapacityModifier
    from areal.core.proximal_recomputer import ProximalRecomputer
```

**B. `__init__` Parameters** (lines 61-76):
```python
def __init__(
    self,
    config: InferenceEngineConfig,
    inference_engine: InferenceEngine,
    staleness_manager: StalenessManager | None = None,
    output_queue: RolloutQueue | None = None,  # NEW
    result_cache: RolloutCache | None = None,  # NEW
    staleness_strategy: StalenessControlStrategy | None = None,  # NEW
    proximal_recomputer: ProximalRecomputer | None = None,  # NEW
    filtered_capacity_modifier: FilteredSamplesCapacityModifier | None = None,  # NEW
    logger: Any = None,
):
```

**C. Default Creation** (in `__init__`):
```python
# Create defaults if not provided
if output_queue is None:
    from areal.core.rollout_queue import LocalRolloutQueue
    max_concurrent = config.max_concurrent_rollouts or config.consumer_batch_size
    qsize = config.queue_size or max_concurrent * 16
    output_queue = LocalRolloutQueue(maxsize=qsize)

if result_cache is None:
    from areal.core.rollout_cache import LocalRolloutCache
    result_cache = LocalRolloutCache()

self.output_queue = output_queue
self.result_cache = result_cache
self.staleness_strategy = staleness_strategy
self.proximal_recomputer = proximal_recomputer
self.filtered_capacity_modifier = filtered_capacity_modifier
self._last_purged_version = -1
```

**D. Staleness Filtering Before Enqueue** (in task completion callback):
```python
# After workflow completes, in the task callback:
if should_accept_traj:
    # Check staleness filtering (segment-wise PPO)
    current_ver = self.inference_engine.get_version()
    should_enqueue = True

    if self.staleness_strategy:
        # Pre-enqueue staleness check
        from tensordict import TensorDict
        if not isinstance(traj, TensorDict):
            traj = TensorDict(traj, batch_size=[])

        should_enqueue = not self.staleness_strategy.is_sample_too_stale(
            traj, current_ver, self.config
        )

    if should_enqueue:
        # Add proximal logprobs if recomputer exists
        if self.proximal_recomputer:
            traj = self.proximal_recomputer.add_proximal_logprobs(traj, current_ver)

        # Notify staleness manager of accepted rollout
        self.staleness_manager.on_rollout_accepted()

        try:
            self.output_queue.put_nowait(_TimedResult(task_obj.create_time, traj))
        except queue.Full:
            raise RuntimeError("Output queue full. Please increase queue_size.")
    else:
        # Filtered due to staleness
        if self.filtered_capacity_modifier:
            self.filtered_capacity_modifier.on_samples_filtered(1, current_ver)
        self.staleness_manager.on_rollout_rejected()
```

**E. Queue Purge and Cache Filtering** (in `wait()` method):
```python
def wait(self, count: int, timeout: float | None = None) -> TensorDict:
    current_ver = self.inference_engine.get_version()

    # Step 1: Purge stale samples from queue (segment-wise PPO)
    if self.staleness_strategy:
        self._last_purged_version = self.staleness_strategy.purge_stale_samples_from_queue(
            output_queue=self.output_queue,
            current_ver=current_ver,
            last_purged_ver=self._last_purged_version,
            inference_engine=self.inference_engine,
            result_cache=self.result_cache,
            config=self.config,
            logger=self.logger,
        )

    # ... existing wait logic ...

    # Step 3: Filter stale from cache (segment-wise PPO)
    if self.staleness_strategy:
        dropped_cache = self.staleness_strategy.filter_stale_from_cache(
            result_cache=self.result_cache,
            current_ver=current_ver,
            config=self.config,
            logger=self.logger,
        )
        if dropped_cache > 0 and self.filtered_capacity_modifier:
            self.filtered_capacity_modifier.on_samples_filtered(dropped_cache, current_ver)
```

#### 4.2 areal/core/remote_inf_engine.py

Change to use factory:
```python
from areal.api.workflow_factory import create_workflow_executor

# In initialize():
self.workflow_executor = create_workflow_executor(
    inference_engine=self,
    config=self.config,
    logger=logger,
    train_data_parallel_size=train_data_parallel_size,
)
```

### Phase 5: Integration with AsyncTaskRunner

Need to understand where in the new code structure to add our modifications:

1. **Task submission** → Add `on_rollout_submitted()` call
2. **Task completion** → Add staleness filtering logic
3. **Result collection** → Add queue purge and cache filtering

Key question: Does new code use `AsyncTaskRunner`? Need to check.

### Phase 6: Testing Strategy

1. **Unit tests** (quick validation):
   ```bash
   pytest areal/tests/sdp/test_staleness_control.py -v
   ```

2. **Integration tests**:
   ```bash
   pytest areal/tests/sdp/test_workflow_api_modifications.py -v
   ```

3. **Full SDP suite**:
   ```bash
   pytest areal/tests/sdp/ -v
   ```

4. **All tests**:
   ```bash
   pytest areal/tests/ -v
   ```

### Phase 7: Verification

1. Import test
2. Configuration test
3. Factory test
4. Feature integration test

## Risk Mitigation

- **Backup**: `backup/segment-wise-ppo-v2-before-integration` branch exists
- **Incremental**: Port files incrementally, test after each group
- **Documentation**: Track all changes in this document
- **Rollback**: Can always return to old branch if needed

## Success Criteria

1. ✅ All 227 SDP tests pass
2. ✅ New async_task_runner tests pass
3. ✅ No regressions in existing tests
4. ✅ Feature works correctly
5. ✅ Clean git history

## Next Steps

1. Create new feature branch from latest main
2. Port new files (no conflicts)
3. Port simple modifications
4. Understand new workflow_executor structure
5. Manually integrate staleness logic
6. Test incrementally
7. Verify and commit

**This approach prioritizes code quality and maintainability over speed.**
