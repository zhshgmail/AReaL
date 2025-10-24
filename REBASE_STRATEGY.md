# Rebase Strategy: Segment-Wise PPO onto Main

## Problem

Main branch has refactored the code structure (commit 09f1003e):
- **Moved** `WorkflowExecutor` from `areal/api/workflow_api.py` → `areal/core/workflow_executor.py`
- **Moved** `check_trajectory_format` from `areal/api/workflow_api.py` → `areal/core/workflow_executor.py`
- **Kept** `RolloutWorkflow` in `areal/api/workflow_api.py`
- **Updated** imports in `sglang_remote.py`, `vllm_remote.py`, `sglang_engine.py`

Our feature branch has modifications to `WorkflowExecutor` in the OLD location (`areal/api/workflow_api.py`).

## Conflicting Files

1. `areal/api/workflow_api.py` - Main removed WorkflowExecutor, we modified it
2. `areal/engine/sglang_remote.py` - Both modified (imports + our factory changes)
3. `areal/engine/vllm_remote.py` - Both modified (imports)
4. `areal/experimental/sglang_engine.py` - Both modified (imports)

## Our Key Modifications to WorkflowExecutor

### Changes Made in Feature Branch:

1. **New dependencies injected into `__init__`**:
   - `output_queue`: `RolloutQueue`
   - `result_cache`: `RolloutCache`
   - `staleness_strategy`: `StalenessControlStrategy | None`
   - `proximal_recomputer`: `ProximalRecomputer | None`
   - `filtered_capacity_modifier`: `FilteredSamplesCapacityModifier | None`

2. **Modified `initialize()` method**:
   - Removed logger parameter requirement (initialize with default if None)
   - Added `train_data_parallel_size` parameter
   - Only creates `StalenessManager` if `self.staleness_manager is None`
   - Applies DP scaling when creating StalenessManager

3. **Modified `_rollout_thread_async()` method**:
   - Uses injected `output_queue` instead of `self.output_queue`
   - Uses injected `result_cache` instead of `self.result_cache`
   - Calls `staleness_strategy.should_enqueue_sample()` before enqueuing
   - Calls `proximal_recomputer.add_proximal_logprobs()` if needed

4. **Modified `wait()` method**:
   - Calls `staleness_strategy.purge_stale_samples_from_queue()`
   - Uses injected `output_queue` and `result_cache`
   - Calls `staleness_strategy.filter_stale_from_cache()`
   - Updates `filtered_capacity_modifier` when samples filtered

5. **Removed debug logging** (in commit 046d56d0)

## Strategy

### Phase 1: Manual Conflict Resolution

1. **For `areal/api/workflow_api.py`**:
   - Accept main's version (only RolloutWorkflow + check_trajectory_format)
   - Our WorkflowExecutor changes go elsewhere

2. **For engine files** (`sglang_remote.py`, `vllm_remote.py`, `sglang_engine.py`):
   - Accept main's import changes
   - Keep our factory-related changes
   - Update imports: `from areal.core import WorkflowExecutor`

### Phase 2: Apply WorkflowExecutor Changes to New Location

Since WorkflowExecutor is now in `areal/core/workflow_executor.py`, we need to:

1. Check out main's version of `areal/core/workflow_executor.py`
2. Apply our modifications from `areal/api/workflow_api.py` to `areal/core/workflow_executor.py`
3. Update all imports in our feature files

### Phase 3: Update workflow_factory.py

Our `areal/api/workflow_factory.py` imports from:
```python
from areal.api.workflow_api import WorkflowExecutor
```

Need to change to:
```python
from areal.core import WorkflowExecutor
```

### Phase 4: Verification

1. Check that all imports are correct
2. Run unit tests
3. Verify no circular dependencies
4. Test example configs

## Files That Need Import Updates

Files that import WorkflowExecutor:
- `areal/api/workflow_factory.py` (our file)
- `areal/engine/sglang_remote.py` (already updated by main)
- `areal/engine/vllm_remote.py` (already updated by main)
- `areal/experimental/sglang_engine.py` (already updated by main)
- `areal/tests/sdp/test_workflow_api_modifications.py` (our test)

## Execution Plan

1. ✅ Abort current rebase
2. ⏳ Try rebase again with strategy
3. ⏳ Resolve conflicts file by file
4. ⏳ Apply WorkflowExecutor modifications to new location
5. ⏳ Update imports
6. ⏳ Test and verify

---

**Current Status**: Strategy defined, ready to execute rebase
