# CUDA OOM Fix - StalenessManager DP Scaling

## Problem

Training script that worked correctly in main branch was causing CUDA OOM errors in the `feature/segment-wise-decoupled-ppo-rebased` branch.

## Root Cause

In the feature branch refactoring, `StalenessManager` was created in `areal/engine/sglang_remote.py:124-128` without applying data parallel (DP) scaling:

```python
# WRONG - No DP scaling applied
staleness_manager = StalenessManager(
    max_concurrent_rollouts=self.config.max_concurrent_rollouts or self.config.consumer_batch_size,
    consumer_batch_size=self.config.consumer_batch_size,
    max_staleness=self.config.max_head_offpolicyness,
)
```

This manager was then passed to the factory, which passed it to `WorkflowExecutor.__init__()`. When `initialize()` was called, it saw that `self.staleness_manager` was already set (not None), so it **skipped** the DP scaling logic in lines 343-365 of `workflow_api.py`.

### Impact

With 4 GPUs (`dp_world_size=4`) and `max_concurrent_rollouts=64`:

**Main branch behavior:**
- `initialize()` creates StalenessManager with capacity: `max(1, 64 // 4) = 16` per GPU
- Total rollouts across 4 GPUs: 16 × 4 = 64 ✓
- Memory usage: Normal

**Feature branch behavior (BEFORE fix):**
- StalenessManager created with capacity: 64 per GPU
- Total rollouts across 4 GPUs: 64 × 4 = **256** ✗
- Memory usage: **4x higher → CUDA OOM!**

## Solution

### Changes Made

#### 1. `areal/api/workflow_factory.py`

**Made `staleness_manager` parameter optional:**
```python
def create_workflow_executor(
    inference_engine: "InferenceEngine",
    config: "InferenceEngineConfig",
    logger: Any,
    staleness_manager: "StalenessManager | None" = None,  # Now optional
    train_data_parallel_size: int | None = None,          # New parameter
) -> "WorkflowExecutor":
```

**Pass `train_data_parallel_size` to `initialize()`:**
```python
# Initialize the executor (starts worker thread and creates staleness_manager if None)
# Pass train_data_parallel_size for proper DP scaling
executor.initialize(logger=logger, train_data_parallel_size=train_data_parallel_size)

# Register capacity modifiers after staleness_manager is created
if staleness_manager is None and executor.staleness_manager is not None:
    register_capacity_modifiers(executor.staleness_manager, filtered_capacity_modifier)
```

#### 2. `areal/engine/sglang_remote.py`

**Removed StalenessManager creation:**
```python
# Create workflow executor using factory
# StalenessManager will be created in initialize() with proper DP scaling
self.workflow_executor = create_workflow_executor(
    inference_engine=self,
    config=self.config,
    logger=self.logger,
    train_data_parallel_size=train_data_parallel_size,  # Pass DP size
)
```

**Removed unused import:**
```python
# Removed: from areal.core.staleness_manager import StalenessManager
```

### How It Works Now

1. `sglang_remote.py` calls factory with `staleness_manager=None` (default) and `train_data_parallel_size`
2. Factory creates `WorkflowExecutor` with `staleness_manager=None`
3. Factory calls `executor.initialize(train_data_parallel_size=...)`
4. `initialize()` sees `self.staleness_manager is None`, so it creates one with DP scaling:
   ```python
   max_concurrent_rollouts = max(1, self.max_concurrent_rollouts // dp_world_size)
   consumer_batch_size = max(1, self.consumer_batch_size // dp_world_size)

   self.staleness_manager = StalenessManager(
       max_concurrent_rollouts=max_concurrent_rollouts,  # Scaled by DP size
       consumer_batch_size=consumer_batch_size,          # Scaled by DP size
       max_staleness=self.config.max_head_offpolicyness,
   )
   ```
5. Factory registers capacity modifiers with the newly created manager

## Design Benefits

### Follows Dependency Injection Principles

The fix adheres to the user's architectural guidelines:

1. **Factory controls creation**: The factory decides when and how to create `StalenessManager`
2. **No monkey patching**: Clean extension via optional parameters
3. **Prepared for scaling**: The design now properly handles distributed training
4. **Backward compatible**: If caller provides pre-scaled `staleness_manager`, it still works

### Separation of Concerns

- **Engine layer** (`sglang_remote.py`): Provides configuration and DP size
- **Factory layer** (`workflow_factory.py`): Orchestrates component assembly
- **Workflow layer** (`workflow_api.py`): Contains DP scaling business logic

Each layer has clear responsibilities without cross-layer knowledge.

## Testing

After applying this fix:
1. **Memory usage**: Should match main branch behavior
2. **Capacity scaling**: Each GPU should get `max_concurrent_rollouts / dp_world_size`
3. **No OOM errors**: Training should complete successfully

## Related Issues Fixed in This Session

1. ✓ Worker thread not starting → Fixed by calling `initialize()` in factory
2. ✓ Logger AttributeError → Fixed by initializing logger in `__init__`
3. ✓ `_TimedResult` type mismatch → Fixed in staleness filtering code
4. ✓ **CUDA OOM** → Fixed by restoring DP scaling (this document)

## Lessons Learned

### When Refactoring, Preserve Original Behavior

The original `WorkflowExecutor` design was:
```python
# Old code (main branch):
executor = WorkflowExecutor(config=..., inference_engine=...)
executor.initialize(train_data_parallel_size=...)  # Creates manager with scaling
```

The refactoring introduced early manager creation, which bypassed the DP scaling logic. The fix restores the original creation pattern while maintaining the new factory abstraction.

### Test with Realistic Configurations

Unit tests may not catch scaling issues. The OOM only appeared when:
- Running on multiple GPUs (dp_world_size > 1)
- With realistic batch sizes
- In actual training (not unit tests)

Always test distributed training changes on actual GPU clusters.

---

**Commit:** 16f6238e
**Branch:** feature/segment-wise-decoupled-ppo-rebased
**Status:** ✅ Fixed
