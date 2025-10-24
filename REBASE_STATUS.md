# Rebase Status: Segment-Wise PPO Feature

## Situation

Main branch has undergone **MAJOR REFACTORING** (commits up to f8f2ea39):

### Key Structural Changes in Main:

1. **WorkflowExecutor Move** (commit 09f1003e):
   - `areal/api/workflow_api.py` → `areal/core/workflow_executor.py`
   - Only `RolloutWorkflow` remains in `workflow_api.py`

2. **New RemoteInfEngine Abstraction** (commit ccba1bb7 + af64fa1e):
   - Created `areal/core/remote_inf_engine.py`
   - Introduces `RemoteInfEngine` base class
   - Introduces `RemoteInfBackendProtocol` for backends (SGLang, vLLM)
   - SGLang/vLLM engines refactored to use this new structure

3. **connect_engine API** (commit 09f1003e):
   - Simplified rollout in training scripts
   - New API for connecting to inference engines

4. **Code Quality** (commits 189d93fb, f8f2ea39):
   - Added ruff pre-commit hooks
   - CI formatting checks

### Impact on Our Feature:

Our segment-wise PPO feature touches:
- ✅ **Low conflict**: Core logic files (staleness_control.py, staleness_manager.py, etc.)
- ✅ **Low conflict**: New files we created (workflow_factory.py, proximal_recomputer.py, etc.)
- ⚠️ **HIGH CONFLICT**: WorkflowExecutor modifications (moved to new location)
- ⚠️ **HIGH CONFLICT**: Engine files (sglang_remote.py, vllm_remote.py - completely refactored)
- ⚠️ **MEDIUM CONFLICT**: Import statements (many files import from old locations)

## Recommended Approach

Given the extent of main's refactoring, **simple rebase/merge is not viable**. Instead:

### Option A: Fresh Integration (RECOMMENDED)
1. Checkout main as new branch
2. Copy our feature files that don't conflict:
   - `areal/api/staleness_control.py`
   - `areal/api/proximal_recomputer.py`
   - `areal/api/cache_api.py`
   - `areal/api/queue_api.py`
   - `areal/core/filtered_capacity_modifier.py`
   - All test files under `areal/tests/sdp/`
3. Manually integrate WorkflowExecutor changes into `areal/core/workflow_executor.py`
4. Create new workflow_factory.py that works with RemoteInfEngine
5. Update engine files to use new factory

### Option B: Incremental Rebase
1. Rebase commit-by-commit
2. Resolve conflicts manually for each commit
3. Adapt each commit to new structure
4. Very time-consuming but preserves history

### Option C: Wait for Stabilization
1. Wait for main to stabilize
2. Communicate with main branch maintainers
3. Coordinate integration timeline

## Complexity Analysis

### Files Requiring Manual Adaptation:

1. **areal/core/workflow_executor.py** (NEW LOCATION)
   - Apply our WorkflowExecutor modifications
   - Changes:
     * Constructor: Add queue/cache/strategy/recomputer/modifier injection
     * initialize(): Add train_data_parallel_size, DP scaling logic
     * _rollout_thread_async(): Add staleness filtering, proximal recomputation
     * wait(): Add queue purging, cache filtering

2. **areal/api/workflow_factory.py** (OUR FILE)
   - Update to work with RemoteInfEngine structure
   - Change imports from workflow_api to core
   - Ensure factory works with new engine abstraction

3. **areal/core/remote_inf_engine.py** (MAIN'S NEW FILE)
   - Understand how it initializes WorkflowExecutor
   - May need to modify to use our factory

4. **areal/engine/sglang_remote.py** (HEAVILY REFACTORED)
   - Main moved most logic to RemoteInfEngine
   - Our factory integration needs to be adapted
   - SGLangBackend class structure changed

5. **areal/engine/vllm_remote.py** (HEAVILY REFACTORED)
   - Similar to sglang_remote.py
   - Needs adaptation to new structure

### Estimated Effort:

- **Option A**: 3-4 hours of focused work
- **Option B**: 6-8 hours (tedious, error-prone)
- **Option C**: Unknown timeline

## Decision

**Proceeding with Option A**: Fresh integration approach

Reasons:
1. Cleaner final result
2. Easier to verify correctness
3. Less error-prone than conflict resolution
4. Better understanding of main's new architecture
5. Can leverage main's improvements

## Next Steps

1. Create new branch from main: `feature/segment-wise-ppo-v2`
2. Copy non-conflicting feature files
3. Integrate WorkflowExecutor changes
4. Adapt factory to new structure
5. Update tests
6. Verify functionality

---

**Status**: Strategy decided, ready to proceed with fresh integration
**Estimated Time**: 3-4 hours
**Risk**: Medium (need to understand new RemoteInfEngine architecture)
