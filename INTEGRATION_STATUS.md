# Segment-Wise PPO Integration Status

**Branch**: `feature/segment-wise-ppo-v2`
**Base**: origin/main (f8f2ea39)
**Strategy**: Fresh integration (Option A)

## Completed ✅ (Commits: f1a5c8cd, 31ee3b33)

### Phase 1: Core Files ✅
- ✅ `areal/api/cache_api.py` - RolloutCache abstraction
- ✅ `areal/api/queue_api.py` - RolloutQueue abstraction
- ✅ `areal/api/staleness_control.py` - Strategy pattern (Standard + SegmentWise)
- ✅ `areal/api/proximal_recomputer.py` - Proximal logprob recomputation
- ✅ `areal/core/filtered_capacity_modifier.py` - Dynamic capacity adjustment
- ✅ All test files (`areal/tests/sdp/*`, `conftest.py`, `test_model_utils.py`)

### Phase 2: Configuration ✅
- ✅ `BaseExperimentConfig.enable_segment_wise_ppo` (cli_args.py:1194)
- ✅ `InferenceEngineConfig.enable_segment_wise_ppo` (cli_args.py:847)
- ✅ `PPOActorConfig.enable_segment_wise_ppo` (cli_args.py:482)
- ✅ Auto-propagation logic in `load_expr_config()` (cli_args.py:1316)

### Architecture Decision ✅
- ✅ **Keep StalenessControlStrategy separate from StalenessManager**
  - Documented in `ARCHITECTURE_ANALYSIS_STALENESS.md`
  - Follows SOLID principles, Strategy pattern
  - Correct architectural layering

## Remaining Work ⏳

### Phase 3: I/O Struct & Core Components (30 min)

#### 1. Add fields to `areal/api/io_struct.py`

**In `FinetuneSpec` dataclass** (find around line 50-100):
```python
# Segment-wise PPO fields (optional, only present if enabled)
proximal_logprobs_t: torch.Tensor | None = None  # [batch, seq_len, 2]
    # [:, :, 0] = token_id (0 if not recomputed)
    # [:, :, 1] = proximal logprob
output_versions: torch.Tensor | None = None      # [batch, seq_len]
_recompute_version: int | None = None            # Metadata for recomputation
```

#### 2. Update `areal/core/capacity_modifier.py`

Check if base class exists. If NOT, add:
```python
from abc import ABC, abstractmethod

class CapacityModifier(ABC):
    """Base class for modifying staleness manager capacity dynamically."""

    @abstractmethod
    def get_capacity_delta(self, version: int) -> int:
        """Return capacity adjustment for given version."""
        pass
```

#### 3. Update `areal/core/staleness_manager.py`

Add capacity modifier support:
```python
# In __init__:
self.capacity_modifiers: List[CapacityModifier] = []

# Add method:
def register_capacity_modifier(self, modifier: CapacityModifier):
    """Register a capacity modifier."""
    self.capacity_modifiers.append(modifier)

# Modify get_capacity():
def get_capacity(self, version: int) -> int:
    # ... existing logic ...
    delta = sum(mod.get_capacity_delta(version) for mod in self.capacity_modifiers)
    return max(0, base_capacity + delta)
```

### Phase 4: Factory Pattern (1 hour)

#### 4. Create `areal/api/workflow_factory.py`

**CRITICAL**: Must work with main's RemoteInfEngine structure

Copy from old branch and adapt:
- Change imports: `from areal.core import WorkflowExecutor`
- Keep all helper functions (`create_queue`, `create_cache`, etc.)
- Main function: `create_workflow_executor()` takes inference_engine, config, logger, staleness_manager, train_data_parallel_size
- Returns fully configured WorkflowExecutor

### Phase 5: WorkflowExecutor Integration (1 hour - MOST CRITICAL)

#### 5. Modify `areal/core/workflow_executor.py`

**This is the key integration point!**

Changes needed:

**A. Imports** (add at top):
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areal.api.cache_api import RolloutCache
    from areal.api.queue_api import RolloutQueue
    from areal.api.staleness_control import StalenessControlStrategy
    from areal.api.proximal_recomputer import ProximalRecomputer
    from areal.core.filtered_capacity_modifier import FilteredSamplesCapacityModifier
```

**B. Constructor** (`__init__` method):
```python
def __init__(
    self,
    inference_engine,
    config,
    staleness_manager=None,
    output_queue: "RolloutQueue | None" = None,  # ← ADD
    result_cache: "RolloutCache | None" = None,  # ← ADD
    staleness_strategy: "StalenessControlStrategy | None" = None,  # ← ADD
    proximal_recomputer: "ProximalRecomputer | None" = None,  # ← ADD
    filtered_capacity_modifier: "FilteredSamplesCapacityModifier | None" = None,  # ← ADD
    logger=None,
):
    # ... existing code ...

    # Use injected or create defaults
    from areal.api.queue_api import LocalRolloutQueue
    from areal.api.cache_api import LocalRolloutCache

    self.output_queue = output_queue or LocalRolloutQueue(maxsize=queue_size)
    self.result_cache = result_cache or LocalRolloutCache()
    self.staleness_strategy = staleness_strategy
    self.proximal_recomputer = proximal_recomputer
    self.filtered_capacity_modifier = filtered_capacity_modifier
    self._last_purged_version = -1  # Track last purge
```

**C. initialize()** method:
Add `train_data_parallel_size` parameter and DP scaling logic.

**D. _rollout_thread_async()** method:
Add staleness filtering and proximal recomputation before enqueue.

**E. wait()** method:
Add queue purging and cache filtering logic.

**Full details in INTEGRATION_PROGRESS.md sections 6.3-6.5**

### Phase 6: RemoteInfEngine Integration (30 min)

#### 6. Update `areal/core/remote_inf_engine.py`

Find where `WorkflowExecutor` is created (likely in `__init__` or `initialize`).

**Replace**:
```python
self.workflow_executor = WorkflowExecutor(...)
```

**With**:
```python
from areal.api.workflow_factory import create_workflow_executor

self.workflow_executor = create_workflow_executor(
    inference_engine=self,
    config=config,
    logger=self.logger,
    train_data_parallel_size=train_data_parallel_size,
)
```

### Phase 7: Training Integration (30 min)

#### 7. Copy training-side changes

From `feature/segment-wise-decoupled-ppo-rebased`:
```bash
git checkout feature/segment-wise-decoupled-ppo-rebased -- \
  realhf/impl/model/utils/ppo_functional.py \
  realhf/impl/model/interface/ppo_interface.py \
  areal/workflow/rlvr.py \
  areal/workflow/vision_rlvr.py \
  areal/engine/ppo/actor.py \
  areal/utils/model.py \
  areal/utils/functional.py
```

#### 8. Copy example config
```bash
git checkout feature/segment-wise-decoupled-ppo-rebased -- \
  examples/math/gsm8k_grpo_segment_wise.yaml
```

### Phase 8: Testing & Verification (1 hour)

#### 9. Update test imports

All tests in `areal/tests/sdp/` need:
```python
from areal.core import WorkflowExecutor  # Not from areal.api.workflow_api
```

Files to update:
- `areal/tests/sdp/test_workflow_api_modifications.py`
- Any test that imports WorkflowExecutor

#### 10. Run tests
```bash
# All SDP tests
pytest areal/tests/sdp/ -v

# Model utils tests
pytest areal/tests/test_model_utils.py -v

# Specific critical tests
pytest areal/tests/sdp/test_staleness_control.py -v
pytest areal/tests/sdp/test_workflow_api_modifications.py -v
```

#### 11. Verify feature toggle
```bash
# Test with feature enabled (default)
python -c "from areal.api.cli_args import GRPOConfig; c = GRPOConfig(experiment_name='test', trial_name='t'); print(c.enable_segment_wise_ppo)"

# Should print: True
```

## Success Criteria

1. ✅ Core feature files copied
2. ✅ Configuration flags added with auto-propagation
3. ⏳ WorkflowExecutor modifications in `areal/core/workflow_executor.py`
4. ⏳ Factory works with RemoteInfEngine
5. ⏳ All tests pass on Windows with mocked uvloop
6. ⏳ Feature can be toggled via `enable_segment_wise_ppo=false`
7. ⏳ No code in `areal/api/workflow_api.py` except `RolloutWorkflow`

## Estimated Time Remaining

- Phase 3 (I/O Struct): 30 min
- Phase 4 (Factory): 1 hour
- Phase 5 (WorkflowExecutor): 1 hour ← CRITICAL
- Phase 6 (RemoteInfEngine): 30 min
- Phase 7 (Training): 30 min
- Phase 8 (Testing): 1 hour

**Total**: ~4.5 hours

## Next Steps (In Order)

1. Add proximal_logprobs_t fields to io_struct.py
2. Update capacity_modifier.py (add base class if missing)
3. Update staleness_manager.py (add register_capacity_modifier)
4. Create workflow_factory.py
5. Integrate WorkflowExecutor changes (CRITICAL - most complex)
6. Update RemoteInfEngine to use factory
7. Copy training-side files
8. Update test imports
9. Run tests and verify

## Key Files to Reference

- `FEATURE_SUMMARY.md` - Complete feature overview
- `INTEGRATION_PROGRESS.md` - Detailed implementation guide
- `ARCHITECTURE_ANALYSIS_STALENESS.md` - Design rationale
- Old branch: `feature/segment-wise-decoupled-ppo-rebased` - Reference implementation

---

**Current Branch**: `feature/segment-wise-ppo-v2`
**Last Commit**: 31ee3b33 (config flags)
**Next**: Add I/O struct fields (Phase 3)
