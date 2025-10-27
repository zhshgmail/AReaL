# Segment-wise Decoupled PPO Rebase Plan

## Overview
This document tracks the migration of segment-wise decoupled PPO from `dev_seg_decouple_rebase` to the latest main branch architecture in `dev_seg_decouple_rebase_v2`.

## Completed Work ✅

### Phase 1: Core Data Structures & Configuration
- [x] Add `enable_segment_wise_ppo` flag to `InferenceEngineConfig`
- [x] Add `proximal_logprobs_t` field to `ModelResponse`
- [x] Add `behav_imp_weight_floor` parameter to `actor_loss_fn`
- [x] Update `realhf` PPO functional with symmetric/asymmetric clipping

### Phase 2: Documentation & Examples
- [x] Copy `docs/algorithms/segment_wise_ppo.md`
- [x] Copy `examples/math/gsm8k_grpo_sdp.yaml`

### Phase 3: Workflows & Loss Functions
- [x] Update `areal/workflow/rlvr.py` to conditionally pass `proximal_logprobs_t`
- [x] Update `areal/workflow/vision_rlvr.py` similarly
- [x] Update `areal/utils/functional.py` loss computation
- [x] Update PPO interfaces in `realhf/impl/model/interface/ppo_interface.py`
- [x] Update `realhf/api/cli_args.py` and `realhf/api/core/model_api.py`

### Phase 4: Test Suite
- [x] Copy all 10 test files to `areal/tests/seg_decoupled_ppo/`
- [x] Tests for: behav_imp_weight_floor, GRPO loss, integration, functional coverage, proximal_t generation, recompute timing, executor wait semantics

### Phase 5: Basic Engine Infrastructure
- [x] Add `logprob_start_len` to SGLang backend payload
- [x] Add `recompute_output_logprobs_sync` to `SGLangBackend`

## Remaining Work 🚧

### Phase 6: Engine Generation with proximal_t Tracking

#### 6.1 RemoteInfEngine Generation Loop (`areal/core/remote_inf_engine.py`)
**Status**: NOT STARTED
**Priority**: HIGH (blocking core functionality)

**Tasks**:
- [ ] Add proximal_logprobs_t initialization in `agenerate()` when `config.enable_segment_wise_ppo=True`
- [ ] Track proximal_t during generation loop:
  - [ ] On first iteration: Initialize proximal_t with output_logprobs
  - [ ] On abort-resume iterations: Use input_logprobs to update previous tokens' proximal_t
- [ ] Pass proximal_t to ModelResponse construction
- [ ] Handle edge cases (empty generation, interrupt, etc.)

**Original Implementation Reference**:
- Feature branch: `dev_seg_decouple_rebase:areal/api/workflow_api.py` lines 220-370
- Look for: `proximal_logprobs_t` tracking, `logprob_start_len` updates

**Complexity**: HIGH - requires understanding async generation state machine

#### 6.2 vLLM Backend Support
**Status**: NOT STARTED
**Priority**: MEDIUM

**Tasks**:
- [ ] Add similar `logprob_start_len` support to vLLM backend (if applicable)
- [ ] Add `recompute_output_logprobs_sync` to vLLM backend
- [ ] Test proximal_t generation with vLLM engine

**Note**: vLLM may not support `logprob_start_len` - need to investigate alternatives

#### 6.3 Experimental Engine
**Status**: NOT STARTED
**Priority**: LOW

**Tasks**:
- [ ] Update `areal/experimental/sglang_engine.py` if used in tests
- [ ] Ensure compatibility with new proximal_t tracking

### Phase 7: Recompute Architecture (DESIGN DECISION NEEDED)

#### Current Feature Branch Approach (Hook-based)
The original implementation in `dev_seg_decouple_rebase` uses:
- Hook system: `pre_pause_hooks`, `post_pause_hooks`, `pre_resume_hooks`, `post_resume_hooks`
- Hooks registered in WorkflowExecutor
- Recompute logic triggered in `pause()` method via pre_pause hook
- Files: 
  - `areal/api/workflow_components.py` - Hook management, Strategy pattern
  - `areal/api/proximal_recomputer.py` - Recompute logic
  - `areal/api/staleness_control.py` - Staleness control strategy

**Hook System Operations**:
1. Iterate through `output_queue` and `result_cache`
2. For each sample, check if any tokens have `version == current_version - 1`
3. If yes, call `engine.recompute_output_logprobs_sync()` to get new logprobs
4. Patch the `proximal_logprobs_t` field for those specific token positions
5. Filter out samples that are too stale

#### Proposed New Approach (Filter/Transformer-based)

**Key Insight**: The operations are primarily data transformations on Queue/Cache:
- **Filter**: Remove over-stale samples (staleness control)
- **Transform**: Update proximal_t for v-1 samples (recompute)

**Proposed Architecture**:

```python
class QueueTransformer(Protocol):
    """Transform/filter items in queue or cache."""
    def apply(self, items: List[Any], engine: InferenceEngine) -> List[Any]:
        """Apply transformation/filtering to items."""
        ...

class ProximalRecomputer(QueueTransformer):
    """Recompute proximal_t for samples with version == current_version - 1."""
    def apply(self, items: List[Dict], engine: InferenceEngine) -> List[Dict]:
        current_version = engine.get_version()
        for item in items:
            if self._has_v_minus_1_tokens(item, current_version):
                self._recompute_proximal_t(item, engine)
        return items

class StalenessFilter(QueueTransformer):
    """Filter out over-stale samples."""
    def __init__(self, max_staleness: int):
        self.max_staleness = max_staleness
    
    def apply(self, items: List[Dict], engine: InferenceEngine) -> List[Dict]:
        current_version = engine.get_version()
        return [
            item for item in items 
            if self._get_sample_version(item) >= current_version - self.max_staleness
        ]

# In WorkflowExecutor or AsyncTaskRunner:
class WorkflowExecutor:
    def __init__(self, ..., queue_transformers: List[QueueTransformer] = None):
        self.queue_transformers = queue_transformers or []
    
    def pause(self):
        # Before pause, apply all transformers to cache
        for transformer in self.queue_transformers:
            self.result_cache = transformer.apply(self.result_cache, self.engine)
        # Then pause engine
        ...
```

**Benefits**:
1. **Separation of Concerns**: Queue operations separated from workflow control
2. **Composability**: Can chain multiple transformers
3. **Testability**: Each transformer testable in isolation
4. **Flexibility**: Easy to add new transformers (e.g., data augmentation, validation)
5. **No Hooks**: Cleaner than hook system, explicit control flow

**Tasks**:
- [ ] **DESIGN DECISION**: Review and approve filter/transformer architecture
- [ ] Create `areal/core/queue_transformer.py` with base protocol
- [ ] Implement `ProximalRecomputer` transformer
- [ ] Implement `StalenessFilter` transformer
- [ ] Integrate transformers into `WorkflowExecutor.pause()` or `wait()`
- [ ] Update tests to work with new architecture

**Complexity**: MEDIUM - cleaner design than hooks but requires refactoring

### Phase 8: Import & Reference Fixes

**Status**: NOT STARTED
**Priority**: HIGH (blocking tests)

**Tasks**:
- [ ] Run tests and identify import errors
- [ ] Fix references to moved classes (e.g., `areal.api.workflow_api.WorkflowExecutor` → `areal.core.WorkflowExecutor`)
- [ ] Update test imports for new module structure
- [ ] Verify `areal/core/__init__.py` exports

**Estimated Issues**:
- Tests importing from old `areal.api.workflow_api`
- References to hook system that doesn't exist yet
- Missing exports in `areal/core/__init__.py`

### Phase 9: CPU-Only Testing & Validation

**Status**: NOT STARTED
**Priority**: HIGH

**Tasks**:
- [ ] Run unit tests: `python -m pytest areal/tests/seg_decoupled_ppo/test_ppo_functional_coverage.py -v`
- [ ] Run integration tests (may need GPU mocking)
- [ ] Test with `enable_segment_wise_ppo=False` (backward compatibility)
- [ ] Test with `enable_segment_wise_ppo=True` (new feature)
- [ ] Verify loss computation correctness
- [ ] Check for performance regressions

**Expected Failures**:
- Import errors (Phase 8)
- Missing recompute logic (Phase 7)
- Engine generation issues (Phase 6)

### Phase 10: Cross-Reference with boba-tmp

**Status**: NOT STARTED  
**Priority**: MEDIUM

**Tasks**:
- [ ] Compare recompute logic between `boba-tmp` and `dev_seg_decouple_rebase`
- [ ] Verify original prototype intent preserved
- [ ] Check for any missing edge cases
- [ ] Validate that π_proximal_t = π_{v+1} semantics are correct

**Reference Commit**: boba-tmp branch (simpler prototype without Strategy pattern)

## Architecture Comparison

### Old Feature Branch (dev_seg_decouple_rebase)
```
WorkflowExecutor (areal/api/workflow_api.py)
├── Hook System (pre_pause, post_pause, pre_resume, post_resume)
├── Strategy Pattern (areal/api/workflow_components.py)
│   ├── StalenessControlStrategy
│   ├── ProximalRecomputer
│   └── StandardPPOStrategy (factory)
├── Manages: input_queue, output_queue, result_cache
└── Async event loop (uvloop) in background thread
```

### New Main Branch
```
WorkflowExecutor (areal/core/workflow_executor.py)
├── Delegates to AsyncTaskRunner (generic async executor)
│   ├── Manages: input_queue, output_queue, result_cache
│   └── No business logic, pure async task runner
├── Uses StalenessManager (areal/core/staleness_manager.py)
│   └── Capacity control based on staleness
└── Uses RemoteInfEngine (areal/core/remote_inf_engine.py)
    └── Backend protocol (SGLang, vLLM)
```

### Proposed Integrated Architecture
```
WorkflowExecutor (areal/core/workflow_executor.py)
├── Delegates to AsyncTaskRunner
│   ├── input_queue, output_queue, result_cache
│   └── QueueTransformers (NEW)
│       ├── ProximalRecomputer (apply before pause)
│       └── StalenessFilter (apply on output)
├── Uses StalenessManager
└── Uses RemoteInfEngine
    └── Backend with recompute_output_logprobs_sync
```

## Migration Strategy

### Phase 1: Quick Wins (Current Phase - Option 2)
1. Fix imports and basic compatibility
2. Add proximal_t tracking to RemoteInfEngine generation
3. Stub out recompute (manual trigger for now)
4. Get basic tests passing

### Phase 2: Filter Architecture (After Design Approval)
1. Implement QueueTransformer protocol
2. Port recompute logic to ProximalRecomputer transformer
3. Port staleness filtering to StalenessFilter transformer
4. Integrate with WorkflowExecutor
5. Remove hook system references from tests

### Phase 3: Full Integration
1. Add transformer composition
2. Test with real training loop
3. Performance benchmarking
4. Documentation updates

## Key Decisions Needed

1. **Filter vs Hook Architecture**: Approve filter/transformer design (recommended)
2. **Transformer Integration Point**: Where to apply transformers?
   - Option A: In `WorkflowExecutor.pause()` before engine pause
   - Option B: In `WorkflowExecutor.wait()` when draining queue
   - Option C: Both (recompute in pause, staleness filter in wait)
3. **Backward Compatibility**: Ensure `enable_segment_wise_ppo=False` works without transformers

## Testing Strategy

### Unit Tests (Per Component)
- [ ] Test `ProximalRecomputer` transformer in isolation
- [ ] Test `StalenessFilter` transformer in isolation
- [ ] Test engine `recompute_output_logprobs_sync` method
- [ ] Test proximal_t tracking during generation

### Integration Tests
- [ ] Test full workflow with segment-wise PPO enabled
- [ ] Test backward compatibility with feature disabled
- [ ] Test staleness control with recompute
- [ ] Test multi-turn generation with version tracking

### Validation Tests
- [ ] Verify π_proximal_t = π_{v+1} semantics
- [ ] Check behavioral importance weight variance reduction
- [ ] Compare with standard PPO (should match when disabled)

## Files Modified So Far

### Configuration & Data
- `areal/api/cli_args.py` - Added `enable_segment_wise_ppo`
- `areal/api/io_struct.py` - Added `proximal_logprobs_t` to ModelResponse
- `realhf/api/cli_args.py` - Added `behav_imp_weight_floor`

### Loss & Algorithms  
- `realhf/impl/model/utils/ppo_functional.py` - Added floor parameter
- `realhf/impl/model/interface/ppo_interface.py` - Interface updates
- `realhf/api/core/model_api.py` - Model API updates
- `areal/utils/functional.py` - Loss computation updates

### Workflows
- `areal/workflow/rlvr.py` - Proximal_t passing
- `areal/workflow/vision_rlvr.py` - Proximal_t passing
- `areal/engine/ppo/actor.py` - Actor updates

### Engine
- `areal/engine/sglang_remote.py` - logprob_start_len, recompute method

### Documentation & Tests
- `docs/algorithms/segment_wise_ppo.md` - Full documentation
- `examples/math/gsm8k_grpo_sdp.yaml` - Example config
- `areal/tests/seg_decoupled_ppo/*.py` - 10 test files

## Files Still Needing Updates

### Critical (Blocking)
- `areal/core/remote_inf_engine.py` - Generation loop proximal_t tracking
- `areal/core/workflow_executor.py` - Transformer integration (if approved)

### New Files to Create
- `areal/core/queue_transformer.py` - Base protocol and implementations
- `areal/core/proximal_recomputer.py` - ProximalRecomputer transformer
- `areal/core/staleness_filter.py` - StalenessFilter transformer (or merge with StalenessManager)

### Optional Updates
- `areal/engine/vllm_remote.py` - vLLM backend support
- `areal/experimental/sglang_engine.py` - Experimental engine support

## Next Steps

1. **Immediate**: Review and approve filter/transformer architecture
2. **Day 1**: Implement RemoteInfEngine proximal_t tracking (Phase 6.1)
3. **Day 2**: Create QueueTransformer protocol and ProximalRecomputer (Phase 7)
4. **Day 3**: Fix imports and run CPU tests (Phase 8, 9)
5. **Day 4**: Integration testing and validation

## Notes

- Main branch has 137 files changed since divergence (13,047 insertions, 3,498 deletions)
- Feature branch diverged at commit `a64122d2`
- Feature branch has 2 main commits:
  1. `5b75fe13` - Core segment-wise PPO implementation
  2. `362c21ae` - Strategy pattern refactoring (NOT fully ported yet)
- Original prototype in `boba-tmp` branch has simpler implementation

## References

- **Feature Doc**: `docs/algorithms/segment_wise_ppo.md`
- **AReaL Paper**: https://arxiv.org/pdf/2505.24298
- **Original Issue**: Reduce variance of behavioral importance weights in async RL
- **Key Insight**: Use π_{v+1}/π_v instead of π_current/π_v for importance weighting
