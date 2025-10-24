# Segment-Wise PPO Fresh Integration Progress

## Branch: feature/segment-wise-ppo-v2

Based on: origin/main (f8f2ea39)

## Completed ✅

### Phase 1: Core Files Copied
- ✅ `areal/api/cache_api.py` - RolloutCache abstraction
- ✅ `areal/api/queue_api.py` - RolloutQueue abstraction
- ✅ `areal/api/staleness_control.py` - Strategy pattern for staleness control
- ✅ `areal/api/proximal_recomputer.py` - Proximal logprob recomputation
- ✅ `areal/core/filtered_capacity_modifier.py` - Dynamic capacity adjustment
- ✅ `areal/tests/conftest.py` - uvloop mocking for Windows
- ✅ `areal/tests/sdp/*.py` - All segment-wise PPO unit tests
- ✅ `areal/tests/test_model_utils.py` - model utils tests
- ✅ Committed as f1a5c8cd

## Remaining Work ⏳

### Phase 2: Configuration & I/O Struct (High Priority)

#### 1. `areal/api/cli_args.py`
**Add to `BaseExperimentConfig` (around line 1218)**:
```python
# Segment-wise Decoupled PPO Feature
enable_segment_wise_ppo: bool = field(
    default=True,
    metadata={
        "help": "Enable segment-wise decoupled PPO algorithm (single switch for entire feature). "
        "When enabled, the inference engine tracks proximal_logprobs_t and output_versions per token, "
        "and the training engine uses proximal_t for accurate behavioral importance weight calculation. "
        "This flag is automatically propagated to both InferenceEngineConfig and PPOActorConfig. "
        "Set to false to disable and use standard PPO behavior."
    },
)
```

**Add to `InferenceEngineConfig` (around line 862)**:
```python
enable_segment_wise_ppo: bool = field(
    default=True,
    metadata={
        "help": "[Auto-populated from BaseExperimentConfig.enable_segment_wise_ppo] "
        "Enable proximal logprob tracking during generation (inference-side). "
        "Tracks proximal_logprobs_t and output_versions per token for segment-wise PPO. "
        "Do not set this directly; set BaseExperimentConfig.enable_segment_wise_ppo instead."
    },
)
```

**Add to `PPOActorConfig` (around line 482)**:
```python
enable_segment_wise_ppo: bool = field(
    default=True,
    metadata={
        "help": "[Auto-populated from BaseExperimentConfig.enable_segment_wise_ppo] "
        "Enable segment-wise decoupled PPO loss computation (training-side). "
        "Uses per-token proximal_logprobs_t for accurate behavioral importance weight calculation. "
        "Do not set this directly; set BaseExperimentConfig.enable_segment_wise_ppo instead."
    },
)
```

**Add auto-propagation in `load_expr_config()` (around line 1322)**:
```python
# Auto-propagate enable_segment_wise_ppo from top-level to child configs
# This ensures a single switch controls the entire feature
if hasattr(cfg, 'enable_segment_wise_ppo'):
    if hasattr(cfg, 'rollout') and hasattr(cfg.rollout, 'enable_segment_wise_ppo'):
        cfg.rollout.enable_segment_wise_ppo = cfg.enable_segment_wise_ppo
    if hasattr(cfg, 'actor') and hasattr(cfg.actor, 'enable_segment_wise_ppo'):
        cfg.actor.enable_segment_wise_ppo = cfg.enable_segment_wise_ppo
```

#### 2. `areal/api/io_struct.py`
**Add to `FinetuneSpec` dataclass**:
```python
# Segment-wise PPO fields (optional, only present if enabled)
proximal_logprobs_t: torch.Tensor | None = None  # Shape: [batch, seq_len, 2] or None
    # [:, :, 0] = token_id (0 if not recomputed)
    # [:, :, 1] = proximal logprob
output_versions: torch.Tensor | None = None      # Shape: [batch, seq_len] or None
_recompute_version: int | None = None            # Metadata for version tracking
```

### Phase 3: Core Component Modifications

#### 3. `areal/core/capacity_modifier.py`
**Check if base class exists, if not create it**:
```python
from abc import ABC, abstractmethod

class CapacityModifier(ABC):
    """Base class for modifying staleness manager capacity dynamically."""

    @abstractmethod
    def get_capacity_delta(self, version: int) -> int:
        """Return capacity adjustment for given version."""
        pass
```

#### 4. `areal/core/staleness_manager.py`
**Add support for capacity modifiers**:
```python
# In __init__:
self.capacity_modifiers: List[CapacityModifier] = []

# Add method:
def register_capacity_modifier(self, modifier: CapacityModifier):
    self.capacity_modifiers.append(modifier)

# Modify get_capacity():
def get_capacity(self, version: int) -> int:
    base_capacity = # ... existing logic ...
    delta = sum(mod.get_capacity_delta(version) for mod in self.capacity_modifiers)
    return max(0, base_capacity + delta)
```

### Phase 4: Factory Pattern (Critical - New Structure)

#### 5. Create `areal/api/workflow_factory.py`

**Key Challenge**: Must work with main's new `RemoteInfEngine` architecture

**Strategy**: Factory creates components, RemoteInfEngine uses them

```python
from areal.core import WorkflowExecutor  # ← New location!
from areal.core.staleness_manager import StalenessManager
from areal.api.staleness_control import (
    StandardPPOStrategy,
    SegmentWisePPOStrategy,
)
# ... etc

def create_workflow_executor(
    inference_engine,
    config,
    logger,
    staleness_manager=None,
    train_data_parallel_size=None,
) -> WorkflowExecutor:
    """
    Factory for creating WorkflowExecutor with feature-based component injection.

    Compatible with RemoteInfEngine structure in main.
    """
    # Create queue/cache based on config
    output_queue = create_queue(config)
    result_cache = create_cache(config)

    # Create strategy based on enable_segment_wise_ppo
    staleness_strategy = create_staleness_strategy(config)

    # Create proximal recomputer if needed
    proximal_recomputer = create_proximal_recomputer(inference_engine, logger, config)

    # Create capacity modifier if needed
    filtered_capacity_modifier = create_filtered_capacity_modifier(config)

    # Register modifier with staleness_manager if provided
    if staleness_manager and filtered_capacity_modifier:
        register_capacity_modifiers(staleness_manager, filtered_capacity_modifier)

    # Create executor with injected dependencies
    executor = WorkflowExecutor(
        inference_engine=inference_engine,
        staleness_manager=staleness_manager,
        output_queue=output_queue,
        result_cache=result_cache,
        staleness_strategy=staleness_strategy,
        proximal_recomputer=proximal_recomputer,
        filtered_capacity_modifier=filtered_capacity_modifier,
        config=config,
        logger=logger,
    )

    # Initialize (creates staleness_manager if None, with DP scaling)
    executor.initialize(logger=logger, train_data_parallel_size=train_data_parallel_size)

    # Register modifiers after staleness_manager created
    if staleness_manager is None and executor.staleness_manager and filtered_capacity_modifier:
        register_capacity_modifiers(executor.staleness_manager, filtered_capacity_modifier)

    return executor
```

### Phase 5: WorkflowExecutor Integration (CRITICAL)

#### 6. Modify `areal/core/workflow_executor.py`

**Location**: This is where WorkflowExecutor now lives!

**Changes Needed**:

1. **Import additions** (top of file):
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areal.api.cache_api import RolloutCache
    from areal.api.queue_api import RolloutQueue
    from areal.api.staleness_control import StalenessControlStrategy
    from areal.api.proximal_recomputer import ProximalRecomputer
    from areal.core.filtered_capacity_modifier import FilteredSamplesCapacityModifier
```

2. **Constructor modifications** (`__init__`):
```python
def __init__(
    self,
    inference_engine: "InferenceEngine",
    config: InferenceEngineConfig,
    staleness_manager: StalenessManager | None = None,
    output_queue: "RolloutQueue | None" = None,          # ← ADD
    result_cache: "RolloutCache | None" = None,          # ← ADD
    staleness_strategy: "StalenessControlStrategy | None" = None,  # ← ADD
    proximal_recomputer: "ProximalRecomputer | None" = None,      # ← ADD
    filtered_capacity_modifier: "FilteredSamplesCapacityModifier | None" = None,  # ← ADD
    logger: Any = None,
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
```

3. **initialize() modifications**:
```python
def initialize(self, logger=None, train_data_parallel_size: int | None = None):
    # Logger initialization
    if logger is not None:
        self.logger = logger
    elif self.logger is None:
        self.logger = logging.getLogger("WorkflowExecutor")

    # Only create staleness_manager if not provided (allows factory injection)
    if self.staleness_manager is None:
        # Detect DP size
        if train_data_parallel_size is not None:
            dp_world_size = train_data_parallel_size
        else:
            # ... existing detection logic ...

        # Apply DP scaling
        max_concurrent_rollouts = max(1, self.max_concurrent_rollouts // dp_world_size)
        consumer_batch_size = max(1, self.consumer_batch_size // dp_world_size)

        self.staleness_manager = StalenessManager(
            max_concurrent_rollouts=max_concurrent_rollouts,
            consumer_batch_size=consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

    # Start worker thread
    # ... existing code ...
```

4. **_rollout_thread_async() modifications** (around line where rollout completes):
```python
# After rollout completes, before enqueuing:
current_ver = self.inference_engine.get_version()

# Pre-enqueue filtering (staleness strategy)
should_enqueue = True
if self.staleness_strategy:
    should_enqueue = self.staleness_strategy.should_enqueue_sample(
        traj, current_ver, self.config
    )

if should_enqueue:
    # Add proximal logprobs if recomputer exists
    if self.proximal_recomputer:
        traj = self.proximal_recomputer.add_proximal_logprobs(traj, current_ver)

    # Enqueue
    try:
        self.output_queue.put_nowait(_TimedResult(create_time, traj))
    except queue.Full:
        raise RuntimeError("Output queue full. Please increase queue_size.")

    self.staleness_manager.on_rollout_accepted()
else:
    # Filtered pre-enqueue
    if self.filtered_capacity_modifier:
        self.filtered_capacity_modifier.on_samples_filtered(1, current_ver)
    self.staleness_manager.on_rollout_rejected()
```

5. **wait() modifications** (beginning of function):
```python
def wait(self, count: int, timeout: float | None = None):
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

    # Step 2: Drain queue to cache
    while True:
        try:
            timed_result = self.output_queue.get_nowait()
            self.result_cache.add(timed_result)
        except queue.Empty:
            break

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

    # ... rest of existing wait() logic ...
```

### Phase 6: RemoteInfEngine Integration

#### 7. Modify `areal/core/remote_inf_engine.py`

**Find where WorkflowExecutor is created** (likely in `__init__` or `initialize`):

**Replace**:
```python
self.workflow_executor = WorkflowExecutor(
    inference_engine=self,
    config=config,
)
```

**With**:
```python
from areal.api.workflow_factory import create_workflow_executor

self.workflow_executor = create_workflow_executor(
    inference_engine=self,
    config=config,
    logger=self.logger,
    train_data_parallel_size=train_data_parallel_size,  # Pass if available
)
```

### Phase 7: Training Integration

#### 8. Copy training-side changes

From old branch:
- `realhf/impl/model/utils/ppo_functional.py` (importance weight calculation)
- `realhf/impl/model/interface/ppo_interface.py` (collect proximal_t)
- `areal/workflow/rlvr.py` (any workflow changes)
- `areal/engine/ppo/actor.py` (any actor changes)

### Phase 8: Example & Documentation

#### 9. Copy example config
- `examples/math/gsm8k_grpo_segment_wise.yaml`

### Phase 9: Testing

#### 10. Update test imports
All tests in `areal/tests/sdp/` need:
```python
from areal.core import WorkflowExecutor  # Not from areal.api.workflow_api
```

#### 11. Run tests
```bash
pytest areal/tests/sdp/ -v
pytest areal/tests/test_model_utils.py -v
```

## Current Status

✅ **Completed**: Core feature files copied (Phase 1)
⏳ **In Progress**: Configuration changes (Phase 2)
⏳ **Next**: WorkflowExecutor integration (Phase 5) - CRITICAL

## Estimated Remaining Time

- Phase 2-4: 30 minutes (straightforward copying)
- Phase 5: 1 hour (careful WorkflowExecutor modifications)
- Phase 6: 30 minutes (RemoteInfEngine integration)
- Phase 7-9: 1 hour (testing and fixes)

**Total**: ~3 hours remaining

## Key Success Criteria

1. ✅ All core feature logic copied
2. ⏳ WorkflowExecutor modifications in `areal/core/workflow_executor.py` (NEW LOCATION)
3. ⏳ Factory works with RemoteInfEngine
4. ⏳ All tests pass on Windows with mocked uvloop
5. ⏳ Feature can be toggled via `enable_segment_wise_ppo=false`
6. ⏳ No code in `areal/api/workflow_api.py` except `RolloutWorkflow`

---

**Branch**: `feature/segment-wise-ppo-v2`
**Base**: origin/main (f8f2ea39)
**Last Commit**: f1a5c8cd (WIP: copy core files)
