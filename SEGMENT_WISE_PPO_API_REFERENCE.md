# Segment-Wise Decoupled PPO - API Reference

## Overview
This document describes the actual API for the segment-wise decoupled PPO feature as implemented.

## Core Components

### 1. FilteredSamplesCapacityModifier

**Purpose**: Tracks samples filtered by staleness control and adjusts capacity to prevent deadlock.

**Location**: `areal/core/filtered_capacity_modifier.py`

**API**:

```python
class FilteredSamplesCapacityModifier(CapacityModifier):
    def __init__(self):
        """Initialize with zero filtered count."""

    def on_samples_filtered(self, count: int, version: int | None = None) -> None:
        """Record that samples were filtered.

        Args:
            count: Number of samples filtered
            version: Model version when filtering occurred (optional)
        """

    def on_capacity_consumed(self, count: int) -> None:
        """Record that capacity was consumed (samples taken for training).

        Args:
            count: Number of samples consumed from cache
        """

    def modify_capacity(
        self,
        base_capacity: int,
        current_version: int,
        stats: RolloutStat
    ) -> int:
        """Add filtered count back to capacity (CapacityModifier interface).

        Args:
            base_capacity: Base capacity calculated by StalenessManager
            current_version: Current model version
            stats: Current rollout statistics

        Returns:
            Adjusted capacity (base_capacity + filtered_count)
        """

    def get_filtered_count(self) -> int:
        """Get current filtered sample count."""

    def get_version_stats(self) -> dict[int, int]:
        """Get filtered sample counts by version."""

    def reset(self) -> None:
        """Reset filtered count (for testing or manual intervention)."""
```

**Thread Safety**: All methods are thread-safe with internal locking.

**Usage**:
```python
modifier = FilteredSamplesCapacityModifier()
staleness_manager.register_capacity_modifier(modifier)

# When samples are filtered:
modifier.on_samples_filtered(5)  # 5 samples filtered

# Capacity will automatically be increased by 5
```

---

### 2. ProximalRecomputer

**Purpose**: Recomputes proximal_logprobs_t for v-1 samples before weight updates.

**Location**: `areal/api/proximal_recomputer.py`

**API**:

#### Main Production API

```python
class ProximalRecomputer:
    def __init__(self, inference_engine: InferenceEngine, logger: Any):
        """Initialize the recomputer."""

    def recompute_all(
        self,
        output_queue: RolloutQueue,
        result_cache: RolloutCache
    ) -> None:
        """Recompute proximal_t for all v-1 samples before weight update.

        This should be called RIGHT BEFORE update_weights() to ensure:
        1. All in-progress rollouts are still at current version
        2. All v-1 samples (in queue or cache) get recomputed
        3. No samples miss their recompute window

        Processes BOTH output_queue and result_cache.
        """
```

#### Test-Friendly API

```python
    def recompute_for_sample(self, trajectory: dict, prompt_len: int) -> None:
        """Recompute proximal_logprobs_t for a single sample (test-friendly API).

        This is a simplified interface for unit testing that works with plain dicts
        instead of TensorDicts. Only recomputes samples with version exactly (current_version - 1).

        Args:
            trajectory: Dict with keys: input_ids, output_version, proximal_logprobs_t
            prompt_len: Length of the prompt (number of input tokens)
        """

    def recompute_batch(self, batch: List[dict], prompt_len: int) -> None:
        """Recompute proximal_logprobs_t for a batch of samples.

        Args:
            batch: List of trajectory dicts
            prompt_len: Length of the prompt
        """
```

**Usage**:
```python
# Production usage
recomputer = ProximalRecomputer(inference_engine, logger)

# Called right before weight update
recomputer.recompute_all(output_queue, result_cache)
trainer.update_weights()

# Testing usage
trajectory = {
    "input_ids": [1, 2, 3, 4, 5],
    "output_version": [4, 4],  # v-1
    "proximal_logprobs_t": [0.9, 0.8]
}
recomputer.recompute_for_sample(trajectory, prompt_len=3)
```

---

### 3. StalenessControlStrategy

**Purpose**: Encapsulates staleness filtering logic for different PPO modes.

**Location**: `areal/api/staleness_control.py`

**Base Strategy**:

```python
class StalenessControlStrategy(ABC):
    def __init__(self, config: InferenceEngineConfig | None = None):
        """Initialize strategy with configuration."""

    @abstractmethod
    def is_sample_too_stale(
        self,
        td: TensorDict,
        current_ver: int,
        config: InferenceEngineConfig
    ) -> bool:
        """Check if a sample exceeds staleness threshold."""

    def should_filter_before_enqueue(self) -> bool:
        """Whether to filter stale samples before enqueueing in rollout thread."""

    def purge_stale_samples_from_queue(...) -> int:
        """Drain output queue and drop stale samples when version increases."""

    def filter_stale_from_cache(...) -> int:
        """Remove stale samples from result_cache."""
```

**StandardPPOStrategy**:

```python
class StandardPPOStrategy(StalenessControlStrategy):
    """Strategy for standard PPO - no staleness filtering.

    Provides backward-compatible behavior matching the original AReaL
    implementation before segment-wise PPO was added.
    """

    def should_filter_before_enqueue(self) -> bool:
        return False  # No filtering for backward compatibility

    def is_sample_too_stale(...) -> bool:
        return False  # Never filters samples
```

**SegmentWisePPOStrategy**:

```python
class SegmentWisePPOStrategy(StalenessControlStrategy):
    """Strategy for segment-wise decoupled PPO - full staleness control.

    Implements aggressive staleness filtering to ensure training stability:
    1. Purge stale samples from queue when version increases
    2. Filter stale samples from cache before returning
    3. Pre-filter samples before enqueue to prevent queue overflow
    """

    def should_filter_before_enqueue(self) -> bool:
        return True  # Aggressive filtering

    def is_sample_too_stale(...) -> bool:
        # Returns True if staleness > allow_staleness
        # v-1 samples: acceptable
        # v-2+ samples: too stale, filtered
```

**Key Behavior Difference**:
- **StandardPPOStrategy**: No staleness control (backward compatibility)
- **SegmentWisePPOStrategy**: Aggressive staleness control (v-1 only)

---

### 4. Factory Functions

**Location**: `areal/api/workflow_factory.py`

**API**:

```python
def create_staleness_strategy(
    config: InferenceEngineConfig
) -> StalenessControlStrategy:
    """Create staleness control strategy based on configuration.

    Returns:
        SegmentWisePPOStrategy if config.enable_segment_wise_ppo else StandardPPOStrategy
    """

def create_proximal_recomputer(
    inference_engine: InferenceEngine,
    logger: Any,
    config: InferenceEngineConfig
) -> ProximalRecomputer | None:
    """Create proximal recomputer if needed.

    Returns:
        ProximalRecomputer if config.enable_segment_wise_ppo else None
    """

def create_filtered_capacity_modifier(
    config: InferenceEngineConfig
) -> FilteredSamplesCapacityModifier | None:
    """Create filtered capacity modifier if needed.

    Returns:
        FilteredSamplesCapacityModifier if config.enable_segment_wise_ppo else None
    """

def create_workflow_executor(
    inference_engine: InferenceEngine,
    staleness_manager: StalenessManager,
    config: InferenceEngineConfig,
    logger: Any
) -> WorkflowExecutor:
    """Create and configure WorkflowExecutor with all dependencies.

    This is the main factory method that assembles all components based on
    configuration and injects them into WorkflowExecutor.
    """

def register_capacity_modifiers(
    staleness_manager: StalenessManager,
    filtered_capacity_modifier: FilteredSamplesCapacityModifier | None
) -> None:
    """Register capacity modifiers with StalenessManager."""
```

---

### 5. Configuration

**Location**: `areal/api/cli_args.py`

**BaseExperimentConfig**:

```python
@dataclass
class BaseExperimentConfig:
    enable_segment_wise_ppo: bool = field(
        default=True,
        metadata={
            "help": "Enable segment-wise decoupled PPO loss computation. "
            "Single switch that controls the entire feature. "
            "Automatically propagates to InferenceEngineConfig and PPOActorConfig."
        }
    )
```

**Auto-Propagation** (in `load_expr_config()`):

```python
if hasattr(cfg, 'enable_segment_wise_ppo'):
    if hasattr(cfg, 'rollout') and hasattr(cfg.rollout, 'enable_segment_wise_ppo'):
        cfg.rollout.enable_segment_wise_ppo = cfg.enable_segment_wise_ppo
    if hasattr(cfg, 'actor') and hasattr(cfg.actor, 'enable_segment_wise_ppo'):
        cfg.actor.enable_segment_wise_ppo = cfg.enable_segment_wise_ppo
```

---

## Usage Patterns

### Enabling the Feature

**YAML Configuration** (recommended):

```yaml
# config.yaml
experiment_name: my_experiment
trial_name: my_trial
enable_segment_wise_ppo: true  # Single switch

rollout:
  # enable_segment_wise_ppo will be auto-propagated
  experiment_name: my_experiment
  trial_name: my_trial

actor:
  # enable_segment_wise_ppo will be auto-propagated
  experiment_name: my_experiment
  trial_name: my_trial
```

**Command-Line Override**:

```bash
python train.py --config config.yaml enable_segment_wise_ppo=false
```

### Disabling the Feature

Set `enable_segment_wise_ppo: false` in top-level config. All child configs will inherit this value.

---

## Abstract Interfaces

### CapacityModifier

```python
class CapacityModifier(ABC):
    @abstractmethod
    def modify_capacity(
        self,
        base_capacity: int,
        current_version: int,
        stats: RolloutStat
    ) -> int:
        """Modify the base capacity calculation."""
```

### RolloutCache

```python
class RolloutCache(ABC):
    @abstractmethod
    def add(self, item: TensorDict) -> None:
        """Add item to cache."""

    @abstractmethod
    def get_all(self) -> list[TensorDict]:
        """Get all items (returns copy)."""

    @abstractmethod
    def take_first_n(self, n: int) -> list[TensorDict]:
        """Remove and return first n items."""

    @abstractmethod
    def size(self) -> int:
        """Get current cache size."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all items."""

    @abstractmethod
    def filter_inplace(self, predicate: Callable[[TensorDict], bool]) -> int:
        """Filter items in place, return count of items removed."""
```

### RolloutQueue

```python
class RolloutQueue(ABC):
    @abstractmethod
    def put(self, item: TensorDict, timeout: float | None = None) -> None:
        """Put item in queue."""

    @abstractmethod
    def get(self, timeout: float | None = None) -> TensorDict:
        """Get item from queue."""

    @abstractmethod
    def put_nowait(self, item: TensorDict) -> None:
        """Put item without blocking."""

    @abstractmethod
    def get_nowait(self) -> TensorDict:
        """Get item without blocking."""

    @abstractmethod
    def empty(self) -> bool:
        """Check if queue is empty."""

    @abstractmethod
    def qsize(self) -> int:
        """Get approximate queue size."""
```

---

## Extension Points

### Adding Custom Capacity Modifier

```python
from areal.core.capacity_modifier import CapacityModifier

class MyModifier(CapacityModifier):
    def modify_capacity(self, base_capacity, current_version, stats):
        # Custom logic
        return base_capacity + 10

# Register
staleness_manager.register_capacity_modifier(MyModifier())
```

### Adding Custom Staleness Strategy

```python
from areal.api.staleness_control import StalenessControlStrategy

class MyStrategy(StalenessControlStrategy):
    def is_sample_too_stale(self, td, current_ver, config):
        # Custom staleness logic
        return False

# Use via factory
def create_staleness_strategy(config):
    if config.my_custom_mode:
        return MyStrategy(config)
    # ... existing logic
```

---

## Thread Safety

- **FilteredSamplesCapacityModifier**: Thread-safe (internal Lock)
- **ProximalRecomputer**: Thread-safe for concurrent recomputation
- **RolloutCache/RolloutQueue**: Implementations must be thread-safe
- **StalenessControlStrategy**: Stateless, inherently thread-safe

---

## Testing

### Unit Testing with Simple API

```python
from areal.api.proximal_recomputer import ProximalRecomputer
from unittest.mock import Mock

# Create with mocked engine
mock_engine = Mock()
mock_engine.get_version.return_value = 5
mock_engine.recompute_output_logprobs_sync.return_value = [0.1, 0.2]

recomputer = ProximalRecomputer(mock_engine, Mock())

# Test with simple dict (no TensorDict needed)
trajectory = {
    "input_ids": [1, 2, 3, 4, 5],
    "output_version": [4, 4],  # v-1
    "proximal_logprobs_t": [0.9, 0.8]
}

recomputer.recompute_for_sample(trajectory, prompt_len=3)
assert trajectory["proximal_logprobs_t"] == [0.1, 0.2]
```

---

## Common Pitfalls

### 1. Don't Mix Configuration Levels

❌ **Wrong**:
```yaml
enable_segment_wise_ppo: true  # Top level
rollout:
  enable_segment_wise_ppo: false  # Will be overridden!
```

✅ **Correct**:
```yaml
enable_segment_wise_ppo: true  # Top level only
rollout:
  # Automatically inherits true
```

### 2. Don't Modify Private Attributes

❌ **Wrong**:
```python
modifier._filtered_count += 5  # Don't access private attribute
```

✅ **Correct**:
```python
modifier.on_samples_filtered(5)  # Use public API
```

### 3. Don't Skip Capacity Modifier Registration

❌ **Wrong**:
```python
modifier = create_filtered_capacity_modifier(config)
# Forgot to register!
```

✅ **Correct**:
```python
modifier = create_filtered_capacity_modifier(config)
register_capacity_modifiers(staleness_manager, modifier)
```

---

## Migration from Test Expectations

If you have tests written against original design docs, here are the API changes:

### FilteredSamplesCapacityModifier

| Original (Tests) | Actual Implementation |
|------------------|----------------------|
| `on_sample_filtered()` | `on_samples_filtered(count, version=None)` |
| `modifier.filtered_count` | `modifier.get_filtered_count()` |
| `get_adjustment()` | `get_filtered_count()` |
| N/A | `modify_capacity()` (CapacityModifier interface) |

### ProximalRecomputer

| Original (Tests) | Actual Implementation |
|------------------|----------------------|
| Only dict-based API | Added: `recompute_for_sample()`, `recompute_batch()` |
| N/A | Primary: `recompute_all()` for production use |

### StalenessControlStrategy

| Original (Tests) | Actual Implementation |
|------------------|----------------------|
| StandardPPO filters (True) | StandardPPO doesn't filter (False) |
| SegmentWise doesn't filter (False) | SegmentWise filters (True) |

---

## Summary

The segment-wise decoupled PPO feature provides:

✅ **Single-switch control** via `enable_segment_wise_ppo`
✅ **Factory pattern** for component assembly
✅ **Strategy pattern** for pluggable staleness control
✅ **Dependency injection** for testability
✅ **Abstract interfaces** for extensibility
✅ **Thread-safe** implementations
✅ **Backward compatible** (StandardPPOStrategy preserves old behavior)

For more details, see:
- Implementation: `areal/api/workflow_factory.py`
- Tests: `areal/tests/sdp/`
- Commit: c3cc7e64
