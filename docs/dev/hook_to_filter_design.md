# Hook System → Filter Architecture Migration

## Original Hook-Based Design Analysis

### What the Hook System Does

The feature branch (`dev_seg_decouple_rebase`) uses **hooks** in `WorkflowExecutor`:
- `pre_pause_hooks`: Called RIGHT BEFORE engine.pause()
- `post_pause_hooks`: Called RIGHT AFTER engine.pause()
- `pre_resume_hooks`: Called RIGHT BEFORE engine.resume()
- `post_resume_hooks`: Called RIGHT AFTER engine.resume()

### Two Main Components

#### 1. **ProximalRecomputer** (Transformer Operation)
**Purpose**: Update proximal_t for samples with version == current_version - 1

**What it does**:
- Iterates through `output_queue` and `result_cache`
- For each sample, finds tokens with `version[i] == current_version - 1`
- Calls `engine.recompute_output_logprobs_sync(input_ids, start_index)`
- Patches `proximal_logprobs_t[i]` with new logprobs under current policy
- Marks sample with `_recompute_version = current_version`

**When called**: Via `pre_pause_hook` - RIGHT BEFORE weight update

**Key Insight**: This is a **data transformation** - modifies sample data in-place

#### 2. **StalenessControlStrategy** (Filter Operation)
**Purpose**: Remove samples that are too stale

**Two Implementations**:
- **StandardPPOStrategy**: No filtering (backward compatible)
- **SegmentWisePPOStrategy**: Aggressive filtering

**What SegmentWisePPOStrategy does**:
- `is_sample_too_stale(td, current_ver)`: Check if sample exceeds max_head_offpolicyness
- `purge_stale_samples_from_queue()`: Drain queue, drop stale samples when version increases
- `filter_stale_from_cache()`: Remove stale samples from result_cache before returning
- `should_filter_before_enqueue()`: Pre-filter in rollout thread to prevent queue overflow

**When called**: 
- Purge: Called in `wait()` when version increases
- Filter cache: Called in `wait()` before returning samples
- Before enqueue: Called in rollout thread before putting to output_queue

**Key Insight**: This is a **data filter** - removes samples from collections

## Proposed Filter/Transformer Architecture

### Core Abstraction

```python
from typing import Protocol, List, Any
from collections.abc import Callable

class QueueTransformer(Protocol):
    """Transform or filter items in queue/cache.
    
    Transformers are stateless, composable operations that can:
    - Transform: Modify items in-place
    - Filter: Remove items from the collection
    - Validate: Check items and log warnings
    """
    
    def apply(
        self, 
        items: List[Any], 
        context: TransformerContext
    ) -> List[Any]:
        """Apply transformation/filtering to items.
        
        Args:
            items: List of items to process
            context: Shared context with engine, config, logger
            
        Returns:
            List of items (may be filtered subset, or same list modified in-place)
        """
        ...

@dataclass
class TransformerContext:
    """Shared context for transformers."""
    engine: InferenceEngine
    config: InferenceEngineConfig
    logger: Any
    current_version: int | None = None  # Can be updated dynamically
```

### Concrete Implementations

#### 1. ProximalRecomputer (Transformer)

```python
class ProximalRecomputer(QueueTransformer):
    """Recompute proximal_t for samples with version == current_version - 1."""
    
    def apply(
        self, 
        items: List[TensorDict], 
        context: TransformerContext
    ) -> List[TensorDict]:
        """Recompute proximal_t for v-1 samples.
        
        This modifies items in-place and returns the same list.
        """
        current_ver = context.engine.get_version()
        total_recomputed = 0
        
        for idx, td in enumerate(items):
            n_recomputed = self._recompute_sample(
                td, current_ver, context.engine, context.logger
            )
            total_recomputed += n_recomputed
        
        if total_recomputed > 0:
            context.logger.info(
                f"[Recompute] Recomputed {total_recomputed} tokens "
                f"at version {current_ver}"
            )
        
        return items  # Same list, modified in-place
    
    def _recompute_sample(self, td, current_ver, engine, logger):
        # ... existing recompute logic ...
        pass
```

#### 2. StalenessFilter (Filter)

```python
class StalenessFilter(QueueTransformer):
    """Filter out samples exceeding staleness threshold."""
    
    def __init__(self, max_staleness: int):
        self.max_staleness = max_staleness
    
    def apply(
        self, 
        items: List[TensorDict], 
        context: TransformerContext
    ) -> List[TensorDict]:
        """Remove stale samples from collection."""
        current_ver = context.engine.get_version()
        
        filtered = []
        dropped = 0
        
        for td in items:
            if self._is_too_stale(td, current_ver, context.config):
                dropped += 1
            else:
                filtered.append(td)
        
        if dropped > 0:
            context.logger.warning(
                f"[StalenessFilter] Dropped {dropped} over-stale samples "
                f"at version {current_ver}"
            )
        
        return filtered  # New list, subset of input
    
    def _is_too_stale(self, td, current_ver, config):
        # ... existing staleness check logic ...
        pass
```

#### 3. QueuePurger (Special Filter for Queue)

```python
class QueuePurger:
    """Purge stale samples from queue when version increases.
    
    This is a special case because it operates on Queue (not List).
    Uses drain-process-putback strategy.
    """
    
    def __init__(self, staleness_filter: StalenessFilter):
        self.staleness_filter = staleness_filter
    
    def purge_if_version_changed(
        self,
        output_queue: queue.Queue,
        current_ver: int,
        last_purged_ver: int,
        context: TransformerContext,
    ) -> int:
        """Purge queue if version increased."""
        if current_ver <= last_purged_ver:
            return last_purged_ver  # No version change
        
        # Drain queue
        items = []
        while True:
            try:
                items.append(output_queue.get_nowait())
            except queue.Empty:
                break
        
        # Filter stale samples
        filtered = self.staleness_filter.apply(items, context)
        
        # Put back
        for item in filtered:
            try:
                output_queue.put_nowait(item)
            except queue.Full:
                output_queue.put(item, timeout=1.0)
        
        return current_ver  # Update last_purged_ver
```

### Integration into WorkflowExecutor

```python
class WorkflowExecutor:
    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: InferenceEngine,
        staleness_manager: StalenessManager,
        # NEW: Transformers for different stages
        pre_pause_transformers: List[QueueTransformer] | None = None,
        pre_wait_transformers: List[QueueTransformer] | None = None,
    ):
        self.config = config
        self.inference_engine = inference_engine
        self.staleness_manager = staleness_manager
        
        # Create transformer context (shared by all transformers)
        self.transformer_context = TransformerContext(
            engine=inference_engine,
            config=config,
            logger=logging.getLogger(__name__),
        )
        
        # Transformers applied at different stages
        self.pre_pause_transformers = pre_pause_transformers or []
        self.pre_wait_transformers = pre_wait_transformers or []
        
        # Queue purger (special case)
        if config.enable_segment_wise_ppo:
            staleness_filter = StalenessFilter(config.max_head_offpolicyness)
            self.queue_purger = QueuePurger(staleness_filter)
        else:
            self.queue_purger = None
        
        self.last_purged_ver = -1
        # ... rest of init ...
    
    def pause(self):
        """Pause generation and apply pre-pause transformers."""
        # Apply transformers to result_cache BEFORE pausing engine
        for transformer in self.pre_pause_transformers:
            self.result_cache = transformer.apply(
                self.result_cache, 
                self.transformer_context
            )
        
        # Then pause engine
        self.inference_engine.pause()
        # ... rest of pause logic ...
    
    def wait(self, count: int, timeout: float | None = None) -> Dict[str, Any]:
        """Wait for samples and apply pre-wait transformers."""
        # Purge queue if version increased
        if self.queue_purger:
            current_ver = self.inference_engine.get_version()
            self.last_purged_ver = self.queue_purger.purge_if_version_changed(
                self.async_task_runner.output_queue,
                current_ver,
                self.last_purged_ver,
                self.transformer_context,
            )
        
        # Drain from queue to cache...
        # ... existing wait logic ...
        
        # Apply transformers to result_cache BEFORE returning
        for transformer in self.pre_wait_transformers:
            self.result_cache = transformer.apply(
                self.result_cache,
                self.transformer_context
            )
        
        # Return samples...
        # ... rest of wait logic ...
```

### Factory Function

```python
def create_workflow_executor(
    config: InferenceEngineConfig,
    inference_engine: InferenceEngine,
    staleness_manager: StalenessManager,
    logger: Any,
) -> WorkflowExecutor:
    """Factory to create WorkflowExecutor with appropriate transformers."""
    
    if config.enable_segment_wise_ppo:
        # Segment-wise PPO mode: Enable recompute + staleness filtering
        pre_pause_transformers = [
            ProximalRecomputer(),  # Recompute v-1 samples before weight update
        ]
        
        pre_wait_transformers = [
            StalenessFilter(max_staleness=config.max_head_offpolicyness),
        ]
        
        logger.debug("Configured for segment-wise PPO with recompute + filtering")
    else:
        # Standard PPO mode: No transformers
        pre_pause_transformers = []
        pre_wait_transformers = []
        
        logger.debug("Configured for standard PPO (no transformers)")
    
    return WorkflowExecutor(
        config=config,
        inference_engine=inference_engine,
        staleness_manager=staleness_manager,
        pre_pause_transformers=pre_pause_transformers,
        pre_wait_transformers=pre_wait_transformers,
    )
```

## Benefits of Filter Architecture

1. **Separation of Concerns**
   - Queue/Cache operations isolated from workflow control
   - Each transformer has single responsibility
   - WorkflowExecutor orchestrates but doesn't contain business logic

2. **Composability**
   - Easy to chain multiple transformers
   - Can add new transformers (e.g., data augmentation, validation) without modifying WorkflowExecutor
   - Order of transformers is explicit and configurable

3. **Testability**
   - Each transformer testable in isolation
   - No need to mock entire WorkflowExecutor
   - Clear input/output contracts

4. **No Hooks**
   - Explicit control flow (no "magic" callbacks)
   - Clear where transformers are applied (pause vs wait)
   - Easier to debug and reason about

5. **Flexibility**
   - Different transformer chains for different configs
   - Can disable feature by passing empty transformer list
   - Easy to A/B test different strategies

## Migration Path

### Phase 1: Create Base Infrastructure
1. Define `QueueTransformer` protocol
2. Define `TransformerContext` dataclass
3. Add transformer lists to `WorkflowExecutor.__init__()`

### Phase 2: Port Components
1. Implement `ProximalRecomputer` (from `areal/api/proximal_recomputer.py`)
2. Implement `StalenessFilter` (from `areal/api/staleness_control.py`)
3. Implement `QueuePurger` helper

### Phase 3: Integration
1. Add transformer application to `WorkflowExecutor.pause()`
2. Add transformer application to `WorkflowExecutor.wait()`
3. Create factory function `create_workflow_executor()`

### Phase 4: Testing
1. Unit test each transformer
2. Integration test with WorkflowExecutor
3. Backward compatibility test (`enable_segment_wise_ppo=False`)

## Files to Create/Modify

### New Files
- `areal/core/queue_transformer.py` - Protocol + TransformerContext
- `areal/core/transformers/proximal_recomputer.py` - ProximalRecomputer
- `areal/core/transformers/staleness_filter.py` - StalenessFilter
- `areal/core/transformers/__init__.py` - Export transformers

### Modified Files
- `areal/core/workflow_executor.py` - Add transformer integration
- `areal/core/__init__.py` - Export create_workflow_executor factory
- Tests - Update to use new architecture

### Deprecated Files (from feature branch)
- `areal/api/workflow_components.py` - Hook system (replaced by factory)
- `areal/api/proximal_recomputer.py` - Logic moved to transformer
- `areal/api/staleness_control.py` - Logic moved to transformer

## Comparison

| Aspect | Hook System | Filter/Transformer |
|--------|-------------|-------------------|
| **Coupling** | High (hooks embedded in WorkflowExecutor) | Low (transformers injected) |
| **Testability** | Hard (need full WorkflowExecutor) | Easy (test transformers in isolation) |
| **Composability** | Limited (fixed hook points) | High (chain transformers) |
| **Clarity** | Implicit (callbacks) | Explicit (direct calls) |
| **Flexibility** | Low (modify WorkflowExecutor to change) | High (inject different transformers) |

## Decision

**✅ RECOMMENDED**: Adopt Filter/Transformer architecture

**Rationale**: 
- Cleaner separation of concerns
- Better testability  
- More maintainable
- Aligns with modern design principles
- You identified the key insight: "operations are mainly on Queue and Cache"
