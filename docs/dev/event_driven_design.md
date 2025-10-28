# Event-Driven Architecture for Queue/Cache Extensions

## Design Overview

This document describes the event-driven architecture for extending queue/cache behavior without modifying core workflow logic.

## Core Concepts

### 1. QueueFilter (Admission Control)

Filters are checked when items are added to queue/cache. They can reject items.

```python
class QueueFilter(Protocol):
    def should_accept(self, item, context: EventContext) -> bool:
        """Return False to reject item from queue/cache."""
        ...
```

**Use Cases**:
- Reject over-stale samples
- Enforce data quality
- Trigger recompute at admission time

**Example: StalenessFilter**
```python
class StalenessFilter(QueueFilter):
    def should_accept(self, item, context):
        staleness = calculate_staleness(item, context.engine.get_version())
        return staleness <= self.max_staleness
```

### 2. EventHandler (React to Events)

Handlers respond to events like policy updates, pause/resume.

```python
class EventHandler(Protocol):
    def on_event(self, context: EventContext) -> None:
        """Handle event synchronously."""
        ...
```

**Use Cases**:
- Recompute proximal_t before policy update
- Scan and filter queue/cache
- Log metrics

**Example: ProximalRecomputer**
```python
class ProximalRecomputer(EventHandler):
    def on_event(self, context):
        if context.event_type == EventType.BEFORE_POLICY_UPDATE:
            # Scan queue and cache
            for item in context.data['queue'] + context.data['cache']:
                if needs_recompute(item):
                    recompute_proximal_t(item, context.engine)
```

### 3. EventRegistry (Manage Handlers)

Registry manages handlers and fires events synchronously.

```python
registry = EventRegistry()
registry.register_handler(EventType.BEFORE_POLICY_UPDATE, handler)
registry.fire_event(context)  # All handlers execute, then returns
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  Training Loop                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │  1. Rollout (generate samples)                   │   │
│  │     ↓                                            │   │
│  │  2. Add to Queue/Cache                          │   │
│  │     ├─→ Apply QueueFilter (admission control)   │   │
│  │     │   - StalenessFilter checks staleness      │   │
│  │     │   - Reject if too stale                   │   │
│  │     └─→ Accept: add to queue/cache              │   │
│  │                                                   │   │
│  │  3. Before Policy Update                        │   │
│  │     ├─→ Fire BEFORE_POLICY_UPDATE event         │   │
│  │     │   - ProximalRecomputer scans queue/cache  │   │
│  │     │   - Recompute proximal_t for v-1 samples  │   │
│  │     └─→ Event returns (all handlers done)       │   │
│  │                                                   │   │
│  │  4. Update Policy Weights                       │   │
│  │     ↓                                            │   │
│  │  5. Fire AFTER_POLICY_UPDATE event              │   │
│  │     └─→ (Optional handlers can run)             │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Integration Points

### Point 1: Filter at Queue Admission

**Option A**: User integrates filter in their code
```python
# In user's rollout submission code
executor, registry = create_workflow_executor_with_events(config, engine)

if hasattr(executor, '_staleness_filter') and executor._staleness_filter:
    # Check filter before submitting
    context = EventContext(EventType.BEFORE_PAUSE, engine, config, logger)
    if executor._staleness_filter.should_accept(sample, context):
        executor.submit(sample, workflow)
    else:
        logger.debug("Sample rejected by staleness filter")
else:
    # No filter, submit directly
    executor.submit(sample, workflow)
```

**Option B**: Executor checks filter internally (cleaner, recommended)
```python
# In WorkflowExecutor.submit() or _commit_one_to_runner()
def _apply_filter(self, item):
    if self._staleness_filter and self._event_context:
        return self._staleness_filter.should_accept(item, self._event_context)
    return True

def submit(self, data, ...):
    # Apply filter before submission
    if not self._apply_filter(data):
        self.logger.debug("Sample rejected by filter")
        return
    # Continue with normal submission
    ...
```

### Point 2: Event Firing Before Policy Update

**In RemoteInfEngine.update_weights_from_disk() or training loop**:
```python
def update_weights_from_disk(self, meta):
    # Fire BEFORE_POLICY_UPDATE event
    if hasattr(self.workflow_executor, '_event_registry'):
        context = EventContext(
            EventType.BEFORE_POLICY_UPDATE,
            engine=self,
            config=self.config,
            logger=self.logger,
            data={
                'queue': self.workflow_executor.runner.output_queue,
                'cache': self.workflow_executor._pending_results,
                'old_version': self.get_version(),
            }
        )
        self.workflow_executor._event_registry.fire_event(context)
        # When this returns, all handlers (e.g., ProximalRecomputer) have completed

    # Now update weights
    fut = self.executor.submit(_update_weights_from_disk, ...)
    return fut
```

## Benefits

### 1. Minimal Workflow Changes
- Core workflow logic unchanged
- Extensions added via filters and handlers
- Clean separation of concerns

### 2. Clearer Responsibility
- **Filters**: Admission control (what gets in)
- **Handlers**: React to events (what happens when)
- **Registry**: Coordinate execution (when and how)

### 3. More Decoupled
- Queue/Cache don't know about PPO logic
- PPO logic in separate filter/handler modules
- Easy to add new filters/handlers without modifying workflow

### 4. Synchronous & Predictable
- `fire_event()` returns when all handlers done
- No async complexity
- Easy to reason about order of operations

## Comparison with Transformer Pattern

| Aspect | Transformer Pattern | Event-Driven Pattern |
|--------|-------------------|---------------------|
| **Integration** | Executor calls transformers explicitly | Filters at admission, Handlers on events |
| **Workflow Changes** | Moderate (add transformer calls) | Minimal (add event firing) |
| **Responsibility** | Mixed (executor orchestrates) | Clear (filters/handlers independent) |
| **Extensibility** | Add to transformer list | Register filter/handler |
| **Coupling** | Executor aware of transformers | Executor only aware of extension points |

## For Segment-Wise PPO

**Configuration**:
```python
config = InferenceEngineConfig(
    enable_segment_wise_ppo=True,
    max_head_offpolicyness=2,
)

engine = RemoteSGLangEngine(config)
executor, registry = create_workflow_executor_with_events(config, engine)
```

**What Happens**:
1. **At Admission**: StalenessFilter rejects over-stale samples
2. **Before Policy Update**: Fire BEFORE_POLICY_UPDATE event
   - ProximalRecomputer scans queue and cache
   - Recomputes proximal_t for all v-1 samples
   - Returns when done
3. **Update Policy**: Weights updated
4. **Training**: Loss uses proximal_t for behavioral importance weighting

**Result**: Clean, decoupled, event-driven segment-wise PPO!

## Next Steps

1. ✅ Implement EventSystem (EventType, EventContext, QueueFilter, EventHandler, EventRegistry)
2. ✅ Refactor StalenessFilter as QueueFilter
3. ✅ Refactor ProximalRecomputer as EventHandler
4. ✅ Create event_factory for configuration
5. ⏳ Add filter application in WorkflowExecutor
6. ⏳ Add event firing before policy updates
7. ⏳ Test and validate

## Files

- `areal/core/event_system.py` - Core protocols and registry
- `areal/core/filters/staleness_filter.py` - QueueFilter implementation
- `areal/core/handlers/proximal_recomputer.py` - EventHandler implementation
- `areal/core/event_factory.py` - Factory function
- `docs/dev/event_driven_design.md` - This document
