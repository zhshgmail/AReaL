# Event-Driven Architecture v2 (Corrected)

## Critical Business Logic

### Recompute Timing
**MUST happen BEFORE policy update, not after!**

**Why?**
- We want π_proximal_t = π_{v+1} for samples with tokens at version v-1
- To get π_{v+1}, we recompute using CURRENT policy (v) before it updates to v+1
- If we recompute AFTER update (using v+1 policy), we'd get wrong values!

**Timeline**:
```
Sample has token at v-1
Current policy is v
Need: π_proximal_t = π_v (which will become v+1 after update)

→ PRE_UPDATE event fires
  → Recompute using current policy v
  → π_proximal_t = π_v ✓
→ Update weights: v → v+1
→ POST_UPDATE event fires
  → Now π_proximal_t = π_v where current is v+1 ✓
```

## Two Extension Mechanisms

### 1. Filters (Admission Control)
**When**: During `add()` operations (queue.put, cache.append)
**Purpose**: Accept/reject items at admission
**Examples**:
- StalenessFilter: Reject over-stale samples when adding to queue

```python
class Queue:
    def __init__(self):
        self.add_filters = []  # List of filters

    def register_add_filter(self, filter: QueueFilter):
        self.add_filters.append(filter)

    def put(self, item):
        # Check all add filters
        for filter in self.add_filters:
            if not filter.should_accept(item, context):
                return  # Rejected
        # Accept and add
        self._internal_queue.put(item)
```

### 2. Event Handlers (React to Events)
**When**: When events fire (PRE_UPDATE, POST_UPDATE, etc.)
**Purpose**: Perform actions in response to system events
**Examples**:
- ProximalRecomputer: On PRE_UPDATE, scan queue/cache and recompute

**Option A**: Queue/Cache implement EventHandler
```python
class Queue(EventHandler):
    def __init__(self):
        self.event_filters = {}  # event_type → List[filters]

    def register_event_filter(self, event_type, filter):
        """Register filter to run on specific event"""
        if event_type not in self.event_filters:
            self.event_filters[event_type] = []
        self.event_filters[event_type].append(filter)

    def on_event(self, event_type, context):
        """Called by EventRegistry when event fires"""
        # Apply event-specific filters
        if event_type in self.event_filters:
            for filter in self.event_filters[event_type]:
                # Scan queue and apply filter
                # Could remove stale items, recompute, etc.
                pass
```

**Option B**: Separate handler that accesses queue/cache
```python
class ProximalRecomputer(EventHandler):
    def on_event(self, context):
        if context.event_type == PRE_UPDATE:
            # Access queue and cache from context
            queue = context.data['queue']
            cache = context.data['cache']

            # Scan and recompute
            for item in self._drain_queue(queue):
                self._recompute(item)
                queue.put(item)
```

## Recommended Design

### Queue/Cache with Filter Registration
```python
class AsyncTaskRunner:
    def __init__(self, ...):
        self.output_queue = queue.Queue()
        self.result_cache = []

        # Filter registration
        self.add_filters = []  # Checked on every put()

    def register_add_filter(self, filter: QueueFilter):
        """Register filter for admission control"""
        self.add_filters.append(filter)

    def _put_to_output_queue(self, item):
        """Internal: add to output queue with filter check"""
        # Check all filters
        for filter in self.add_filters:
            if hasattr(filter, 'should_accept'):
                if not filter.should_accept(item, self._filter_context):
                    self.logger.debug("Item rejected by add filter")
                    return  # Rejected
        # Accept
        self.output_queue.put(item)

    def on_event(self, event_type: str, context: EventContext):
        """Handle events by scanning queue/cache"""
        # This can be registered with EventRegistry
        # Allows queue/cache to react to events like PRE_UPDATE
        pass  # Implemented by subclass or configured
```

### Event Registry with Handler Registration
```python
registry = EventRegistry()

# Option 1: Register queue/cache as event handler
registry.register_handler(EventType.PRE_UPDATE, async_task_runner)

# Option 2: Register separate handler with queue/cache access
recomputer = ProximalRecomputer(async_task_runner)
registry.register_handler(EventType.PRE_UPDATE, recomputer)

# Fire event BEFORE update
registry.fire_event(EventContext(
    EventType.PRE_UPDATE,
    engine=engine,
    config=config,
    logger=logger,
))
# Returns when all handlers done

# Now update weights
engine.update_weights()

# Fire event AFTER update
registry.fire_event(EventContext(
    EventType.POST_UPDATE, ...
))
```

## Integration Points

### Point 1: Filter Registration (At Initialization)
```python
# In create_workflow_executor_with_events()
executor, registry = create_workflow_executor_with_events(config, engine)

if config.enable_segment_wise_ppo:
    # Register add filter for admission control
    staleness_filter = StalenessFilter(config.max_head_offpolicyness)
    executor.runner.register_add_filter(staleness_filter)

    # Register event handler for PRE_UPDATE
    recomputer = ProximalRecomputer(executor.runner)
    registry.register_handler(EventType.PRE_UPDATE, recomputer)
```

### Point 2: Event Firing (In Model Update)
```python
# In RemoteInfEngine.update_weights_from_disk()
def update_weights_from_disk(self, meta):
    # Fire PRE_UPDATE event
    if hasattr(self, '_event_registry'):
        context = EventContext(
            EventType.PRE_UPDATE,
            engine=self,
            config=self.config,
            logger=self.logger,
        )
        self._event_registry.fire_event(context)
        # Returns here = all handlers (recompute) done

    # NOW update weights
    fut = self.executor.submit(_update_weights_from_disk, ...)

    # Fire POST_UPDATE event (after future completes)
    def on_complete(future):
        if hasattr(self, '_event_registry'):
            context = EventContext(
                EventType.POST_UPDATE,
                engine=self,
                config=self.config,
                logger=self.logger,
            )
            self._event_registry.fire_event(context)

    fut.add_done_callback(on_complete)
    return fut
```

## Summary

**Filters**:
- Registered on Queue/Cache
- Checked during add() operations
- Accept/reject at admission

**Event Handlers**:
- Registered in EventRegistry
- React to events (PRE_UPDATE, POST_UPDATE)
- Synchronous execution

**Critical**: Recompute happens in PRE_UPDATE handler, BEFORE weight update!
