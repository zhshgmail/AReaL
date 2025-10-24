# Capacity Recovery Bug Analysis

## Bug Description

When a sample is filtered out due to staleness (before enqueueing to `output_queue`), the capacity recovery logic is **inconsistent** with the reference implementation.

## Current Behavior (INCORRECT)

In `workflow_executor.py` lines 458-493:

```python
if should_accept_traj:
    # Check staleness filtering
    current_ver = self.inference_engine.get_version()
    should_enqueue = True

    if self.staleness_strategy:
        should_enqueue = not self.staleness_strategy.is_sample_too_stale(
            traj, current_ver, self.config
        )

    if should_enqueue:
        # ACCEPTED: running-1, accepted+1
        self.staleness_manager.on_rollout_accepted()  # Line 470
        self.output_queue.put_nowait(_TimedResult(...))
    else:
        # FILTERED: running-1, accepted NOT changed!  ← BUG
        if self.filtered_capacity_modifier:
            self.filtered_capacity_modifier.on_samples_filtered(1, current_ver)  # Workaround
        self.staleness_manager.on_rollout_rejected()  # Line 493
```

### The Problem:

1. **Line 470**: `on_rollout_accepted()` → `accepted += 1, running -= 1`
2. **Line 493**: `on_rollout_rejected()` → `running -= 1` only

**But wait!** Looking more carefully at the code flow:
- Line 446: `if should_accept_traj:` - only enters this block if accepted
- Line 470: `on_rollout_accepted()` - called when `should_enqueue = True`
- Line 493: `on_rollout_rejected()` - called when `should_enqueue = False`

**Actually, the issue is different**:
- When `should_enqueue = False`, we never called `on_rollout_accepted()` yet
- But we're inside the `if should_accept_traj:` block, which means the workflow already accepted it
- **The real question**: Is `should_accept_traj` evaluated **before** staleness check?

Let me trace backwards...

## Code Flow Analysis

Looking at the workflow callback (need to check where `should_accept_traj` comes from):

```python
# Line 446
if should_accept_traj:
    # We are here because workflow said YES
    # Now we do staleness check
    if staleness_strategy.is_sample_too_stale():
        # Filtered!
        on_rollout_rejected()
```

**Key insight**: `should_accept_traj` is from the **workflow's decision**, not from staleness check.

So the flow is:
1. Workflow completes → generates trajectory
2. Workflow evaluates `should_accept` (e.g., checks rewards, length) → `should_accept_traj = True`
3. If workflow accepts, **then** we check staleness
4. If too stale, we filter it out

The question: At what point is `accepted` count incremented in the reference implementation?

## Reference Branch Behavior

In `b_li/boba-tmp:areal/api/workflow_api.py`:

```python
# When sample is dropped before enqueue:
if drop:
    with self.lock:
        self.rollout_stat.accepted -= 1  # ← DECREMENTS accepted!
        self.drop_count += 1
```

This implies that in the reference implementation:
- `accepted` was **already incremented** before the staleness check
- When filtered, it **decrements** accepted to compensate

## Root Cause

The issue is in the **order of operations**:

### Reference Implementation (b_li/boba-tmp):
1. Workflow completes → `accepted += 1, running -= 1`
2. Staleness check before enqueue
3. If filtered → `accepted -= 1` (to undo step 1)

### Current Implementation:
1. Workflow completes
2. Staleness check
3. If passes → `accepted += 1, running -= 1`
4. If filtered → `running -= 1` only

**Difference**: Current implementation does staleness check **before** incrementing accepted, which is actually **BETTER DESIGN**!

But there's still a subtle issue: Where is `running` incremented? Let me check...

## Checking Running Count

The `running` count should be incremented when we **submit** a rollout, not when it completes.

Looking at the counter:
- `submitted`: Incremented when calling `submit()`
- `running`: Should be `submitted - completed`
- `accepted`: Subset of completed that passed checks

Actually, let me re-read the `on_rollout_accepted` and `on_rollout_rejected` logic:

```python
def on_rollout_accepted(self) -> None:
    """When a rollout completes successfully and is accepted."""
    with self.lock:
        self.rollout_stat.accepted += 1
        self.rollout_stat.running -= 1

def on_rollout_rejected(self) -> None:
    """When a rollout completes but is rejected."""
    with self.lock:
        self.rollout_stat.running -= 1
```

So both decrement `running`, which is correct (both completed).
The difference is whether to increment `accepted`.

## The Real Issue

Looking back at lines 458-493, I think the logic is actually **CORRECT**:

```python
if should_accept_traj:  # Workflow said YES
    # Check staleness
    should_enqueue = not self.staleness_strategy.is_sample_too_stale(...)

    if should_enqueue:
        self.staleness_manager.on_rollout_accepted()  # accepted+1, running-1
    else:
        self.staleness_manager.on_rollout_rejected()  # running-1 only
```

This means:
- If filtered by staleness → `running -= 1`, `accepted` unchanged (stays 0)
- No need to decrement `accepted` because it was never incremented!

**The `filtered_capacity_modifier` is the correct design!**
- It compensates for samples that completed but weren't accepted
- Adds them back to capacity to allow new samples

## Conclusion

After careful analysis, the current implementation is **ACTUALLY CORRECT**!

The difference from reference implementation is:
- **Reference**: Increments `accepted` early, then decrements if filtered
- **Current**: Only increments `accepted` if not filtered

Both achieve the same result, but current design is cleaner.

The `filtered_capacity_modifier` is needed because:
- Capacity = `max_concurrent - running`
- When sample is filtered, `running` decreases, freeing capacity
- But we want to generate **more** samples to compensate for filtered ones
- So we add `filtered_count` to capacity

## Final Verification Needed

Need to verify that `running` count is managed correctly throughout the lifecycle:
1. Submit → `running += 1`?
2. Complete (accepted) → `running -= 1`
3. Complete (rejected) → `running -= 1`

Let me check where `running` is incremented...
