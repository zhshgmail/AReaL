# SGLang Cache Flush Race Condition Issue

## Problem

BOBA training fails at step 240 with:

```
AssertionError: Cache flush failed after updating weights
Cache not flushed because there are pending requests. #running-req: 29
```

## Root Cause Analysis

### Weight Update Flow in AReaL

1. Training engine updates model weights via NCCL/distributed
1. Inference engine (SGLang) receives weight update request
1. SGLang must flush KV cache because:
   - KV cache stores `K = tokens @ W_key`, `V = tokens @ W_value`
   - When W_key/W_value are updated, cached K/V are computed with OLD weights
   - Using stale cache would produce incorrect inference results
   - **Cache flush is REQUIRED for correctness**

### The Race Condition

```python
# In SGLang scheduler.py:
def update_weights_from_distributed(recv_req):
    success = tp_worker.update_weights_from_distributed(recv_req)
    if success:
        if recv_req.flush_cache:  # Default is True
            flush_cache_success = self.flush_cache()  # Synchronous!
            assert flush_cache_success  # ← CRASHES HERE

def flush_cache(self):
    if (waiting_queue.empty() and running_batch.is_empty()):
        # ... flush cache ...
        return True
    else:
        logger.warning(f"Cache not flushed, #running-req: {n}")
        return False  # ← Returns False if requests still running
```

**The problem:**

1. `recv_req.abort_all_requests=True` triggers **async** request abort
1. `flush_cache()` is called **immediately** (synchronous)
1. Abort hasn't completed yet → requests still running
1. `flush_cache()` returns `False` → assertion fails → crash

### Why This Happens

- `pause()` in AReaL stops accepting NEW requests but doesn't wait for existing ones
- `abort_all_requests` aborts asynchronously in tokenizer_manager
- Weight update happens immediately without waiting for abort
- 29 requests were still in flight when flush was attempted

## Incorrect Solutions (And Why They Don't Work)

### ❌ Solution 1: Set `flush_cache=False`

**Why it doesn't work:** KV cache MUST be flushed after weight update for correctness.
Cached K/V are computed with old weights and will produce wrong results.

### ❌ Solution 2: Add explicit `flush_cache=True`

**Why it doesn't work:** It's already `True` by default in SGLang's dataclass
definition.

## Correct Solutions

### Option A: Add Retry Logic to SGLang (Upstream Fix)

Modify `scheduler.py` in SGLang:

```python
def update_weights_from_distributed(recv_req):
    success = tp_worker.update_weights_from_distributed(recv_req)
    if success and recv_req.flush_cache:
        # Retry flush with timeout
        max_retries = 10
        retry_delay = 0.1  # 100ms
        for i in range(max_retries):
            if self.flush_cache():
                break
            time.sleep(retry_delay)
        else:
            # After retries, log error but don't crash
            logger.error("Cache flush failed after retries")
            # Or make it non-fatal in production
```

**Status:** This needs to be reported to SGLang team.

### Option B: Wait Before Weight Update (Workaround in AReaL)

Add delay after `pause()` to let requests complete:

```python
rollout.pause()
time.sleep(0.5)  # Wait for requests to abort
actor.update_weights(weight_update_meta)
```

**Downsides:**

- Adds latency to training loop
- 0.5s may not be enough under heavy load
- Doesn't solve root cause

### Option C: Make Pause Synchronous (Proper Fix in AReaL)

Modify `pause()` to wait for requests to complete:

```python
def pause(self):
    self._engine.pause()
    # Poll until no running requests
    max_wait = 5.0  # seconds
    start = time.time()
    while time.time() - start < max_wait:
        if self._get_num_running_requests() == 0:
            return
        time.sleep(0.05)
    logger.warning(f"pause() timeout, still have running requests")
```

**Status:** Requires implementing `_get_num_running_requests()` API.

### ~~Option D: Increase Max Offpolicyness~~ (INVALID)

**This does NOT help!** `max_head_offpolicyness` only controls future request
submission, not already-running requests. The 29 running requests that cause the flush
to fail were already accepted before pause. Increasing this parameter won't help.

## Recommended Action

**The only real solutions are A, B, or C:**

1. **Short-term workaround:** Implement Option B (add sleep after pause) - quick but
   hacky
1. **Medium-term fix:** Implement Option C (synchronous pause) in AReaL - proper fix
1. **Long-term fix:** Report to SGLang team and wait for Option A (retry logic) - best
   fix

## Test Case for SGLang Team

```python
# Reproduce the issue:
# 1. Start SGLang server with decoupled PPO training
# 2. Generate requests at high rate (29+ concurrent)
# 3. Call update_weights_from_distributed with abort_all_requests=True
# 4. Observe: assertion failure due to race condition
```

## References

- SGLang scheduler.py: `update_weights_from_distributed()` at line ~1072
- SGLang io_struct.py: `UpdateWeightsFromDistributedReqInput` default `flush_cache=True`
- AReaL issue: Seen at step 240 in BOBA training on H100
