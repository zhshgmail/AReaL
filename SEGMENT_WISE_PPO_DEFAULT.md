# Segment-Wise PPO: Default Enabled Status

## Current Default Value

**YES**, `enable_segment_wise_ppo` defaults to **`True`** in the codebase.

Location: `areal/api/cli_args.py:1218-1227`

```python
enable_segment_wise_ppo: bool = field(
    default=True,  # ← DEFAULT IS TRUE!
    metadata={
        "help": "Enable segment-wise decoupled PPO algorithm (single switch for entire feature). "
        "When enabled, the inference engine tracks proximal_logprobs_t and output_versions per token, "
        "and the training engine uses proximal_t for accurate behavioral importance weight calculation. "
        "This flag is automatically propagated to both InferenceEngineConfig and PPOActorConfig. "
        "Set to false to disable and use standard PPO behavior."
    },
)
```

## Impact on Your Command

Your command:
```bash
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
  --config examples/math/gsm8k_grpo.yaml \
  experiment_name=areal_smoke_test \
  trial_name=trail_0 \
  ...
```

Since `gsm8k_grpo.yaml` does **NOT** explicitly set `enable_segment_wise_ppo`, it will use the default value of `True`.

**This means your training is ALREADY using the segment-wise decoupled PPO feature!**

## How to Verify the Feature is Active

### 1. Check for Queue Purge Logs (Most Obvious)

When the model version increases (after each training step), you should see logs like:

```
[QueuePurge] ver_switch to 1: drained=123 picked_prev=45 dropped=12 kept=111 cache_size=234 v0(picked/dropped/kept)=30/5/25
```

These logs come from `areal/api/staleness_control.py:339-344` and **ONLY appear when segment-wise PPO is enabled**.

**Explanation of the log fields:**
- `ver_switch to N`: Model version switched to N
- `drained`: Total samples drained from output queue for inspection
- `picked_prev`: Samples containing tokens from previous version (N-1)
- `dropped`: Samples dropped due to staleness
- `kept`: Samples kept in queue
- `cache_size`: Current result cache size
- `v0(picked/dropped/kept)`: Statistics for samples with version 0 tokens

### 2. Check for Cache Filter Logs

If samples in the cache become stale, you'll see:

```
[CacheFilter] dropped_cache=5 size=200
```

This log appears at `areal/api/staleness_control.py:373`.

### 3. Check Initialization Logs

The factory creates different strategies based on the flag. You can add a temporary log to verify:

In `areal/api/workflow_factory.py:72-75`:
```python
if config.enable_segment_wise_ppo:
    logger.info("Using SegmentWisePPOStrategy (staleness filtering ENABLED)")
    return SegmentWisePPOStrategy(config)
else:
    logger.info("Using StandardPPOStrategy (staleness filtering DISABLED)")
    return StandardPPOStrategy(config)
```

### 4. Check Rollout Tracing Logs (if enabled)

If you set `rollout.enable_rollout_tracing: true` in the YAML, you'll see detailed logs like:

```
Submit rollout rid 123. Submit: 256, running: 128, accepted: 64.
Finish and accept rollout 123. Submit: 256, running: 127, accepted: 65.
Finish but reject rollout 124. Submit: 256, running: 126, accepted: 65.
Finish rollout 125 but filtered due to staleness (pre-enqueue)
```

## How to Disable the Feature

To use **standard PPO** behavior (no staleness filtering), add this to your YAML config:

```yaml
# At the top level of gsm8k_grpo.yaml
enable_segment_wise_ppo: false
```

Or pass it via command line:

```bash
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
  --config examples/math/gsm8k_grpo.yaml \
  enable_segment_wise_ppo=false \
  ...
```

## Behavioral Differences

### When `enable_segment_wise_ppo=True` (DEFAULT):

1. **StalenessControlStrategy**: `SegmentWisePPOStrategy`
   - Purges stale samples from queue when version increases
   - Filters stale samples from cache before returning
   - Pre-filters samples before enqueue to prevent queue overflow

2. **ProximalRecomputer**: Created and used
   - Recomputes proximal policy logprobs for head tokens

3. **FilteredSamplesCapacityModifier**: Created and registered
   - Dynamically adjusts capacity based on filtered samples

4. **Staleness Tracking**: Per-token version tracking via `_recompute_version` key

### When `enable_segment_wise_ppo=False`:

1. **StalenessControlStrategy**: `StandardPPOStrategy`
   - No staleness filtering (backward compatible with original AReaL)
   - No queue purging
   - No cache filtering

2. **ProximalRecomputer**: `None` (not created)

3. **FilteredSamplesCapacityModifier**: `None` (not created)

4. **Staleness Tracking**: Not used

## Summary

✅ **Your current command DOES enable segment-wise PPO** (because default is `True`)

✅ **Look for `[QueuePurge]` logs to confirm** - these only appear when the feature is active

⚠️ **To disable**, explicitly set `enable_segment_wise_ppo: false` in YAML or command line

---

**Recommendation**: Since the feature is already enabled by default, you might want to consider:

1. **Option A**: Keep it enabled (current state) - uses the new segment-wise PPO algorithm
2. **Option B**: Explicitly set it to `false` for backward compatibility with standard PPO
3. **Option C**: Add it to your YAML explicitly (either `true` or `false`) to make the behavior clear and intentional

For production use, I recommend **Option C** - being explicit about which mode you're using, rather than relying on defaults that might change.
