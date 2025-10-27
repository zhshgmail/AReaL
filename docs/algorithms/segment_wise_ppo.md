# Segment-wise Decoupled PPO

## Overview

Segment-wise decoupled PPO (SDP) is an improvement to AReaL's existing decoupled PPO for asynchronous RL training. The key innovation is **replacing π_proximal with π_proximal_t in the behavioral importance weight only**, using the "next version of behavior policy" (π_{v+1}) instead of the current policy. This significantly reduces variance in importance weights and improves training stability.

## Motivation

### Background: Decoupled PPO in AReaL

AReaL already supports decoupled PPO (see [AReaL paper](https://arxiv.org/pdf/2505.24298)), which separates the PPO objective into two parts:

```
Loss = PPO_clipped_ratio × behavioral_importance_weight

Where:
  PPO_clipped_ratio = clip(π_current / π_proximal)
  behavioral_importance_weight = π_proximal / π_behavior
```

In asynchronous training, π_proximal (current policy) can be many versions ahead of π_behavior, leading to **high variance** in behavioral importance weights.

### Improvement: Segment-wise Decoupled PPO

SDP modifies **only the behavioral importance weight** by replacing π_proximal with π_proximal_t:

```
Standard Decoupled PPO:
  behavioral_importance_weight = π_proximal / π_behavior
  where π_proximal = π_current (current policy)

Segment-wise Decoupled PPO:
  behavioral_importance_weight = π_proximal_t / π_behavior
  where π_proximal_t = π_{v+1} (next version after behavior)

Note: The PPO_clipped_ratio still uses π_proximal (current policy)
```

**Key Benefits:**
1. **Reduced Variance**: π_{v+1} is much closer to π_v than π_current → more stable importance weights
2. **Per-token Granularity**: Different tokens use different π_proximal_t based on when they were generated
3. **Better Training Stability**: Lower variance in behavioral importance weights → more stable gradients

### Why This Works

Example with max_head_offpolicyness=2:
- Token generated at v=5, trained when current=v=7

**Standard Decoupled PPO:**
```
behavioral_importance_weight = π_7 / π_5  (large gap → high variance)
```

**Segment-wise Decoupled PPO:**
```
behavioral_importance_weight = π_6 / π_5  (one-step ratio → low variance)
```

The one-step importance weight (π_{v+1}/π_v) is much more stable than the multi-step weight (π_current/π_v).

## Configuration

Enable the feature by adding one line to your rollout config:

```yaml
rollout:
  max_head_offpolicyness: 2  # Control staleness tolerance
  enable_segment_wise_ppo: true  # Enable segment-wise decoupled PPO
```

**Default**: `enable_segment_wise_ppo: true` (enabled by default)

When disabled (`false`), the system behaves as standard PPO without per-token version tracking.

## How It Works

### 1. Generation Phase

During rollout at version v, the engine tracks:
- `output_versions[t]`: Which model version generated token t
- `old_logprobs[t]`: log π_v(token_t) - behavior policy
- `proximal_logprobs_t[t]`: Initialized to old_logprobs[t]

### 2. Recompute Phase (Automatic)

Before weight updates at version v_current, `pause()` automatically triggers recompute via a pre-pause hook.

For each token generated at version v, we recompute proximal_t using version v+1:

```python
if token_version == v_current - 1:  # Tokens from v-1
    # Use current policy (v_current) as the "next version" (v+1)
    proximal_logprobs_t[t] = log π_{v_current}(token_t | context)
# Tokens at v_current are not recomputed (already current version)
```

**Result:** Each token's proximal_t represents π_{v+1} (next version after its behavior policy π_v).

### 3. Training Phase

The loss function uses both standard PPO clipping AND behavioral importance weighting:

```python
# Standard PPO clipped ratio (uses current policy)
ratio = exp(logprobs - proximal_logprobs)  # π_current / π_proximal

# Behavioral importance weight (uses proximal_t)
if proximal_logprobs_t is not None:  # SDP enabled
    behav_imp_weight = exp(proximal_logprobs_t - old_logprobs)  # π_proximal_t / π_behavior
else:  # Standard decoupled PPO
    behav_imp_weight = exp(proximal_logprobs - old_logprobs)  # π_proximal / π_behavior

# Final loss
pg_loss = clipped_ppo_loss * behav_imp_weight
```

**Key Point:** Only the behavioral importance weight uses π_proximal_t. The PPO clipping still uses π_proximal (current policy).

## Hook System

The feature uses an extensible hook/callback architecture:

```python
# Hooks are registered automatically when feature is enabled
# But users can also register custom hooks

executor.register_pre_pause_hook(my_custom_function)
executor.register_post_pause_hook(another_function)
executor.register_pre_resume_hook(before_resume)
executor.register_post_resume_hook(after_resume)
```

Hooks execute synchronously in registration order, allowing for future extensions without coupling core logic.

## Example Configuration

See [examples/math/gsm8k_grpo_sdp.yaml](../../examples/math/gsm8k_grpo_sdp.yaml) for a complete configuration example.

```yaml
experiment_name: gsm8k-grpo
trial_name: trial0

rollout:
  max_concurrent_rollouts: 256
  max_head_offpolicyness: 2
  enable_segment_wise_ppo: true  # Enable the feature

actor:
  path: Qwen/Qwen2.5-1.5B-Instruct
  eps_clip: 0.4
  behav_imp_weight_cap: 5.0
  behav_imp_weight_floor: 0.2
  use_decoupled_loss: true
```

## Monitoring

When enabled, you can monitor the following metrics:
- `grpo_actor/behav_imp_weight/avg`: Average behavioral importance weight
  - **With SDP**: Should be closer to 1.0 (one-step ratio π_{v+1}/π_v)
  - **Without SDP**: Can be much larger (multi-step ratio π_current/π_v)
- `grpo_actor/behav_imp_weight/std`: Standard deviation of importance weights
  - **With SDP**: Lower variance → more stable training
  - **Without SDP**: Higher variance → less stable gradients
- `grpo_actor/behav_kl/avg`: KL divergence between proximal and behavior policies

The key advantage of SDP is **reduced variance** in importance weights, leading to more stable and efficient training.

## Backward Compatibility

When `enable_segment_wise_ppo: false`:
- Engine does not populate `proximal_logprobs_t` (returns None)
- Workflows skip creating the key in result dictionaries
- Loss computation falls back to standard PPO behavior
- No performance overhead

This ensures the system remains fully backward compatible with existing training scripts.

## Implementation Details

The feature is implemented across several components:

1. **Engine** (`areal/engine/sglang_remote.py`): Tracks versions and proximal_t during generation
2. **Workflow** (`areal/workflow/rlvr.py`, `vision_rlvr.py`): Conditionally adds proximal_t to data
3. **Executor** (`areal/api/workflow_api.py`): Hook system and automatic recompute
4. **Loss** (`areal/utils/functional.py`): Segment-wise importance weight calculation

## Testing

The feature includes comprehensive test coverage (130+ tests):
- Hook system tests (execution order, timing, exception handling)
- Backward compatibility tests
- Integration tests for recompute workflow
- Edge case handling

Run tests with:
```bash
python -m pytest areal/tests/seg_decoupled_ppo/ -v
```

## Technical Summary

### Decoupled PPO Loss (AReaL Baseline)

```
Loss = clipped_PPO_loss × behavioral_importance_weight

Where:
  clipped_PPO_loss uses: ratio = π_current / π_proximal
  behavioral_importance_weight = π_proximal / π_behavior

With π_proximal = π_current, this becomes:
  behavioral_importance_weight = π_current / π_behavior
```

**Problem:** High variance when π_current is many versions ahead of π_behavior.

### Segment-wise Decoupled PPO (This Feature)

```
Loss = clipped_PPO_loss × behavioral_importance_weight

Where:
  clipped_PPO_loss uses: ratio = π_current / π_proximal (unchanged)
  behavioral_importance_weight = π_proximal_t / π_behavior (changed!)

For token generated at v, trained at v_current:
  π_proximal_t = π_{v_current} (which is the "next version" after v)
  behavioral_importance_weight = π_{v+1} / π_v
```

**Solution:** One-step importance weight → **much lower variance** than multi-step weight.

### Key Difference

- **Standard**: Replaces entire π_proximal with π_current everywhere
- **Segment-wise**: Only replaces π_proximal with π_proximal_t in the behavioral importance weight
- The PPO clipping still uses π_current as π_proximal

## References

Based on AReaL's decoupled PPO architecture ([paper](https://arxiv.org/pdf/2505.24298)). The key innovation is replacing π_proximal with π_proximal_t (next version after behavior) **only in the behavioral importance weight**, which minimizes variance while maintaining the standard PPO clipping objective.
