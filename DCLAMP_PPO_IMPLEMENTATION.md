# DClamp-PPO Implementation Summary

This document summarizes the implementation of Directional-Clamp PPO (DClamp-PPO) based on the paper "Directional-Clamp PPO" (https://arxiv.org/pdf/2511.02577).

## Overview

DClamp-PPO addresses a key limitation of standard PPO: during training, approximately 35-40% of importance ratios move in the "wrong direction," meaning they decrease the likelihood of advantageous actions or increase the likelihood of disadvantageous ones.

### Key Insight from the Paper

The paper identifies that PPO's objective encourages ratios to move in the "right" direction:
- **Right direction**:
  - Positive advantage → increase ratio (ratio > 1)
  - Negative advantage → decrease ratio (ratio < 1)

- **Wrong direction**:
  - Positive advantage → decrease ratio (ratio < 1)
  - Negative advantage → increase ratio (ratio > 1)

DClamp-PPO introduces a penalty specifically for the **strict wrong direction** regions:
- For positive advantages: strict wrong direction is when `ratio < 1 - β`
- For negative advantages: strict wrong direction is when `ratio > 1 + β`

## Mathematical Formulation

### Standard PPO Objective
```
J_PPO = min(w * A, clip(w, 1-ε, 1+ε) * A)
```

### DClamp-PPO Objective
```
J_DClamp = min(w * A, clip(w, 1-ε, 1+ε) * A, f_DClamp(w, α, β) * A)
```

Where:
```
f_DClamp(w, α, β) = {
  α * w - (α - 1) * (1 - β)  if A > 0
  α * w - (α - 1) * (1 + β)  if A < 0
}
```

### Hyperparameters

1. **α (alpha)**: Controls the slope of the penalty in strict wrong direction regions
   - Must be > 1.0
   - Paper recommends: **3.0**

2. **β (beta)**: Defines the strict wrong direction region
   - Must be in (0, 1]
   - Paper recommends: **same as ε (eps_clip), typically 0.2**
   - Defaults to `eps_clip` if not specified

## Implementation Details

### Files Modified

1. **`areal/api/cli_args.py`**
   - Added `dclamp_alpha` parameter (lines 474-479)
   - Added `dclamp_beta` parameter (lines 480-485)
   - Both parameters in `PPOActorConfig` class

2. **`areal/utils/functional.py`**
   - Added `dclamp_alpha` and `dclamp_beta` parameters to `ppo_actor_loss_fn` (lines 277-278)
   - Implemented DClamp penalty logic (lines 334-359)
   - Added `dclamp_mask` to statistics tracking (lines 374, 381)
   - Updated docstring with parameter documentation (lines 296-302)

3. **`areal/engine/ppo/actor.py`**
   - Added parameters to `grpo_loss_fn` signature (lines 338-339)
   - Passed parameters to `ppo_actor_loss_fn` call (lines 391-392)
   - Added parameters to `functools.partial` call (lines 288-289)
   - Added `dclamp_tokens` to stats tracking (line 401)
   - Added scalar logging for DClamp parameters (lines 257-266)

4. **`areal/tests/test_functional.py`**
   - Added comprehensive test suite for DClamp-PPO (lines 764-1033)
   - Tests cover: basic functionality, parameter validation, penalty application, and integration with other PPO features

5. **`examples/dclamp-ppo-example-config.yaml`**
   - Created example configuration file showing how to use DClamp-PPO

### Key Implementation Points

1. **Backward Compatibility**: DClamp-PPO is disabled by default (when `dclamp_alpha=None`)
2. **Default Behavior**: When `dclamp_beta=None` and `dclamp_alpha` is set, beta defaults to `eps_clip`
3. **Integration**: Works seamlessly with existing PPO features:
   - Dual clipping (`c_clip`)
   - Decoupled loss (`use_decoupled_loss`)
   - Behavioral importance weighting (`behav_imp_weight_cap`)
   - M2PO (`m2_threshold`)
   - GSPO (`importance_sampling_level='sequence'`)

4. **Statistics Tracking**: Added `dclamp_mask` and `dclamp_tokens` metrics to monitor when the penalty is applied

## Usage Example

### Basic Configuration

```yaml
actor:
  # Standard PPO parameters
  eps_clip: 0.2

  # Enable DClamp-PPO
  dclamp_alpha: 3.0
  dclamp_beta: 0.2  # Optional, defaults to eps_clip
```

### Disabling DClamp-PPO

```yaml
actor:
  eps_clip: 0.2
  # DClamp-PPO disabled (default)
  dclamp_alpha: null
```

### Python Code Example

```python
from areal.utils.functional import ppo_actor_loss_fn

loss, stat = ppo_actor_loss_fn(
    logprobs=logprobs,
    proximal_logprobs=proximal_logprobs,
    old_logprobs=old_logprobs,
    advantages=advantages,
    eps_clip=0.2,
    loss_mask=loss_mask,
    dclamp_alpha=3.0,    # Enable DClamp-PPO
    dclamp_beta=0.2,     # Define strict wrong direction region
)

# Access DClamp statistics
dclamp_affected = stat['dclamp_mask'].sum()
print(f"Tokens affected by DClamp: {dclamp_affected}")
```

## Expected Results

Based on the paper's findings:

1. **Reduced Wrong Direction Updates**: DClamp-PPO reduces the proportion of samples in strict wrong direction regions by approximately 33% compared to standard PPO

2. **Performance Improvements**:
   - Hopper-v4: +38.3% improvement
   - Swimmer-v4: +26.3% improvement
   - Humanoid-v4: +26.3% improvement
   - Other environments: Competitive or improved performance

3. **Better Trust-Region Adherence**: Lower MSE of importance ratios from 1, indicating updates stay closer to the intended trust-region

## Monitoring

Monitor these metrics during training:

1. **`use_dclamp`**: Whether DClamp is enabled (0 or 1)
2. **`dclamp_alpha`**: The alpha parameter value
3. **`dclamp_beta`**: The beta parameter value
4. **`dclamp_tokens`**: Proportion of tokens affected by DClamp penalty
5. **`importance_weight`**: Track ratio distributions to verify they stay closer to 1

## Testing

Run the test suite to verify the implementation:

```bash
pytest areal/tests/test_functional.py::TestDClampPPO -v
```

Tests cover:
- Basic functionality and shapes
- Parameter validation (alpha > 1, beta in (0,1])
- Penalty application in strict wrong direction
- No penalty in right direction
- Integration with dual clipping
- Edge cases and consistency

## References

- Paper: "Directional-Clamp PPO" (https://arxiv.org/pdf/2511.02577)
- Authors: Gilad Karpel, Shoham Sabach, Ruida Zhou, Mohammad Ghavamzadeh
- Institution: Technion - Israel Institute of Technology, Amazon AGI

## Notes

1. The implementation follows the paper's formulation exactly
2. All existing PPO functionality is preserved
3. DClamp-PPO can be easily toggled on/off via configuration
4. The penalty only activates in strict wrong direction regions, preserving exploration
5. Compatible with the existing decoupled PPO implementation in AReaL
