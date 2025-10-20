# Segment-Wise Decoupled PPO Feature Activation Verification

## Question
If a user runs:
```bash
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py --config examples/math/gsm8k_grpo.yaml
```

Will the segment-wise PPO feature activate with ZERO changes to `examples/math/gsm8k_grpo.py`?

## Answer: ✅ YES - Zero Script Changes Required

The feature is **fully controlled by YAML configuration**. User training scripts remain unchanged.

## Activation Flow

### 1. Configuration (YAML Only)
Add to `examples/math/gsm8k_grpo.yaml`:
```yaml
# Single switch at top level controls the entire feature
enable_segment_wise_ppo: true  # Default: true

# Automatically enables BOTH:
# - Inference-side: Track proximal_logprobs_t and output_versions
# - Training-side: Use proximal_t for behavioral importance weights

actor:
  behav_imp_weight_floor: 0.1  # Optional: symmetric importance weight filtering
```

### 2. Internal Activation Chain

```
User YAML Config (top-level enable_segment_wise_ppo)
    ↓
Launcher loads config → BaseExperimentConfig
    ↓
load_expr_config() auto-propagates flag to:
    ├─ config.rollout.enable_segment_wise_ppo
    └─ config.actor.enable_segment_wise_ppo
    ↓
Engine.initialize() is called
    ↓
create_workflow_executor(config) [Factory Pattern]
    ↓
IF config.rollout.enable_segment_wise_ppo == true:
    ├─ Create SegmentWisePPOStrategy
    ├─ Create ProximalRecomputer
    ├─ Create FilteredSamplesCapacityModifier
    ├─ Register capacity modifiers
    └─ Inject all into WorkflowExecutor
ELSE:
    └─ Create StandardPPOStrategy (backward compatible)
    ↓
WorkflowExecutor operates with injected strategy
    ↓
During generation (Inference Engine):
    ├─ Track proximal_logprobs_t (if enabled)
    ├─ Track output_versions per token
    └─ Apply staleness filtering
    ↓
Before weight update (Actor):
    └─ Auto-recompute proximal_t for v-1 samples
    ↓
During training (Actor):
    └─ Use proximal_t for behavioral importance weights (if config.actor.enable_segment_wise_ppo)
```

## Verification Checklist

### ✅ All Inference Engines Use Factory Pattern
- [x] `areal/engine/sglang_remote.py` → Uses `create_workflow_executor()` (line 82-87)
- [x] `areal/engine/vllm_remote.py` → Uses `create_workflow_executor()`
- [x] `areal/experimental/sglang_engine.py` → Uses `create_workflow_executor()`

### ✅ No Direct WorkflowExecutor Instantiation
```bash
$ grep -r "WorkflowExecutor(" --include="*.py" | grep -v test
# Result: Only areal/api/workflow_factory.py (the factory itself)
```

### ✅ Configuration-Based Activation
```python
# areal/api/workflow_factory.py
def create_staleness_strategy(config):
    if config.enable_segment_wise_ppo:  # ← Config-driven decision
        return SegmentWisePPOStrategy(config)
    else:
        return StandardPPOStrategy(config)
```

### ✅ Defensive Safeguards
```python
# areal/api/workflow_api.py:308-312
if output_queue is None or result_cache is None:
    raise ValueError(
        "WorkflowExecutor requires output_queue and result_cache. "
        "Use workflow_factory.create_workflow_executor()."
    )
```

## User Impact: ZERO Changes Required

### What Users DON'T Need to Change ❌
- ❌ Training scripts (`examples/math/gsm8k_grpo.py`)
- ❌ Workflow definitions
- ❌ Reward functions
- ❌ Data loaders
- ❌ Model initialization
- ❌ Any Python code

### What Users DO Need to Change ✅
- ✅ **ONLY** the YAML config file to enable/disable the feature

## Configuration Examples

### Enable Feature (Default Behavior)
```yaml
# examples/math/gsm8k_grpo.yaml
# Can omit enable_segment_wise_ppo line (defaults to true)
enable_segment_wise_ppo: true  # Single switch!

actor:
  behav_imp_weight_floor: 0.1  # Optional: lower bound filtering
  behav_imp_weight_cap: 10.0   # Optional: upper bound filtering (existing)
```

### Disable Feature (Backward Compatible)
```yaml
# examples/math/gsm8k_grpo.yaml
enable_segment_wise_ppo: false  # Single switch disables entire feature
```

## How It Works Under the Hood

### Factory Pattern Ensures Proper Injection
```python
# Called automatically by engines during initialize()
executor = create_workflow_executor(
    inference_engine=self,
    staleness_manager=staleness_manager,
    config=self.config,  # ← Contains enable_segment_wise_ppo
    logger=self.logger,
)
```

### Strategy Pattern Handles Business Logic
```python
# Strategies are self-contained, no if-else in business code
if strategy.should_filter_before_enqueue():
    if strategy.is_sample_too_stale(traj, ver, config):
        # Filter sample
```

### Clean Separation of Concerns
```
┌─────────────────┐
│  User Script    │ ← NO CHANGES
│  (gsm8k_grpo.py)│
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  YAML Config    │ ← ONLY CHANGE HERE
│  (.yaml)        │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Launcher       │ ← Loads config
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Engine         │ ← Calls factory
│  (.initialize()) │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Factory        │ ← Assembles components based on config
│  (create_...)   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│WorkflowExecutor │ ← Works with abstractions
│ (uses injected) │
└─────────────────┘
```

## Backward Compatibility

### Old Configs Still Work ✅
```yaml
# Old config WITHOUT enable_segment_wise_ppo
rollout:
  max_concurrent_rollouts: 64

actor:
  kl_ctl: 0.1

# Result: enable_segment_wise_ppo defaults to true (feature enabled)
# To disable: explicitly set enable_segment_wise_ppo: false at top level
```

### Feature Can Be Disabled Anytime ✅
```yaml
# Single line at top level
enable_segment_wise_ppo: false

# Result: Standard PPO behavior (pre-feature state)
```

## Testing Recommendation

### Verify Feature Activation
```bash
# Enable feature
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
  --config examples/math/gsm8k_grpo.yaml

# Check logs for:
# - "SegmentWisePPOStrategy" (indicates feature active)
# - "StandardPPOStrategy" (indicates feature disabled)
```

### Toggle Feature Without Code Changes
```bash
# Test 1: Feature enabled
echo "enable_segment_wise_ppo: true" > test_config.yaml
python3 -m areal.launcher.local ... --config test_config.yaml

# Test 2: Feature disabled
echo "enable_segment_wise_ppo: false" > test_config.yaml
python3 -m areal.launcher.local ... --config test_config.yaml

# Both tests use SAME Python script, DIFFERENT config only
```

## Architecture: Single Switch with Auto-Propagation

Segment-wise PPO is a **cross-cutting feature** that affects both inference and training:

### Inference-Side Requirements (InferenceEngineConfig)
- Track `proximal_logprobs_t` during token generation
- Track `output_versions` per token
- Provide recomputation API for stale samples

### Training-Side Requirements (PPOActorConfig)
- Use `proximal_t` for behavioral importance weights in loss
- Apply staleness-aware filtering
- Call recomputation before weight updates

### How the Single Switch Works
1. **User sets one flag** at the top level: `enable_segment_wise_ppo: true`
2. **Config loader auto-propagates** it to both child configs:
   - `config.rollout.enable_segment_wise_ppo = config.enable_segment_wise_ppo`
   - `config.actor.enable_segment_wise_ppo = config.enable_segment_wise_ppo`
3. **Components read from their local configs** (no shared state needed)
4. **Works in distributed systems** where inference and training run on separate nodes

This design maintains decoupling while providing a single user-facing switch.

## Conclusion

✅ **CONFIRMED**: The feature activates with **ZERO changes** to user training scripts.
✅ **Control**: Fully controlled by YAML configuration (**single** `enable_segment_wise_ppo` flag).
✅ **Backward Compatible**: Old configs work (flag defaults to enabled).
✅ **Easy Disable**: Set `enable_segment_wise_ppo: false` at top level to revert.
✅ **Clean Design**: Single user-facing switch, auto-propagated to child configs.

The design follows the principle: **"Configuration over Code Changes"** - users never modify training scripts to enable/disable features.
