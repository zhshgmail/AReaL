# Hydra Config Override Issue: Why '+' Was Required

## The Problem

When trying to override `enable_segment_wise_ppo` from the command line:

```bash
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
  --config examples/math/gsm8k_grpo.yaml \
  enable_segment_wise_ppo=False  # ✗ FAILED!
```

**Error:**
```
omegaconf.errors.ConfigAttributeError: Key 'enable_segment_wise_ppo' is not in struct
    full_key: enable_segment_wise_ppo
    object_type=dict
```

**Workaround that worked:**
```bash
+enable_segment_wise_ppo=False  # ✓ Works but confusing
```

## Root Cause

### How Hydra Config Loading Works

1. **Step 1**: Hydra loads the YAML file → creates unstructured config
   ```python
   # gsm8k_grpo.yaml does NOT contain enable_segment_wise_ppo
   yaml_cfg = {
       "experiment_name": "gsm8k-grpo",
       "seed": 1,
       # ... no enable_segment_wise_ppo field
   }
   ```

2. **Step 2**: Hydra applies command-line overrides **BEFORE** merging with dataclass
   ```python
   # Tries to set enable_segment_wise_ppo on yaml_cfg
   yaml_cfg['enable_segment_wise_ppo'] = False  # ✗ Key doesn't exist!
   ```

3. **Step 3**: The YAML config is loaded with implicit `struct=True` mode
   - This prevents adding new keys that don't exist
   - Override fails with "not in struct" error

4. **Step 4** (if override succeeded): Later, merge with structured config
   ```python
   # This is where enable_segment_wise_ppo would be added from BaseExperimentConfig
   default_cfg = OmegaConf.structured(GRPOConfig)  # Has enable_segment_wise_ppo
   cfg = OmegaConf.merge(default_cfg, yaml_cfg)    # But too late!
   ```

### Why '+' Works

The `+` prefix tells Hydra to **force add** the key even in struct mode:
```bash
+enable_segment_wise_ppo=False  # Force add new key
```

This bypasses the struct mode restriction, but it's confusing for users who don't understand Hydra internals.

## The Solution

### Approach Taken

Created a NEW example config `examples/math/gsm8k_grpo_segment_wise.yaml` that explicitly sets the flag:

```yaml
experiment_name: gsm8k-grpo
trial_name: trial0

seed: 1
total_train_epochs: 10
tokenizer_path: ${actor.path}
async_training: true

# Segment-wise Decoupled PPO (defaults to true in code if not specified)
# Set to false to use standard PPO behavior
enable_segment_wise_ppo: true  # ← Added this
```

### Why This Works

Now the field exists in the YAML, so command-line overrides work normally:

1. **Step 1**: Hydra loads YAML
   ```python
   yaml_cfg = {
       "experiment_name": "gsm8k-grpo",
       "enable_segment_wise_ppo": True,  # ✓ Field exists!
   }
   ```

2. **Step 2**: Apply override
   ```python
   yaml_cfg['enable_segment_wise_ppo'] = False  # ✓ Works!
   ```

3. **Step 3**: Merge with structured config (inherits the overridden value)

## Usage

### Using Existing Configs (Backward Compatible)

For configs that DON'T have the field explicitly set, use '+' prefix:

```bash
# Use existing config, explicitly disable feature
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
  --config examples/math/gsm8k_grpo.yaml \
  +enable_segment_wise_ppo=false
```

### Using New Example Config (Recommended)

For the new example config with explicit flag:

```bash
# Use new config with normal override
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
  --config examples/math/gsm8k_grpo_segment_wise.yaml \
  enable_segment_wise_ppo=false
```

## Benefits of Adding to YAML

1. **Explicit Configuration**: Users can see the setting in the config file
2. **Easy Override**: No need for special Hydra syntax (`+` prefix)
3. **Self-Documenting**: Comments explain what the flag does
4. **Consistent Behavior**: Same as other config fields

## When to Use '+' Prefix

The `+` prefix is still useful for **truly optional** fields that:
- Don't exist in the dataclass schema
- Are experimental or debugging options
- Should not be in the default config

For **supported features** like `enable_segment_wise_ppo`, always add them to the YAML explicitly.

## Related OmegaConf Concepts

### Struct Mode

```python
# Structured config (struct=True by default)
cfg = OmegaConf.structured(MyDataClass)
cfg.new_field = "value"  # ✗ Error: not in struct

# With struct=False
OmegaConf.set_struct(cfg, False)
cfg.new_field = "value"  # ✓ Works
```

### Override Syntax

```bash
key=value           # Set existing key (fails if struct=True and key missing)
+key=value          # Force add new key (even in struct mode)
~key                # Delete key
++key=value         # Force override (even if key is missing or read-only)
```

## Recommendation for Config Design

When adding new top-level config fields:

1. ✅ **DO**: Create NEW example configs that demonstrate the feature
2. ✅ **DO**: Add comments explaining the field and its impact
3. ✅ **DO**: Use the default value from the dataclass in the new example
4. ✅ **DO**: Keep existing example configs unchanged for backward compatibility
5. ❌ **DON'T**: Modify existing user-facing example configs
6. ❌ **DON'T**: Require users to use `+` prefix for standard features in new examples

## Summary

- **Problem**: Missing field in YAML + struct mode = override fails
- **Workaround**: Use `+` prefix to force add (e.g., `+enable_segment_wise_ppo=false`)
- **Proper Solution**: Create new example config with field explicitly set
- **Backward Compatibility**: Existing configs continue to work, use `+` for overrides
- **New Config**: `examples/math/gsm8k_grpo_segment_wise.yaml` demonstrates the feature

---

**New File**: `examples/math/gsm8k_grpo_segment_wise.yaml`
**Status**: ✅ Created (existing configs unchanged for backward compatibility)
