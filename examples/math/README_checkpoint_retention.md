# Checkpoint Retention Configuration Example

This directory contains an example configuration for BOBA training with checkpoint
retention management enabled.

## Configuration File

**`boba_grpo_vllm_with_retention.yaml`** - BOBA GRPO training with:

- Proximal log-probability approximation (linear method)
- Automatic checkpoint retention (keep last 2 step checkpoints)
- Auto-recovery enabled

## Key Features Demonstrated

### 1. Proximal Log-Probability Approximation

```yaml
actor:
  use_prox_approx: true
  prox_approx_method: linear
  log_prox_approx_metrics: false
```

This eliminates expensive forward passes for proximal policy log-probabilities in
decoupled PPO, providing ~1.27x speedup with minimal accuracy impact.

### 2. Checkpoint Retention Management

```yaml
saver:
  freq_epochs: 1         # Save at end of each epoch
  freq_steps: 100        # Save every 100 steps
  enable_retention: true
  step_max_to_keep: 2    # Keep only last 2 step checkpoints
  protect_epoch_checkpoints: true  # Never delete epoch checkpoints
```

**Behavior:**

- Step checkpoints saved every 100 steps
- Only last 2 step checkpoints retained (older ones automatically deleted)
- Epoch checkpoints saved at end of each epoch (protected from deletion)
- Significant disk space savings for long training runs

### 3. Automatic Recovery

```yaml
recover:
  mode: auto            # Automatically resume if checkpoints exist
  freq_epochs: 1        # Save recovery checkpoint each epoch
  freq_steps: 100       # Save recovery checkpoint every 100 steps
  freq_secs: 3600       # Time-based backup (hourly)
  retries: 3            # Retry up to 3 times on failure
```

**Behavior:**

- Training automatically resumes from last recovery checkpoint if interrupted
- Recovery checkpoints are separate from regular checkpoints
- Multiple recovery strategies (step-based, epoch-based, time-based)

## Usage

### Basic Usage

```bash
# Use the configuration file directly
python -m areal.launcher.slurm \
    --config examples/math/boba_grpo_vllm_with_retention.yaml
```

### Override Parameters

```bash
# Keep more step checkpoints
python -m areal.launcher.slurm \
    --config examples/math/boba_grpo_vllm_with_retention.yaml \
    saver.step_max_to_keep=5

# Change checkpoint frequency
python -m areal.launcher.slurm \
    --config examples/math/boba_grpo_vllm_with_retention.yaml \
    saver.freq_steps=50

# Disable retention (keep all checkpoints)
python -m areal.launcher.slurm \
    --config examples/math/boba_grpo_vllm_with_retention.yaml \
    saver.enable_retention=false
```

### Archive Old Checkpoints Instead of Deleting

```bash
# Move old checkpoints to archive directory
python -m areal.launcher.slurm \
    --config examples/math/boba_grpo_vllm_with_retention.yaml \
    saver.step_cleanup_action=archive \
    saver.archive_root=/path/to/archive
```

### Compress and Archive

```bash
# Compress old checkpoints to save space
python -m areal.launcher.slurm \
    --config examples/math/boba_grpo_vllm_with_retention.yaml \
    saver.step_cleanup_action=compress_archive \
    saver.archive_root=/path/to/archive
```

## Checkpoint Directory Structure

With this configuration, checkpoints are organized as:

```
/tmp/areal/experiments/checkpoints/{user}/boba_ppo_vllm/trial0/
├── default/
│   ├── epoch0epochstep100globalstep100/    # Step checkpoint (rotated)
│   ├── epoch0epochstep200globalstep200/    # Step checkpoint (kept - last 2)
│   ├── epoch0epochstep300globalstep300/    # Step checkpoint (kept - last 2)
│   ├── epoch0epochstep99globalstep99/      # Epoch checkpoint (protected)
│   ├── epoch1epochstep99globalstep199/     # Epoch checkpoint (protected)
│   ├── recover_checkpoint/                 # Recovery checkpoint (separate)
│   └── .checkpoint_metadata.json           # Retention metadata
└── recover_info/                           # Recovery state
```

## Comparison with Original Configuration

| Feature                | Original           | With Retention        |
| ---------------------- | ------------------ | --------------------- |
| Proximal Approximation | ❌ Disabled        | ✅ Enabled (linear)   |
| Step Checkpoints       | ❌ Not saved       | ✅ Every 100 steps    |
| Checkpoint Cleanup     | ❌ Manual          | ✅ Automatic (keep 2) |
| Auto Recovery          | ❌ Disabled        | ✅ Enabled            |
| Disk Space Usage       | Grows indefinitely | Bounded               |

## Advanced Configuration Options

### Separate Policies for Epoch and Step Checkpoints

```yaml
saver:
  # Step checkpoints: keep last 3, compress and archive
  step_max_to_keep: 3
  step_cleanup_action: compress_archive

  # Epoch checkpoints: keep last 10, move to archive
  epoch_max_to_keep: 10
  epoch_cleanup_action: archive
  protect_epoch_checkpoints: false  # Apply retention to epoch checkpoints

  archive_root: /mnt/archive/boba_checkpoints
```

### Fine-tune Recovery Frequency

```yaml
recover:
  mode: auto
  freq_epochs: 1      # Recovery checkpoint each epoch
  freq_steps: 50      # Recovery checkpoint every 50 steps (more frequent)
  freq_secs: 1800     # Recovery checkpoint every 30 minutes
  retries: 5          # More retries for unstable clusters
```

## Expected Performance Impact

### Disk Space Savings

For a training run with 10 epochs × 1000 steps per epoch:

| Configuration           | Total Checkpoints      | Disk Usage (est.) |
| ----------------------- | ---------------------- | ----------------- |
| No retention            | 10,000 step + 10 epoch | ~1 TB             |
| With retention (keep 2) | 2 step + 10 epoch      | ~12 GB            |

### Training Speed

- Proximal approximation provides ~1.27x speedup vs recomputation
- Checkpoint cleanup has negligible overhead (\<1% of training time)
- Recovery overhead: ~30 seconds per checkpoint save

## Troubleshooting

### Checkpoints Not Being Cleaned Up

Check that retention is enabled:

```bash
# Verify configuration
python -c "
from areal.api.cli_args import load_expr_config, GRPOConfig
cfg, _ = load_expr_config(['examples/math/boba_grpo_vllm_with_retention.yaml'], GRPOConfig)
print(f'Retention enabled: {cfg.saver.enable_retention}')
print(f'Step max to keep: {cfg.saver.step_max_to_keep}')
"
```

### Recovery Not Working

Verify recovery mode and checkpoint paths:

```bash
# Check recovery configuration
python -c "
from areal.api.cli_args import load_expr_config, GRPOConfig
cfg, _ = load_expr_config(['examples/math/boba_grpo_vllm_with_retention.yaml'], GRPOConfig)
print(f'Recovery mode: {cfg.recover.mode}')
print(f'Recovery freq steps: {cfg.recover.freq_steps}')
"

# Check if recovery checkpoint exists
ls -la /tmp/areal/experiments/checkpoints/$(whoami)/boba_ppo_vllm/trial0/default/recover_checkpoint/
```

### Disk Space Still Growing

- Check if `enable_retention: true` is set
- Verify `step_max_to_keep` is not `null`
- Ensure checkpoint frequency creates enough checkpoints to trigger cleanup
- Check logs for retention manager errors

## Migration from Original Configuration

To migrate from `boba_grpo_vllm.yaml` to the retention-enabled version:

1. **Backup existing checkpoints** (optional):

   ```bash
   cp -r /tmp/areal/experiments/checkpoints /tmp/areal/experiments/checkpoints.backup
   ```

1. **Update configuration**:

   - Copy `boba_grpo_vllm_with_retention.yaml`
   - Adjust parameters as needed

1. **First run with retention**:

   - Existing checkpoints are automatically scanned and registered
   - Retention policies apply only to new checkpoints
   - Manually clean old checkpoints if needed:
     ```bash
     # List old checkpoints
     find /tmp/areal/experiments/checkpoints -name "epoch*step*" -type d

     # Remove old checkpoints (BE CAREFUL!)
     # find /tmp/areal/experiments/checkpoints -name "epoch*step*" -type d | head -n -2 | xargs rm -rf
     ```

## References

- [Checkpoint Retention Documentation](../../docs/checkpoint_retention.md) (if exists)
- [Recovery System Documentation](../../docs/recovery.md) (if exists)
- [Proximal Approximation Documentation](../../docs/algorithms/prox_approx.md)

## Questions or Issues?

- GitHub Issues: https://github.com/inclusionAI/AReaL/issues
- For this specific feature: Reference PR #XXX (checkpoint retention)
