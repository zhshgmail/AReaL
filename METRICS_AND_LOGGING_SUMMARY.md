# Segment-Wise PPO Metrics and Debug Logging Summary

This document describes the comprehensive metrics and debug logging added to the segment-wise PPO feature for improved observability and debugging.

## Overview

Added metrics tracking and debug logging at all critical points in the segment-wise PPO algorithm:

1. **Staleness Filtering** (workflow_executor.py) - Track samples filtered before enqueue
2. **Capacity Modification** (filtered_capacity_modifier.py) - Track capacity adjustments
3. **Staleness Manager** (staleness_manager.py) - Track rollout capacity and version changes
4. **Staleness Strategy** (staleness_control.py) - Track staleness check decisions

## Files Modified

### 1. areal/core/workflow_executor.py

**Metrics Added:**
- `_metrics_filtered_total`: Total samples filtered due to staleness
- `_metrics_accepted_total`: Total samples accepted after staleness check
- `_metrics_rejected_total`: Total samples rejected

**Logging Added:**

#### DEBUG Level - Per-Sample Filtering (lines 473-491)
```python
self.logger.debug(
    f"[StalenessFilter] Sample accepted: version={sample_version}, "
    f"current_ver={current_ver}, staleness={current_ver - sample_version}, "
    f"total_accepted={self._metrics_accepted_total}"
)

self.logger.debug(
    f"[StalenessFilter] Sample filtered due to staleness: version={sample_version}, "
    f"current_ver={current_ver}, staleness={current_ver - sample_version}, "
    f"total_filtered={self._metrics_filtered_total}"
)
```

**When:** Before enqueueing each rollout result
**What:** Logs whether sample was accepted or filtered, with version and staleness info

#### INFO Level - Batch Summary (lines 697-703)
```python
self.logger.info(
    f"[StalenessMetrics] Total samples: "
    f"filtered={self._metrics_filtered_total}, "
    f"accepted={self._metrics_accepted_total}, "
    f"filter_rate={...:.2%}"
)
```

**When:** At end of `wait()` when results are ready
**What:** Summary of filtering effectiveness (filter rate)

### 2. areal/core/filtered_capacity_modifier.py

**Changes:**
- Added optional `logger` parameter to `__init__`
- Logger passed from factory

**Logging Added:**

#### DEBUG Level - Filtered Count Updates (lines 74-79)
```python
self.logger.debug(
    f"[CapacityModifier] Filtered samples count updated: "
    f"added={count}, total={self._filtered_count} (was {old_count}), "
    f"version={version}"
)
```

**When:** Called when samples are filtered (`on_samples_filtered`)
**What:** Tracks accumulation of filtered samples that need capacity compensation

#### DEBUG Level - Capacity Adjustments (lines 118-123)
```python
self.logger.debug(
    f"[CapacityModifier] Capacity adjusted: "
    f"base={base_capacity}, filtered_count={self._filtered_count}, "
    f"adjusted={modified_capacity}, version={current_version}"
)
```

**When:** When modifying capacity calculation
**What:** Shows how many extra rollouts are allowed due to filtered samples

### 3. areal/core/staleness_manager.py

**Changes:**
- Added optional `logger` parameter to `__init__`
- Added `_last_logged_version` to track when to log

**Logging Added:**

#### INFO Level - Version Change and Capacity (lines 125-134)
```python
self.logger.info(
    f"[StalenessManager] Version {current_version}: "
    f"capacity={modified_capacity} (base={base_capacity}, "
    f"concurrency={concurrency_capacity}, staleness={staleness_capacity}), "
    f"stats(submitted={...}, running={...}, accepted={...})"
)
```

**When:** When model version changes
**What:** Shows capacity breakdown and rollout statistics for the new version

### 4. areal/api/staleness_control.py

**Changes:**
- Added optional `logger` parameter to `SegmentWisePPOStrategy.__init__`
- Added metrics: `_total_checks`, `_stale_count`

**Logging Added:**

#### DEBUG Level - Periodic Staleness Checks (lines 244-250)
```python
# Every 100 checks
self.logger.debug(
    f"[StalenessCheck] Checked {self._total_checks} samples: "
    f"stale={self._stale_count}, "
    f"fresh={self._total_checks - self._stale_count}, "
    f"stale_rate={self._stale_count / self._total_checks:.2%}"
)
```

**When:** Every 100 staleness checks
**What:** Summary of staleness check decisions (stale rate)

### 5. areal/api/workflow_factory.py

**Changes:**
- Updated factories to accept and pass `logger` parameter:
  - `create_staleness_strategy(config, logger)`
  - `create_filtered_capacity_modifier(config, logger)`
- Call sites updated to pass logger

## Logging Hierarchy

```
[StalenessManager] INFO - Version changes, capacity breakdown
    |
    +-- [StalenessFilter] DEBUG - Per-sample accept/reject decisions
    |       |
    |       +-- [StalenessMetrics] INFO - Batch filtering summary
    |
    +-- [CapacityModifier] DEBUG - Filtered count updates
    |       |
    |       +-- [CapacityModifier] DEBUG - Capacity adjustments
    |
    +-- [StalenessCheck] DEBUG - Periodic staleness check summary (every 100)
```

## Usage

### Enable Debug Logging

To see all debug logs, set log level to DEBUG:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Typical Log Output (INFO level)

```
[StalenessManager] Version 0: capacity=10 (base=10, concurrency=10, staleness=12), stats(submitted=0, running=0, accepted=0)
[QueuePurge] ver_switch to 1: drained=5 picked_prev=3 dropped=2 kept=3 cache_size=10 v0(picked/dropped/kept)=1/0/2
[CacheFilter] dropped_cache=2 size=8
[StalenessMetrics] Total samples: filtered=15, accepted=85, filter_rate=15.00%
```

### Typical Debug Log Output

```
[StalenessFilter] Sample accepted: version=0, current_ver=0, staleness=0, total_accepted=1
[StalenessFilter] Sample filtered due to staleness: version=0, current_ver=2, staleness=2, total_filtered=1
[CapacityModifier] Filtered samples count updated: added=1, total=1 (was 0), version=2
[CapacityModifier] Capacity adjusted: base=5, filtered_count=1, adjusted=6, version=2
[StalenessCheck] Checked 100 samples: stale=15, fresh=85, stale_rate=15.00%
```

## Benefits

1. **Debugging**: Quickly identify why samples are being filtered
2. **Performance**: Track filtering effectiveness and capacity adjustments
3. **Observability**: Monitor system behavior at version boundaries
4. **Troubleshooting**: Diagnose deadlocks or unexpected behavior

## Log Tag Reference

| Tag | Level | Location | Purpose |
|-----|-------|----------|---------|
| `[StalenessManager]` | INFO | staleness_manager.py | Version changes, capacity |
| `[StalenessFilter]` | DEBUG | workflow_executor.py | Per-sample filtering decisions |
| `[StalenessMetrics]` | INFO | workflow_executor.py | Filtering summary |
| `[CapacityModifier]` | DEBUG | filtered_capacity_modifier.py | Capacity adjustments |
| `[StalenessCheck]` | DEBUG | staleness_control.py | Staleness check summary |
| `[QueuePurge]` | INFO | staleness_control.py | Queue purging results |
| `[CacheFilter]` | INFO | staleness_control.py | Cache filtering results |

## Testing

All existing tests pass with these changes:
- `areal/tests/sdp/test_filtered_capacity_modifier.py` - 32 tests ✓
- `areal/tests/sdp/test_segment_wise_ppo_config.py` - 24 tests ✓

The logger parameter is optional and defaults to None, maintaining backward compatibility.
