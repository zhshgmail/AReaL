"""Queue/cache transformers for segment-wise decoupled PPO.

This module provides composable transformers for processing rollout samples:
- ProximalRecomputer: Update proximal_t for v-1 samples
- StalenessFilter: Remove over-stale samples
"""

from .proximal_recomputer import RECOMPUTE_VERSION_KEY, ProximalRecomputer, ensure_recompute_key
from .staleness_filter import StalenessFilter

__all__ = [
    "ProximalRecomputer",
    "StalenessFilter",
    "RECOMPUTE_VERSION_KEY",
    "ensure_recompute_key",
]
