"""Event handlers for system events."""

from .cache_proximal_recomputer import CacheProximalRecomputer
from .event_propagator import EventPropagator
from .proximal_recompute_logic import (
    RECOMPUTE_VERSION_KEY,
    ProximalRecomputeLogic,
    ensure_recompute_key,
)
from .queue_proximal_recomputer import QueueProximalRecomputer

# Keep old ProximalRecomputer for backward compatibility (deprecated)
from .proximal_recomputer import ProximalRecomputer

__all__ = [
    "EventPropagator",
    "QueueProximalRecomputer",
    "CacheProximalRecomputer",
    "ProximalRecomputeLogic",
    "RECOMPUTE_VERSION_KEY",
    "ensure_recompute_key",
    # Deprecated
    "ProximalRecomputer",
]
