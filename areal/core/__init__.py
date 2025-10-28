"""Core components for AREAL."""

from areal.api.event_api import EventContext, EventHandler, EventType
from areal.api.filter_api import Filter

from .event_system import EventRegistry
from .filters import StalenessFilter
from .handlers import (
    RECOMPUTE_VERSION_KEY,
    ProximalRecomputer,
    ensure_recompute_key,
)
from .remote_inf_engine import (
    RemoteInfBackendProtocol,
    RemoteInfEngine,
)
from .staleness_manager import StalenessManager
from .workflow_executor import (
    WorkflowExecutor,
    check_trajectory_format,
)
from .workflow_factory import (
    create_workflow_executor,
    create_workflow_executor_with_events,
)

__all__ = [
    "RemoteInfBackendProtocol",
    "RemoteInfEngine",
    "StalenessManager",
    "WorkflowExecutor",
    "check_trajectory_format",
    "create_workflow_executor",
    "create_workflow_executor_with_events",
    # Event-driven architecture
    "EventType",
    "EventContext",
    "EventRegistry",
    "Filter",  # Renamed from QueueFilter
    "EventHandler",
    # Filters and handlers
    "StalenessFilter",
    "ProximalRecomputer",
    "RECOMPUTE_VERSION_KEY",
    "ensure_recompute_key",
]
