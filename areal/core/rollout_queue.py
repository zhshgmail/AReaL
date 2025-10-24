"""Local rollout queue implementation.

This module contains the concrete implementation of RolloutQueue using Python queue.Queue.
"""

from __future__ import annotations

import queue
from typing import Any, Optional

from areal.api.queue_api import RolloutQueue


class LocalRolloutQueue(RolloutQueue):
    """Local implementation of RolloutQueue using Python queue.Queue.

    This is the default implementation used when no distributed
    queue is configured.

    Args:
        maxsize: Maximum number of items in queue (0 = unlimited)
    """

    def __init__(self, maxsize: int = 0):
        """Initialize local queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._queue = queue.Queue(maxsize=maxsize)

    def put(self, item: Any, timeout: Optional[float] = None) -> None:
        """Add item to queue."""
        self._queue.put(item, timeout=timeout)

    def put_nowait(self, item: Any) -> None:
        """Add item to queue without blocking."""
        self._queue.put_nowait(item)

    def get(self, timeout: Optional[float] = None) -> Any:
        """Remove and return item from queue."""
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> Any:
        """Remove and return item from queue without blocking."""
        return self._queue.get_nowait()

    def qsize(self) -> int:
        """Return approximate size of queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if queue is empty."""
        return self._queue.empty()

    @property
    def maxsize(self) -> int:
        """Maximum size of queue."""
        return self._queue.maxsize
