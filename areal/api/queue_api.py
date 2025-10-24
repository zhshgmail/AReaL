"""Queue abstraction for rollout workflow.

This module provides abstract queue interface that can be implemented
using local Python queue, distributed queue (Ray, Redis, Kafka), etc.
"""

from __future__ import annotations

import queue
from abc import ABC, abstractmethod
from typing import Any, Optional


class RolloutQueue(ABC):
    """Abstract queue interface for rollout outputs.

    This abstraction allows the queue implementation to be swapped
    (e.g., local queue.Queue, Ray queue, Redis queue, etc.) without
    changing the WorkflowExecutor code.
    """

    @abstractmethod
    def put(self, item: Any, timeout: Optional[float] = None) -> None:
        """Add item to queue.

        Args:
            item: Item to add to queue
            timeout: Optional timeout in seconds

        Raises:
            queue.Full: If queue is full and timeout expires
        """
        pass

    @abstractmethod
    def put_nowait(self, item: Any) -> None:
        """Add item to queue without blocking.

        Args:
            item: Item to add to queue

        Raises:
            queue.Full: If queue is full
        """
        pass

    @abstractmethod
    def get(self, timeout: Optional[float] = None) -> Any:
        """Remove and return item from queue.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Item from queue

        Raises:
            queue.Empty: If queue is empty and timeout expires
        """
        pass

    @abstractmethod
    def get_nowait(self) -> Any:
        """Remove and return item from queue without blocking.

        Returns:
            Item from queue

        Raises:
            queue.Empty: If queue is empty
        """
        pass

    @abstractmethod
    def qsize(self) -> int:
        """Return approximate size of queue.

        Returns:
            Number of items in queue
        """
        pass

    @abstractmethod
    def empty(self) -> bool:
        """Return True if queue is empty.

        Returns:
            True if queue is empty, False otherwise
        """
        pass

    @property
    @abstractmethod
    def maxsize(self) -> int:
        """Maximum size of queue.

        Returns:
            Maximum number of items queue can hold (0 = unlimited)
        """
        pass


# Future implementations:
# class RayRolloutQueue(RolloutQueue):
#     """Distributed queue implementation using Ray."""
#     pass
#
# class RedisRolloutQueue(RolloutQueue):
#     """Distributed queue implementation using Redis."""
#     pass
