"""Queue API base class for async task execution.

This module defines the Queue interface that AsyncTaskRunner uses.
Concrete implementations can use different backends (local queue.Queue,
ZeroMQ, Redis, etc.) without changing the TaskRunner code.
"""

from __future__ import annotations

import abc
from typing import Any, Generic, TypeVar

# Type variable for generic queue item types
T = TypeVar("T")


class QueueAPI(abc.ABC, Generic[T]):
    """Abstract base class defining the Queue interface for AsyncTaskRunner.

    This interface abstracts queue operations, allowing different
    implementations (local, distributed, persistent, etc.) without
    changing client code.

    Implementations should handle:
    - Thread-safe operations
    - Capacity management
    - Filter-based admission control (if applicable)

    Examples of concrete implementations:
    - LocalQueue: Uses queue.Queue with local filters
    - ZeroMQQueue: Distributed queue using ZeroMQ
    - RedisQueue: Persistent queue using Redis
    - EtcdQueue: Distributed queue using Etcd
    """

    @abc.abstractmethod
    def put(self, item: T, block: bool = True, timeout: float | None = None) -> None:
        """Put an item into the queue.

        Parameters
        ----------
        item : T
            Item to put into queue
        block : bool, optional
            Whether to block if queue is full. Default is True.
        timeout : float | None, optional
            Timeout in seconds for blocking put. Default is None (wait forever).

        Raises
        ------
        queue.Full
            If queue is full and operation times out
        """
        pass

    @abc.abstractmethod
    def put_nowait(self, item: T) -> None:
        """Put an item into the queue without blocking.

        Parameters
        ----------
        item : T
            Item to put into queue

        Raises
        ------
        queue.Full
            If queue is full
        """
        pass

    @abc.abstractmethod
    def get(self, block: bool = True, timeout: float | None = None) -> T:
        """Remove and return an item from the queue.

        Parameters
        ----------
        block : bool, optional
            Whether to block if queue is empty. Default is True.
        timeout : float | None, optional
            Timeout in seconds for blocking get. Default is None (wait forever).

        Returns
        -------
        T
            Item from queue

        Raises
        ------
        queue.Empty
            If queue is empty and operation times out
        """
        pass

    @abc.abstractmethod
    def get_nowait(self) -> T:
        """Remove and return an item from the queue without blocking.

        Returns
        -------
        T
            Item from queue

        Raises
        ------
        queue.Empty
            If queue is empty
        """
        pass

    @abc.abstractmethod
    def qsize(self) -> int:
        """Return the approximate size of the queue.

        Returns
        -------
        int
            Number of items in queue
        """
        pass

    @abc.abstractmethod
    def empty(self) -> bool:
        """Return True if the queue is empty.

        Returns
        -------
        bool
            Whether queue is empty
        """
        pass

    @abc.abstractmethod
    def full(self) -> bool:
        """Return True if the queue is full.

        Returns
        -------
        bool
            Whether queue is full
        """
        pass
