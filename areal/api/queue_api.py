"""Queue API protocol for async task execution.

This module defines the Queue interface that AsyncTaskRunner uses.
Concrete implementations can use different backends (local queue.Queue,
ZeroMQ, Redis, etc.) without changing the TaskRunner code.
"""

from __future__ import annotations

from typing import Any, Protocol


class QueueAPI(Protocol):
    """Protocol defining the Queue interface for AsyncTaskRunner.

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

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        """Put an item into the queue.

        Parameters
        ----------
        item : Any
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
        ...

    def put_nowait(self, item: Any) -> None:
        """Put an item into the queue without blocking.

        Parameters
        ----------
        item : Any
            Item to put into queue

        Raises
        ------
        queue.Full
            If queue is full
        """
        ...

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Remove and return an item from the queue.

        Parameters
        ----------
        block : bool, optional
            Whether to block if queue is empty. Default is True.
        timeout : float | None, optional
            Timeout in seconds for blocking get. Default is None (wait forever).

        Returns
        -------
        Any
            Item from queue

        Raises
        ------
        queue.Empty
            If queue is empty and operation times out
        """
        ...

    def get_nowait(self) -> Any:
        """Remove and return an item from the queue without blocking.

        Returns
        -------
        Any
            Item from queue

        Raises
        ------
        queue.Empty
            If queue is empty
        """
        ...

    def qsize(self) -> int:
        """Return the approximate size of the queue.

        Returns
        -------
        int
            Number of items in queue
        """
        ...

    def empty(self) -> bool:
        """Return True if the queue is empty.

        Returns
        -------
        bool
            Whether queue is empty
        """
        ...

    def full(self) -> bool:
        """Return True if the queue is full.

        Returns
        -------
        bool
            Whether queue is full
        """
        ...
