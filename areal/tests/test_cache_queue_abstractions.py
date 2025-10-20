"""Unit tests for RolloutCache and RolloutQueue abstractions.

This module provides comprehensive test coverage for the abstract interfaces
and concrete implementations of RolloutCache and RolloutQueue.
All tests run without GPU.
"""

import queue
import time

import pytest

from areal.api.cache_api import LocalRolloutCache, RolloutCache
from areal.api.queue_api import LocalRolloutQueue, RolloutQueue


class TestRolloutCacheInterface:
    """Test RolloutCache abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that RolloutCache cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RolloutCache()


class TestLocalRolloutCacheBasics:
    """Test basic functionality of LocalRolloutCache."""

    def test_initialization(self):
        """Test LocalRolloutCache initialization."""
        cache = LocalRolloutCache()

        assert isinstance(cache, RolloutCache)
        assert cache.size() == 0

    def test_put_and_get(self):
        """Test putting and getting items."""
        cache = LocalRolloutCache()

        cache.put("key1", {"data": "value1"})
        result = cache.get("key1")

        assert result == {"data": "value1"}

    def test_get_nonexistent_key_returns_none(self):
        """Test that getting nonexistent key returns None."""
        cache = LocalRolloutCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_size_increases_with_puts(self):
        """Test that size increases with puts."""
        cache = LocalRolloutCache()

        assert cache.size() == 0

        cache.put("key1", "value1")
        assert cache.size() == 1

        cache.put("key2", "value2")
        assert cache.size() == 2

        cache.put("key3", "value3")
        assert cache.size() == 3

    def test_put_overwrites_existing_key(self):
        """Test that putting same key overwrites value but size stays same."""
        cache = LocalRolloutCache()

        cache.put("key1", "value1")
        assert cache.size() == 1

        cache.put("key1", "value2")
        assert cache.size() == 1  # Size should not increase
        assert cache.get("key1") == "value2"  # Value should be updated

    def test_pop_removes_and_returns_item(self):
        """Test that pop removes item from cache."""
        cache = LocalRolloutCache()

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert cache.size() == 2

        result = cache.pop("key1")
        assert result == "value1"
        assert cache.size() == 1
        assert cache.get("key1") is None  # Should be removed

    def test_pop_nonexistent_key_returns_none(self):
        """Test that popping nonexistent key returns None."""
        cache = LocalRolloutCache()

        result = cache.pop("nonexistent")

        assert result is None

    def test_pop_with_default(self):
        """Test pop with default value."""
        cache = LocalRolloutCache()

        result = cache.pop("nonexistent", "default_value")

        assert result == "default_value"

    def test_multiple_items(self):
        """Test cache with multiple items."""
        cache = LocalRolloutCache()

        items = {f"key{i}": f"value{i}" for i in range(10)}

        for key, value in items.items():
            cache.put(key, value)

        assert cache.size() == 10

        for key, expected_value in items.items():
            assert cache.get(key) == expected_value


class TestRolloutQueueInterface:
    """Test RolloutQueue abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that RolloutQueue cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RolloutQueue()


class TestLocalRolloutQueueBasics:
    """Test basic functionality of LocalRolloutQueue."""

    def test_initialization(self):
        """Test LocalRolloutQueue initialization."""
        q = LocalRolloutQueue(maxsize=10)

        assert isinstance(q, RolloutQueue)

    def test_initialization_with_default_maxsize(self):
        """Test initialization with default maxsize."""
        q = LocalRolloutQueue()

        # Should work without maxsize
        q.put({"item": 1})
        result = q.get(timeout=1)
        assert result == {"item": 1}

    def test_put_and_get(self):
        """Test putting and getting items."""
        q = LocalRolloutQueue(maxsize=10)

        q.put({"data": "value1"})
        result = q.get(timeout=1)

        assert result == {"data": "value1"}

    def test_get_blocks_until_item_available(self):
        """Test that get blocks until item is available."""
        import threading

        q = LocalRolloutQueue(maxsize=10)

        result_container = []

        def getter():
            result = q.get(timeout=2)
            result_container.append(result)

        thread = threading.Thread(target=getter)
        thread.start()

        # Wait a bit, then put item
        time.sleep(0.1)
        q.put("item")

        thread.join()

        assert result_container == ["item"]

    def test_get_timeout(self):
        """Test that get times out if no item available."""
        q = LocalRolloutQueue(maxsize=10)

        with pytest.raises(queue.Empty):
            q.get(timeout=0.1)

    def test_get_without_timeout(self):
        """Test get without timeout parameter."""
        q = LocalRolloutQueue(maxsize=10)

        q.put("item")
        result = q.get()  # No timeout

        assert result == "item"

    def test_qsize(self):
        """Test qsize method."""
        q = LocalRolloutQueue(maxsize=10)

        assert q.qsize() == 0

        q.put("item1")
        assert q.qsize() == 1

        q.put("item2")
        assert q.qsize() == 2

        q.get(timeout=1)
        assert q.qsize() == 1

    def test_fifo_order(self):
        """Test that queue maintains FIFO order."""
        q = LocalRolloutQueue(maxsize=10)

        items = ["first", "second", "third", "fourth"]

        for item in items:
            q.put(item)

        results = []
        for _ in range(len(items)):
            results.append(q.get(timeout=1))

        assert results == items

    def test_maxsize_limit(self):
        """Test that maxsize is enforced (queue blocks when full)."""
        q = LocalRolloutQueue(maxsize=2)

        q.put("item1")
        q.put("item2")

        # Queue is full, next put should block
        # We test with put_nowait which raises Full
        with pytest.raises(queue.Full):
            q.put("item3", block=False)


class TestLocalRolloutCacheEdgeCases:
    """Test edge cases for LocalRolloutCache."""

    def test_empty_cache_size(self):
        """Test size of empty cache."""
        cache = LocalRolloutCache()

        assert cache.size() == 0

    def test_cache_with_none_values(self):
        """Test cache with None values."""
        cache = LocalRolloutCache()

        cache.put("key1", None)

        assert cache.get("key1") is None
        assert cache.size() == 1  # None is a valid value

    def test_cache_with_complex_values(self):
        """Test cache with complex data structures."""
        cache = LocalRolloutCache()

        complex_value = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (4, 5, 6),
        }

        cache.put("key1", complex_value)
        result = cache.get("key1")

        assert result == complex_value

    def test_pop_all_items(self):
        """Test popping all items from cache."""
        cache = LocalRolloutCache()

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        cache.pop("key1")
        cache.pop("key2")
        cache.pop("key3")

        assert cache.size() == 0


class TestLocalRolloutQueueEdgeCases:
    """Test edge cases for LocalRolloutQueue."""

    def test_empty_queue_qsize(self):
        """Test qsize of empty queue."""
        q = LocalRolloutQueue(maxsize=10)

        assert q.qsize() == 0

    def test_queue_with_none_values(self):
        """Test queue with None values."""
        q = LocalRolloutQueue(maxsize=10)

        q.put(None)
        result = q.get(timeout=1)

        assert result is None

    def test_queue_with_complex_values(self):
        """Test queue with complex data structures."""
        q = LocalRolloutQueue(maxsize=10)

        complex_value = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        q.put(complex_value)
        result = q.get(timeout=1)

        assert result == complex_value

    def test_queue_stress_test(self):
        """Test queue with many items."""
        q = LocalRolloutQueue(maxsize=1000)

        num_items = 100

        for i in range(num_items):
            q.put(f"item{i}")

        for i in range(num_items):
            result = q.get(timeout=1)
            assert result == f"item{i}"

        assert q.qsize() == 0


# Parametrized tests
@pytest.mark.parametrize("num_items", [1, 5, 10, 50, 100])
def test_parametrized_cache_size(num_items):
    """Test cache with various numbers of items."""
    cache = LocalRolloutCache()

    for i in range(num_items):
        cache.put(f"key{i}", f"value{i}")

    assert cache.size() == num_items


@pytest.mark.parametrize("maxsize", [1, 5, 10, 100])
def test_parametrized_queue_maxsize(maxsize):
    """Test queue with various maxsize values."""
    q = LocalRolloutQueue(maxsize=maxsize)

    # Fill queue to maxsize
    for i in range(maxsize):
        q.put(f"item{i}")

    # Queue should be full
    with pytest.raises(queue.Full):
        q.put("extra", block=False)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=areal.api.cache_api", "--cov=areal.api.queue_api"])
