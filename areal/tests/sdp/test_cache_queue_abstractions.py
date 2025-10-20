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

    def test_add_and_get_all(self):
        """Test adding and getting all items."""
        cache = LocalRolloutCache()

        cache.add({"data": "value1"})
        result = cache.get_all()

        assert result == [{"data": "value1"}]

    def test_get_all_returns_copy(self):
        """Test that get_all returns a copy, not reference."""
        cache = LocalRolloutCache()

        cache.add("value1")
        result = cache.get_all()
        result.append("value2")  # Modify returned list

        assert cache.get_all() == ["value1"]  # Original unchanged

    def test_size_increases_with_adds(self):
        """Test that size increases with adds."""
        cache = LocalRolloutCache()

        assert cache.size() == 0

        cache.add("value1")
        assert cache.size() == 1

        cache.add("value2")
        assert cache.size() == 2

        cache.add("value3")
        assert cache.size() == 3

    def test_take_first_n_removes_items(self):
        """Test that take_first_n removes and returns items."""
        cache = LocalRolloutCache()

        cache.add("value1")
        cache.add("value2")
        cache.add("value3")
        assert cache.size() == 3

        result = cache.take_first_n(2)
        assert result == ["value1", "value2"]
        assert cache.size() == 1
        assert cache.get_all() == ["value3"]

    def test_take_first_n_with_fewer_items(self):
        """Test take_first_n when cache has fewer items than requested."""
        cache = LocalRolloutCache()

        cache.add("value1")
        result = cache.take_first_n(5)

        assert result == ["value1"]
        assert cache.size() == 0

    def test_take_first_n_zero(self):
        """Test take_first_n with n=0."""
        cache = LocalRolloutCache()

        cache.add("value1")
        result = cache.take_first_n(0)

        assert result == []
        assert cache.size() == 1

    def test_clear_removes_all_items(self):
        """Test that clear removes all items."""
        cache = LocalRolloutCache()

        cache.add("value1")
        cache.add("value2")
        cache.add("value3")
        assert cache.size() == 3

        cache.clear()

        assert cache.size() == 0
        assert cache.get_all() == []

    def test_filter_inplace_keeps_matching_items(self):
        """Test filter_inplace keeps only matching items."""
        cache = LocalRolloutCache()

        cache.add(1)
        cache.add(2)
        cache.add(3)
        cache.add(4)
        cache.add(5)

        # Keep only even numbers
        removed = cache.filter_inplace(lambda x: x % 2 == 0)

        assert removed == 3  # Removed 1, 3, 5
        assert cache.size() == 2
        assert cache.get_all() == [2, 4]

    def test_filter_inplace_removes_all(self):
        """Test filter_inplace when no items match."""
        cache = LocalRolloutCache()

        cache.add(1)
        cache.add(3)
        cache.add(5)

        # Keep only even numbers (none match)
        removed = cache.filter_inplace(lambda x: x % 2 == 0)

        assert removed == 3
        assert cache.size() == 0
        assert cache.get_all() == []

    def test_filter_inplace_keeps_all(self):
        """Test filter_inplace when all items match."""
        cache = LocalRolloutCache()

        cache.add(2)
        cache.add(4)
        cache.add(6)

        # Keep only even numbers (all match)
        removed = cache.filter_inplace(lambda x: x % 2 == 0)

        assert removed == 0
        assert cache.size() == 3
        assert cache.get_all() == [2, 4, 6]

    def test_multiple_items(self):
        """Test cache with multiple items."""
        cache = LocalRolloutCache()

        items = [f"value{i}" for i in range(10)]

        for item in items:
            cache.add(item)

        assert cache.size() == 10
        assert cache.get_all() == items


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
        assert q.maxsize == 10

    def test_initialization_with_default_maxsize(self):
        """Test initialization with default maxsize."""
        q = LocalRolloutQueue()

        assert q.maxsize == 0  # Unlimited

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

    def test_put_nowait(self):
        """Test put_nowait method."""
        q = LocalRolloutQueue(maxsize=10)

        q.put_nowait("item")
        result = q.get_nowait()

        assert result == "item"

    def test_get_nowait(self):
        """Test get_nowait on empty queue raises Empty."""
        q = LocalRolloutQueue(maxsize=10)

        with pytest.raises(queue.Empty):
            q.get_nowait()

    def test_empty_method(self):
        """Test empty method."""
        q = LocalRolloutQueue(maxsize=10)

        assert q.empty() is True

        q.put("item")
        assert q.empty() is False

        q.get(timeout=1)
        assert q.empty() is True

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

    def test_maxsize_limit_with_put_nowait(self):
        """Test that maxsize is enforced with put_nowait."""
        q = LocalRolloutQueue(maxsize=2)

        q.put("item1")
        q.put("item2")

        # Queue is full, put_nowait should raise Full
        with pytest.raises(queue.Full):
            q.put_nowait("item3")


class TestLocalRolloutCacheEdgeCases:
    """Test edge cases for LocalRolloutCache."""

    def test_empty_cache_size(self):
        """Test size of empty cache."""
        cache = LocalRolloutCache()

        assert cache.size() == 0

    def test_cache_with_none_values(self):
        """Test cache with None values."""
        cache = LocalRolloutCache()

        cache.add(None)

        assert cache.get_all() == [None]
        assert cache.size() == 1  # None is a valid value

    def test_cache_with_complex_values(self):
        """Test cache with complex data structures."""
        cache = LocalRolloutCache()

        complex_value = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (4, 5, 6),
        }

        cache.add(complex_value)
        result = cache.get_all()[0]

        assert result == complex_value

    def test_take_all_items(self):
        """Test taking all items from cache."""
        cache = LocalRolloutCache()

        cache.add("value1")
        cache.add("value2")
        cache.add("value3")

        result = cache.take_first_n(3)

        assert result == ["value1", "value2", "value3"]
        assert cache.size() == 0

    def test_filter_with_complex_predicate(self):
        """Test filter with complex predicate function."""
        cache = LocalRolloutCache()

        cache.add({"score": 10})
        cache.add({"score": 5})
        cache.add({"score": 15})
        cache.add({"score": 3})

        # Keep only items with score >= 10
        removed = cache.filter_inplace(lambda x: x["score"] >= 10)

        assert removed == 2
        assert cache.size() == 2
        assert cache.get_all() == [{"score": 10}, {"score": 15}]


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
        cache.add(f"value{i}")

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
        q.put_nowait("extra")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=areal.api.cache_api", "--cov=areal.api.queue_api"])
