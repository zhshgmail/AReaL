"""Integration tests for event-driven architecture.

Tests the complete event system including:
- Filter registration and admission control
- PRE_UPDATE and POST_UPDATE event firing
- ProximalRecomputer handler execution
- Queue/cache access via context.data
"""

import pytest
from unittest.mock import Mock, MagicMock
from areal.core.event_system import (
    EventContext,
    EventRegistry,
    EventType,
    QueueFilter,
    EventHandler,
)
from areal.core.async_task_runner import AsyncTaskRunner


class TestFilterAdmissionControl:
    """Test filter registration and admission control in AsyncTaskRunner."""

    def test_filter_rejects_item(self):
        """Test that filter can reject items at output queue admission."""

        # Create a filter that rejects items with value > 5
        class SimpleFilter:
            def should_accept(self, item, context):
                return item["value"] <= 5

        # Create runner with filter
        runner = AsyncTaskRunner[dict](max_queue_size=10)
        runner.register_add_filter(SimpleFilter())

        # Create mock context
        mock_context = Mock()
        runner.set_filter_context(mock_context)

        runner.initialize()

        # Submit tasks
        async def make_item(value):
            return {"value": value}

        # Submit items with values 1-10
        for i in range(1, 11):
            runner.submit(make_item, i)

        # Wait for results - should only get items with value <= 5
        results = runner.wait(count=5, timeout=5.0)
        runner.destroy()

        # Check that all results have value <= 5
        assert len(results) == 5
        for result in results:
            assert result["value"] <= 5

    def test_multiple_filters(self):
        """Test that multiple filters can be registered and all are checked."""

        class MinFilter:
            def should_accept(self, item, context):
                return item["value"] >= 3

        class MaxFilter:
            def should_accept(self, item, context):
                return item["value"] <= 7

        runner = AsyncTaskRunner[dict](max_queue_size=20)
        runner.register_add_filter(MinFilter())
        runner.register_add_filter(MaxFilter())

        mock_context = Mock()
        runner.set_filter_context(mock_context)
        runner.initialize()

        async def make_item(value):
            return {"value": value}

        # Submit items 1-10
        for i in range(1, 11):
            runner.submit(make_item, i)

        # Should only get items in range [3, 7]
        results = runner.wait(count=5, timeout=5.0)
        runner.destroy()

        assert len(results) == 5
        for result in results:
            assert 3 <= result["value"] <= 7

    def test_no_filter_accepts_all(self):
        """Test that without filters, all items are accepted."""
        runner = AsyncTaskRunner[dict](max_queue_size=10)
        runner.initialize()

        async def make_item(value):
            return {"value": value}

        for i in range(1, 6):
            runner.submit(make_item, i)

        results = runner.wait(count=5, timeout=5.0)
        runner.destroy()

        assert len(results) == 5


class TestEventFiring:
    """Test event registry and handler execution."""

    def test_handler_called_on_event(self):
        """Test that registered handlers are called when events fire."""
        registry = EventRegistry()
        handler_called = {"count": 0}

        class TestHandler:
            def on_event(self, context):
                handler_called["count"] += 1
                assert context.event_type == EventType.BEFORE_POLICY_UPDATE

        handler = TestHandler()
        registry.register_handler(EventType.BEFORE_POLICY_UPDATE, handler)

        # Fire event
        mock_engine = Mock()
        mock_config = Mock()
        mock_logger = Mock()

        context = EventContext(
            EventType.BEFORE_POLICY_UPDATE,
            mock_engine,
            mock_config,
            mock_logger,
            data={"test": "data"},
        )

        registry.fire_event(context)

        assert handler_called["count"] == 1

    def test_multiple_handlers_for_same_event(self):
        """Test that multiple handlers can be registered for the same event."""
        registry = EventRegistry()
        call_order = []

        class Handler1:
            def on_event(self, context):
                call_order.append("handler1")

        class Handler2:
            def on_event(self, context):
                call_order.append("handler2")

        registry.register_handler(EventType.BEFORE_POLICY_UPDATE, Handler1())
        registry.register_handler(EventType.BEFORE_POLICY_UPDATE, Handler2())

        context = EventContext(
            EventType.BEFORE_POLICY_UPDATE, Mock(), Mock(), Mock()
        )
        registry.fire_event(context)

        assert call_order == ["handler1", "handler2"]

    def test_handler_accesses_queue_via_context(self):
        """Test that handlers can access queue/cache via context.data."""
        registry = EventRegistry()
        accessed_data = {"queue": None, "cache": None}

        class DataAccessHandler:
            def on_event(self, context):
                accessed_data["queue"] = context.data.get("queue")
                accessed_data["cache"] = context.data.get("cache")

        registry.register_handler(
            EventType.BEFORE_POLICY_UPDATE, DataAccessHandler()
        )

        mock_queue = Mock()
        mock_cache = [{"item": 1}, {"item": 2}]

        context = EventContext(
            EventType.BEFORE_POLICY_UPDATE,
            Mock(),
            Mock(),
            Mock(),
            data={"queue": mock_queue, "cache": mock_cache},
        )

        registry.fire_event(context)

        assert accessed_data["queue"] is mock_queue
        assert accessed_data["cache"] == mock_cache


class TestEventTypes:
    """Test different event types."""

    def test_pre_and_post_update_events(self):
        """Test that PRE and POST update events are distinct."""
        registry = EventRegistry()
        events_fired = []

        class EventLogger:
            def on_event(self, context):
                events_fired.append(context.event_type)

        logger = EventLogger()
        registry.register_handler(EventType.BEFORE_POLICY_UPDATE, logger)
        registry.register_handler(EventType.AFTER_POLICY_UPDATE, logger)

        # Fire PRE_UPDATE
        context_pre = EventContext(
            EventType.BEFORE_POLICY_UPDATE, Mock(), Mock(), Mock()
        )
        registry.fire_event(context_pre)

        # Fire POST_UPDATE
        context_post = EventContext(
            EventType.AFTER_POLICY_UPDATE, Mock(), Mock(), Mock()
        )
        registry.fire_event(context_post)

        assert len(events_fired) == 2
        assert events_fired[0] == EventType.BEFORE_POLICY_UPDATE
        assert events_fired[1] == EventType.AFTER_POLICY_UPDATE


class TestIntegrationScenario:
    """Test realistic integration scenarios."""

    def test_filter_and_event_together(self):
        """Test that filters and events work together in a realistic scenario.

        Simulates:
        1. Samples are generated and filtered at admission
        2. PRE_UPDATE event fires before policy update
        3. Handler accesses queue/cache during event
        4. POST_UPDATE event fires after update
        """
        # Setup runner with filter
        runner = AsyncTaskRunner[dict](max_queue_size=20)

        class StalenessFilter:
            def should_accept(self, item, context):
                # Reject items with staleness > 2
                return item.get("staleness", 0) <= 2

        runner.register_add_filter(StalenessFilter())

        mock_context = Mock()
        runner.set_filter_context(mock_context)
        runner.initialize()

        # Submit some items
        async def make_sample(staleness):
            return {"staleness": staleness, "data": "sample"}

        for s in [0, 1, 2, 3, 4]:  # Mix of stale and fresh samples
            runner.submit(make_sample, s)

        # Wait for results - should only get staleness <= 2
        results = runner.wait(count=3, timeout=5.0)

        # Setup event registry
        registry = EventRegistry()
        handler_data = {"queue_accessed": False, "cache_accessed": False}

        class RecomputeHandler:
            def on_event(self, context):
                if context.event_type == EventType.BEFORE_POLICY_UPDATE:
                    # Access queue and cache
                    queue = context.data.get("queue")
                    cache = context.data.get("cache")
                    handler_data["queue_accessed"] = queue is not None
                    handler_data["cache_accessed"] = cache is not None

        registry.register_handler(
            EventType.BEFORE_POLICY_UPDATE, RecomputeHandler()
        )

        # Fire PRE_UPDATE event
        context = EventContext(
            EventType.BEFORE_POLICY_UPDATE,
            Mock(),
            Mock(),
            Mock(),
            data={
                "queue": runner.output_queue,
                "cache": results,
                "old_version": 0,
            },
        )
        registry.fire_event(context)

        runner.destroy()

        # Verify results
        assert len(results) == 3
        for result in results:
            assert result["staleness"] <= 2

        # Verify handler accessed queue/cache
        assert handler_data["queue_accessed"]
        assert handler_data["cache_accessed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
