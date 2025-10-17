"""
Unit tests for WorkflowExecutor helper functions.

These tests verify each helper function independently,
demonstrating the improved testability from the refactoring.
"""

import queue
import sys
from typing import List
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

# Mock uvloop for Windows compatibility
if sys.platform == 'win32':
    sys.modules['uvloop'] = MagicMock()

# Mock megatron to avoid import dependencies
sys.modules['megatron'] = MagicMock()
sys.modules['megatron.core'] = MagicMock()
sys.modules['megatron.core.parallel_state'] = MagicMock()

from areal.api.workflow_api import RECOMPUTE_VERSION_KEY, WorkflowExecutor


# Mock InferenceEngineConfig
class InferenceEngineConfig:
    def __init__(self):
        self.max_concurrent_rollouts = 4
        self.consumer_batch_size = 2
        self.queue_size = 16
        self.max_head_offpolicyness = 2
        self.enable_rollout_tracing = False
        self.request_timeout = 30


class MockInferenceEngine:
    """Mock inference engine for testing."""

    def __init__(self, version=0):
        self._version = version

    def get_version(self):
        return self._version

    def recompute_output_logprobs_sync(self, input_ids, start_index):
        """Mock recompute that returns fake logprobs."""
        return [0.5] * (len(input_ids) - start_index - 1)


def create_sample(
    seq_len: int = 10,
    versions: List[int] = None,
    loss_mask: List[int] = None,
    recompute_version: int = -1,
) -> TensorDict:
    """Create a mock TensorDict sample for testing."""
    if versions is None:
        versions = [0] * seq_len
    if loss_mask is None:
        loss_mask = [1] * seq_len

    td = TensorDict(
        {
            "input_ids": torch.tensor([list(range(seq_len))]),
            "versions": torch.tensor([versions]),
            "loss_mask": torch.tensor([loss_mask]),
            "proximal_logprobs_t": torch.randn(1, seq_len),
            "attention_mask": torch.ones(1, seq_len),
        },
        batch_size=[1],
    )

    if recompute_version >= 0:
        td.set(RECOMPUTE_VERSION_KEY, torch.tensor([[recompute_version]]))

    return td


@pytest.fixture
def config():
    """Create test configuration for standard PPO."""
    cfg = InferenceEngineConfig()
    # Standard PPO (no staleness control)
    return cfg


@pytest.fixture
def config_sdp():
    """Create test configuration with segment-wise PPO enabled."""
    cfg = InferenceEngineConfig()
    cfg.enable_segment_wise_ppo = True  # Enable segment-wise PPO features
    return cfg


@pytest.fixture
def mock_engine():
    """Create mock inference engine."""
    return MockInferenceEngine(version=0)


@pytest.fixture
def executor(config_sdp, mock_engine):
    """Create WorkflowExecutor for testing with SDP enabled (to access strategies)."""
    from unittest.mock import Mock
    from areal.api.workflow_api import create_workflow_executor

    # Use factory to create executor with proper components
    executor = create_workflow_executor(config_sdp, mock_engine)
    executor.rollout_tasks = {}
    # Mock logger to prevent AttributeError
    executor.logger = Mock()
    executor.dp_world_size = 1
    return executor


class TestCalculateStaleness:
    """Test calculate_staleness helper function (now in strategy)."""

    def test_tail_staleness_for_normal_sample(self, executor):
        """Test tail staleness calculation for non-recomputed samples."""
        # Access strategy method
        staleness, allow, max_v = executor.staleness_strategy.calculate_staleness(
            versions=[0, 1, 2, 3, 4],
            loss_mask=[0, 1, 1, 1, 1],  # First is input, rest are output
            current_ver=7,
            recompute_version=-1,  # Not recomputed
            config=executor.config
        )

        # tail_staleness = current_ver - max_version = 7 - 4 = 3
        assert staleness == 3
        assert allow == 1  # Default
        assert max_v == 4

    def test_head_staleness_for_recomputed_sample(self, executor):
        """Test head staleness calculation for recomputed samples."""
        staleness, allow, max_v = executor.staleness_strategy.calculate_staleness(
            versions=[0, 1, 3, 4, 4],
            loss_mask=[0, 1, 1, 1, 1],
            current_ver=7,
            recompute_version=5,  # Recomputed
            config=executor.config
        )

        # head_staleness = current_ver - min_version = 7 - 1 = 6
        assert staleness == 6
        assert allow == 2  # max(1, max_head_offpolicyness=2)
        assert max_v == 4

    def test_empty_output_versions(self, executor):
        """Test staleness with no valid output versions."""
        staleness, allow, max_v = executor.staleness_strategy.calculate_staleness(
            versions=[-1, -1, -1],
            loss_mask=[0, 1, 1],
            current_ver=5,
            recompute_version=-1,
            config=executor.config
        )

        assert staleness == 0
        assert allow == 1
        assert max_v == -1

    def test_all_loss_mask_zero(self, executor):
        """Test staleness when no tokens are in loss."""
        staleness, allow, max_v = executor.staleness_strategy.calculate_staleness(
            versions=[0, 1, 2, 3],
            loss_mask=[0, 0, 0, 0],  # No output tokens
            current_ver=5,
            recompute_version=-1,
            config=executor.config
        )

        assert staleness == 0
        assert allow == 1
        assert max_v == -1

    def test_allow_staleness_with_max_head_offpolicyness(self, executor):
        """Test that allow_staleness respects max_head_offpolicyness config."""
        executor.config.max_head_offpolicyness = 5

        staleness, allow, max_v = executor.staleness_strategy.calculate_staleness(
            versions=[0, 1, 2],
            loss_mask=[0, 1, 1],
            current_ver=10,
            recompute_version=8,  # Recomputed
            config=executor.config
        )

        # For recomputed samples, allow = max(1, max_head_offpolicyness) = 5
        assert allow == 5


class TestIsSampleTooStale:
    """Test is_sample_too_stale helper function (now in strategy)."""

    def test_fresh_sample_not_stale(self, executor):
        """Test that fresh samples are not considered stale."""
        sample = create_sample(versions=[4, 4, 4, 4, 4])

        is_stale = executor.staleness_strategy.is_sample_too_stale(sample, current_ver=5, config=executor.config)

        # staleness = 5 - 4 = 1, allow = 1, not stale
        assert is_stale == False

    def test_very_stale_sample(self, executor):
        """Test that very stale samples are dropped."""
        sample = create_sample(
            seq_len=5,
            versions=[0, 1, 1, 1, 1],  # First is input, versions 1,1,1,1 are output
            loss_mask=[0, 1, 1, 1, 1]
        )

        is_stale = executor.staleness_strategy.is_sample_too_stale(sample, current_ver=5, config=executor.config)

        # staleness = 5 - 1 = 4, allow = 1, stale!
        assert is_stale == True

    def test_sample_without_versions_not_stale(self, executor):
        """Test that samples without version info are kept."""
        sample = TensorDict({"data": torch.tensor([[1, 2, 3]])}, batch_size=[1])

        is_stale = executor.staleness_strategy.is_sample_too_stale(sample, current_ver=5, config=executor.config)

        # Keep samples without version info
        assert is_stale == False

    def test_recomputed_sample_staleness(self, executor):
        """Test staleness for recomputed samples uses head_staleness."""
        sample = create_sample(
            versions=[1, 2, 4, 4, 4],
            loss_mask=[0, 1, 1, 1, 1],
            recompute_version=5
        )

        is_stale = executor.staleness_strategy.is_sample_too_stale(sample, current_ver=5, config=executor.config)

        # head_staleness = 5 - 2 = 3, allow = max(1, 2) = 2, stale!
        assert is_stale == True


class TestFilterStaleFromCache:
    """Test filter_stale_from_cache helper function (now in strategy)."""

    def test_filters_stale_samples(self, executor):
        """Test that stale samples are removed from cache."""
        executor.result_cache = [
            create_sample(versions=[1] * 10),  # Stale
            create_sample(versions=[4] * 10),  # Fresh
            create_sample(versions=[2] * 10),  # Stale
        ]

        dropped = executor.staleness_strategy.filter_stale_from_cache(
            executor.result_cache, current_ver=5, config=executor.config, logger=executor.logger
        )

        assert dropped == 2
        assert len(executor.result_cache) == 1
        assert executor.result_cache[0].get("versions")[0, 0].item() == 4

    def test_keeps_all_fresh_samples(self, executor):
        """Test that all fresh samples are kept."""
        executor.result_cache = [
            create_sample(versions=[4] * 10),
            create_sample(versions=[5] * 10),
        ]

        dropped = executor.staleness_strategy.filter_stale_from_cache(
            executor.result_cache, current_ver=5, config=executor.config, logger=executor.logger
        )

        assert dropped == 0
        assert len(executor.result_cache) == 2

    def test_empty_cache(self, executor):
        """Test filtering empty cache doesn't crash."""
        executor.result_cache = []

        dropped = executor.staleness_strategy.filter_stale_from_cache(
            executor.result_cache, current_ver=5, config=executor.config, logger=executor.logger
        )

        assert dropped == 0
        assert len(executor.result_cache) == 0


class TestCollectSamplesFromQueue:
    """Test _collect_samples_from_queue helper function."""

    def test_collects_until_count_reached(self, executor):
        """Test that collection stops when count is reached."""
        import time
        # Add samples to queue
        for i in range(5):
            executor.output_queue.put(create_sample(versions=[i] * 10))

        start = time.perf_counter()
        collected = executor._collect_samples_from_queue(
            count=3,
            timeout=10.0,
            start_time=start,
            should_accept=None
        )

        assert collected == 3
        assert len(executor.result_cache) == 3
        assert executor.output_queue.qsize() == 2

    def test_should_accept_filter(self, executor):
        """Test that should_accept filter is applied."""
        import time
        # Add samples with different versions
        for v in [0, 1, 2, 3, 4]:
            executor.output_queue.put(create_sample(versions=[v] * 10))

        # Only accept even versions
        def accept_even(td):
            return td.get("versions")[0, 0].item() % 2 == 0

        start = time.perf_counter()
        collected = executor._collect_samples_from_queue(
            count=2,
            timeout=10.0,
            start_time=start,
            should_accept=accept_even
        )

        # Should have collected v0 and v2 (2 samples)
        assert len(executor.result_cache) == 2
        versions = [td.get("versions")[0, 0].item() for td in executor.result_cache]
        assert all(v % 2 == 0 for v in versions)

    def test_timeout_stops_collection(self, executor):
        """Test that collection respects timeout."""
        import time
        # Add one sample, but need 3
        executor.output_queue.put(create_sample(versions=[0] * 10))

        start = time.perf_counter()
        collected = executor._collect_samples_from_queue(
            count=3,
            timeout=0.2,  # Short timeout
            start_time=start,
            should_accept=None
        )

        # Should only get 1 sample before timeout
        assert len(executor.result_cache) == 1

    def test_empty_queue_returns_zero(self, executor):
        """Test that collecting from empty queue returns 0."""
        import time
        start = time.perf_counter()
        collected = executor._collect_samples_from_queue(
            count=2,
            timeout=0.1,
            start_time=start,
            should_accept=None
        )

        assert collected == 0
        assert len(executor.result_cache) == 0

    def test_should_accept_rejection_counts(self, executor):
        """Test that should_accept rejections are properly counted."""
        import time

        # Add samples that will be rejected
        for i in range(5):
            executor.output_queue.put(create_sample(versions=[i] * 10))

        # Track rejections
        rejected_count = [0]
        def reject_all(td):
            rejected_count[0] += 1
            return False

        start = time.perf_counter()
        collected = executor._collect_samples_from_queue(
            count=2,
            timeout=0.5,
            start_time=start,
            should_accept=reject_all
        )

        # Should have tried all samples and rejected them
        assert collected == 0
        assert rejected_count[0] == 5


class TestPurgeStaleSamplesFromQueue:
    """Test purge_stale_samples_from_queue helper function (now in strategy)."""

    def test_only_purges_when_version_increases(self, executor, mock_engine):
        """Test that purge only happens when version increases."""
        mock_engine._version = 1
        last_purged = 1

        executor.output_queue.put(create_sample(versions=[0] * 10))

        # Should not purge since version hasn't changed
        new_last_purged = executor.staleness_strategy.purge_stale_samples_from_queue(
            executor.output_queue, current_ver=1, last_purged_ver=last_purged,
            inference_engine=executor.inference_engine, result_cache=executor.result_cache,
            config=executor.config, logger=executor.logger
        )

        assert executor.output_queue.qsize() == 1
        assert new_last_purged == 1

    def test_purges_and_updates_version(self, executor, mock_engine):
        """Test that purge updates last_purged_ver."""
        mock_engine._version = 2
        last_purged = 0

        executor.output_queue.put(create_sample(versions=[1] * 10))

        new_last_purged = executor.staleness_strategy.purge_stale_samples_from_queue(
            executor.output_queue, current_ver=2, last_purged_ver=last_purged,
            inference_engine=executor.inference_engine, result_cache=executor.result_cache,
            config=executor.config, logger=executor.logger
        )

        # Should have purged and updated version
        assert new_last_purged == 2

    def test_puts_fresh_samples_back(self, executor):
        """Test that fresh samples are put back in queue."""
        # Add mixed samples
        executor.output_queue.put(create_sample(versions=[0] * 10))  # Stale
        executor.output_queue.put(create_sample(versions=[4] * 10))  # Fresh

        executor.staleness_strategy.purge_stale_samples_from_queue(
            executor.output_queue, current_ver=5, last_purged_ver=0,
            inference_engine=executor.inference_engine, result_cache=executor.result_cache,
            config=executor.config, logger=executor.logger
        )

        # Should have kept the fresh sample
        assert executor.output_queue.qsize() == 1
        remaining = executor.output_queue.get()
        assert remaining.get("versions")[0, 0].item() == 4

    def test_handles_exception_during_purge(self, executor):
        """Test that exceptions during purge are caught and sample is kept."""
        # Create a sample that will cause exception in staleness check
        bad_sample = TensorDict(
            {"versions": torch.tensor([[]]), "loss_mask": torch.tensor([[]])},
            batch_size=[1]
        )
        executor.output_queue.put(bad_sample)

        # Should not crash, sample should be kept
        executor.staleness_strategy.purge_stale_samples_from_queue(
            executor.output_queue, current_ver=5, last_purged_ver=0,
            inference_engine=executor.inference_engine, result_cache=executor.result_cache,
            config=executor.config, logger=executor.logger
        )

        # Sample should be put back (kept on error)
        assert executor.output_queue.qsize() == 1

    def test_queue_full_error_during_putback(self, executor):
        """Test RuntimeError when queue is full during put-back."""
        # Fill queue completely
        for i in range(executor.output_queue.maxsize):
            executor.output_queue.put(create_sample(versions=[4] * 10))

        # Mock put to always raise Full even with timeout
        import queue
        original_put = executor.output_queue.put

        def mock_put(item, timeout=None):
            raise queue.Full()

        executor.output_queue.put = mock_put

        # Should raise RuntimeError with clear message and log error
        with pytest.raises(RuntimeError, match="queue_size"):
            executor.staleness_strategy.purge_stale_samples_from_queue(
                executor.output_queue, current_ver=5, last_purged_ver=0,
                inference_engine=executor.inference_engine, result_cache=executor.result_cache,
                config=executor.config, logger=executor.logger
            )
        # Check that error was logged
        executor.logger.error.assert_called()

        executor.output_queue.put = original_put


class TestRecomputeStaleLogprobs:
    """Test _recompute_stale_logprobs helper function."""

    def test_recomputes_v_minus_1_tokens(self, executor, mock_engine):
        """Test that v-1 tokens get recomputed."""
        mock_engine._version = 3

        sample = create_sample(
            seq_len=5,
            versions=[0, 2, 2, 2, 2],  # v-1 = 2
            loss_mask=[0, 1, 1, 1, 1]
        )
        executor.result_cache.append(sample)

        executor._recompute_stale_logprobs(current_ver=3)

        # Should have recompute version set
        result = executor.result_cache[0]
        recompute_ver = result.get(RECOMPUTE_VERSION_KEY)
        assert recompute_ver is not None
        assert recompute_ver[0, 0].item() == 3

    def test_skips_current_version_tokens(self, executor, mock_engine):
        """Test that current version tokens are not recomputed."""
        mock_engine._version = 3

        sample = create_sample(
            seq_len=5,
            versions=[0, 3, 3, 3, 3],  # All current version
            loss_mask=[0, 1, 1, 1, 1]
        )
        executor.result_cache.append(sample)

        executor._recompute_stale_logprobs(current_ver=3)

        # Should not set recompute version (or set to -1)
        result = executor.result_cache[0]
        recompute_ver = result.get(RECOMPUTE_VERSION_KEY)
        if recompute_ver is not None:
            # If set, should be -1 (default, not patched)
            assert recompute_ver[0, 0].item() == -1

    def test_handles_missing_recompute_method(self, executor):
        """Test graceful handling when engine doesn't support recompute."""
        # Create mock engine without recompute method
        class NoRecomputeEngine:
            def get_version(self):
                return 3

        executor.inference_engine = NoRecomputeEngine()

        sample = create_sample(versions=[0, 2, 2, 2, 2], loss_mask=[0, 1, 1, 1, 1])
        executor.result_cache.append(sample)

        # Should not crash
        executor._recompute_stale_logprobs(current_ver=3)
        assert len(executor.result_cache) == 1

    def test_handles_exception_during_recompute(self, executor, mock_engine):
        """Test that exceptions during recompute are caught and sample is kept."""
        mock_engine._version = 3

        # Mock recompute to raise an exception
        def failing_recompute(input_ids, start_index):
            raise RuntimeError("Recompute failed")

        mock_engine.recompute_output_logprobs_sync = failing_recompute

        sample = create_sample(
            seq_len=5,
            versions=[0, 2, 2, 2, 2],
            loss_mask=[0, 1, 1, 1, 1]
        )
        executor.result_cache.append(sample)

        # Should not crash, sample should be kept
        executor._recompute_stale_logprobs(current_ver=3)
        assert len(executor.result_cache) == 1

    def test_sample_without_versions_skipped(self, executor, mock_engine):
        """Test that samples without version info are skipped."""
        mock_engine._version = 3

        # Sample without versions key
        sample = TensorDict({"data": torch.tensor([[1, 2, 3]])}, batch_size=[1])
        executor.result_cache.append(sample)

        # Should not crash
        executor._recompute_stale_logprobs(current_ver=3)
        assert len(executor.result_cache) == 1

    def test_mixed_versions_recompute_correctly(self, executor, mock_engine):
        """Test recompute with mix of v-1 and current version tokens."""
        mock_engine._version = 5

        sample = create_sample(
            seq_len=8,
            versions=[0, 4, 4, 5, 5, 4, 4, 5],  # Mix of v-1 (4) and v (5)
            loss_mask=[0, 1, 1, 1, 1, 1, 1, 1]
        )
        executor.result_cache.append(sample)

        executor._recompute_stale_logprobs(current_ver=5)

        # Should have recompute version set (has v-1 tokens)
        result = executor.result_cache[0]
        recompute_ver = result.get(RECOMPUTE_VERSION_KEY)
        assert recompute_ver is not None
        assert recompute_ver[0, 0].item() == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
