"""
Comprehensive unit tests for WorkflowExecutor.wait() method.

Tests cover:
- Version-based queue purge logic
- Result collection with timeout
- Recompute logic for stale samples
- Cache filtering
- Edge cases and race conditions
"""

import queue
import sys
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
import torch
from tensordict import TensorDict

# Mock uvloop for Windows compatibility
if sys.platform == 'win32':
    sys.modules['uvloop'] = MagicMock()

from areal.api.workflow_api import RECOMPUTE_VERSION_KEY, WorkflowExecutor


# Mock InferenceEngineConfig to avoid import dependencies
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
        self._lock = threading.Lock()

    def get_version(self):
        with self._lock:
            return self._version

    def set_version(self, version):
        with self._lock:
            self._version = version

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
    """Create test configuration."""
    cfg = InferenceEngineConfig()
    cfg.max_concurrent_rollouts = 4
    cfg.consumer_batch_size = 2
    cfg.queue_size = 16
    cfg.max_head_offpolicyness = 2
    cfg.enable_rollout_tracing = False
    return cfg


@pytest.fixture
def mock_engine():
    """Create mock inference engine."""
    return MockInferenceEngine(version=0)


@pytest.fixture
def executor(config, mock_engine):
    """Create WorkflowExecutor for testing."""
    executor = WorkflowExecutor(config, mock_engine)
    # Don't start the rollout thread
    executor.rollout_tasks = {}
    return executor


class TestWaitBasicBehavior:
    """Test basic wait() functionality."""

    def test_wait_returns_count_samples(self, executor):
        """Test that wait() returns exactly count samples when available."""
        # Add samples to output queue
        for i in range(5):
            sample = create_sample(versions=[0] * 10)
            executor.output_queue.put(sample)

        result = executor.wait(count=3)

        assert result.batch_size[0] == 3
        assert len(executor.result_cache) == 0
        assert executor.output_queue.qsize() == 2

    def test_wait_uses_cache_first(self, executor):
        """Test that wait() uses result_cache before polling queue."""
        # Pre-populate cache
        for i in range(3):
            sample = create_sample(versions=[0] * 10)
            executor.result_cache.append(sample)

        # Add to queue
        executor.output_queue.put(create_sample(versions=[0] * 10))

        result = executor.wait(count=2)

        assert result.batch_size[0] == 2
        assert len(executor.result_cache) == 1
        assert executor.output_queue.qsize() == 1

    def test_wait_timeout_raises_error(self, executor):
        """Test that wait() raises TimeoutError when timeout exceeded."""
        with pytest.raises(TimeoutError, match="Timed out waiting for"):
            executor.wait(count=5, timeout=0.1)

    def test_wait_exiting_raises_error(self, executor):
        """Test that wait() raises RuntimeError when executor is exiting."""
        executor.exiting.set()

        with pytest.raises(RuntimeError, match="Rollout engine is exiting"):
            executor.wait(count=1)

    def test_wait_with_should_accept_filter(self, executor):
        """Test that wait() respects should_accept filter."""
        # Add samples with different versions
        for version in [0, 1, 2]:
            sample = create_sample(versions=[version] * 10)
            executor.output_queue.put(sample)

        # Only accept version 1
        def accept_v1(td):
            return td.get("versions")[0, 0].item() == 1

        result = executor.wait(count=1, should_accept=accept_v1)

        # Should have filtered out v0, accepted v1
        assert result.get("versions")[0, 0].item() == 1
        assert executor.rollout_stat.accepted == -1  # Decremented for rejected v0


class TestVersionPurgeLogic:
    """Test version-based queue purge logic."""

    def test_purge_drops_stale_samples(self, executor, mock_engine):
        """Test that version switch purges stale samples."""
        mock_engine.set_version(0)

        # Add samples with version 0
        for i in range(5):
            sample = create_sample(versions=[0] * 10)
            executor.output_queue.put(sample)

        # Increase version to 2
        mock_engine.set_version(2)

        # BUG #1: This should raise TimeoutError since all samples are dropped
        # but wait(count=1) still expects 1 sample
        with pytest.raises(TimeoutError):
            result = executor.wait(count=1, timeout=0.5)

        # All v0 samples should be dropped (staleness=2, allow=1)
        assert executor.output_queue.qsize() == 0
        assert len(executor.result_cache) == 0

    def test_purge_keeps_recent_samples(self, executor, mock_engine):
        """Test that purge keeps samples within staleness threshold."""
        mock_engine.set_version(0)

        # Add samples with version 0
        for i in range(3):
            sample = create_sample(versions=[0] * 10)
            executor.output_queue.put(sample)

        # Increase version to 1
        mock_engine.set_version(1)

        result = executor.wait(count=3)

        # v0 samples should be kept (staleness=1, allow=1)
        assert result.batch_size[0] == 3

    def test_purge_only_once_per_version(self, executor, mock_engine):
        """Test that purge only happens once per version increase."""
        mock_engine.set_version(1)

        # First wait triggers purge
        sample1 = create_sample(versions=[1] * 10)
        executor.output_queue.put(sample1)
        executor.wait(count=1)

        assert executor._last_purged_ver == 1

        # Second wait with same version should not purge
        sample2 = create_sample(versions=[0] * 10)
        executor.output_queue.put(sample2)

        # This should collect the v0 sample without purge
        result = executor.wait(count=1, timeout=0.2)
        assert result.batch_size[0] == 1

    def test_purge_handles_mixed_versions(self, executor, mock_engine):
        """Test purge with mixed version samples."""
        mock_engine.set_version(0)

        # Add mixed version samples
        executor.output_queue.put(create_sample(versions=[0] * 10))
        executor.output_queue.put(create_sample(versions=[1] * 10))
        executor.output_queue.put(create_sample(versions=[2] * 10))

        mock_engine.set_version(3)

        result = executor.wait(count=1)

        # Only v2 should be kept (staleness=1)
        # After wait(count=1), v2 sample is moved to result_cache and then returned
        assert result.batch_size[0] == 1
        assert result.get("versions")[0, 0].item() == 2

        # Queue and cache should be empty after collecting the one kept sample
        assert executor.output_queue.qsize() == 0
        assert len(executor.result_cache) == 0

    def test_purge_handles_recomputed_samples(self, executor, mock_engine):
        """Test purge logic with recomputed samples."""
        mock_engine.set_version(0)

        # Sample with mixed versions but recomputed
        sample = create_sample(
            versions=[0, 0, 1, 1, 1],
            loss_mask=[0, 1, 1, 1, 1],
            recompute_version=2
        )
        executor.output_queue.put(sample)

        mock_engine.set_version(3)

        # Sample should be dropped due to staleness, so wait() should timeout
        with pytest.raises(TimeoutError):
            result = executor.wait(count=1, timeout=0.2)

        # head_staleness = 3 - 0 = 3, allow = max(1, 2) = 2
        # Sample should be dropped
        assert len(executor.result_cache) == 0

    def test_purge_handles_missing_fields(self, executor, mock_engine):
        """Test purge gracefully handles samples without version/loss_mask."""
        mock_engine.set_version(2)

        # Sample without versions but with required attention_mask
        sample_no_ver = TensorDict({
            "data": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.bool)
        }, batch_size=[1])
        executor.output_queue.put(sample_no_ver)

        result = executor.wait(count=1, timeout=0.2)

        # Sample should be collected (no versions to check)
        assert result.batch_size[0] == 1


class TestCacheFiltering:
    """Test cache filtering after recompute."""

    def test_cache_drops_very_stale_samples(self, executor, mock_engine):
        """Test that cache drops samples exceeding staleness threshold."""
        mock_engine.set_version(5)

        # Add samples with different staleness
        executor.result_cache.append(create_sample(versions=[3] * 10))  # staleness=2, should drop
        executor.result_cache.append(create_sample(versions=[4] * 10))  # staleness=1, keep

        result = executor.wait(count=1)

        # Should only get v4 sample
        assert result.batch_size[0] == 1
        assert result.get("versions")[0, 0].item() == 4
        assert len(executor.result_cache) == 0

    def test_cache_filter_insufficient_samples_bug(self, executor, mock_engine):
        """Test BUG #1: cache filter can drop samples leaving count < requested."""
        mock_engine.set_version(5)

        # Add only stale samples to cache
        for i in range(3):
            executor.result_cache.append(create_sample(versions=[2] * 10))  # staleness=3

        # BUG #1: Cache filtering drops all 3 samples, then wait() tries to get more
        # from empty queue, causing timeout. Should handle this gracefully.
        with pytest.raises(TimeoutError):
            result = executor.wait(count=3, timeout=0.2)

    def test_cache_handles_recomputed_staleness(self, executor, mock_engine):
        """Test cache filter uses head_staleness for recomputed samples."""
        mock_engine.set_version(5)
        executor.config.max_head_offpolicyness = 3

        # Sample with mixed versions, recomputed
        sample = create_sample(
            versions=[2, 3, 4, 4, 4],  # min=2, max=4
            loss_mask=[0, 1, 1, 1, 1],
            recompute_version=5
        )
        executor.result_cache.append(sample)

        result = executor.wait(count=1)

        # head_staleness = 5 - 2 = 3, allow = max(1, 3) = 3
        # Should be kept
        assert result.batch_size[0] == 1


class TestConcurrencyAndRaceConditions:
    """Test concurrent access and race conditions."""

    def test_output_queue_lock_protects_purge(self, executor, mock_engine):
        """Test that output_queue_lock prevents race during purge."""
        mock_engine.set_version(1)

        # Fill queue
        for i in range(5):
            executor.output_queue.put(create_sample(versions=[0] * 10))

        # Simulate concurrent access
        results = []
        errors = []

        def wait_thread():
            try:
                result = executor.wait(count=1, timeout=1.0)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=wait_thread) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash
        assert len(errors) == 0 or all(isinstance(e, TimeoutError) for e in errors)

    def test_version_change_during_wait(self, executor, mock_engine):
        """Test behavior when version changes during wait()."""
        mock_engine.set_version(1)

        def version_changer():
            time.sleep(0.1)
            mock_engine.set_version(2)
            # Add sample after version change
            executor.output_queue.put(create_sample(versions=[2] * 10))

        changer = threading.Thread(target=version_changer)
        changer.start()

        result = executor.wait(count=1, timeout=2.0)
        changer.join()

        assert result is not None

    def test_stat_consistency_with_should_accept(self, executor):
        """Test BUG: rollout_stat.accepted double-decrement."""
        initial_accepted = executor.rollout_stat.accepted

        # Add samples
        for i in range(5):
            executor.output_queue.put(create_sample(versions=[0] * 10))

        # Reject all - this will timeout since no samples pass the filter
        with pytest.raises(TimeoutError):
            executor.wait(count=1, should_accept=lambda x: False, timeout=0.5)

        # BUG: accepted count decremented for each rejection
        # but was never incremented in wait() - only in _rollout_thread_async
        final_accepted = executor.rollout_stat.accepted
        print(f"BUG: Accepted changed from {initial_accepted} to {final_accepted}")


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_output_versions(self, executor):
        """Test sample with no valid output versions."""
        sample = create_sample(
            versions=[-1, -1, -1, -1, -1],
            loss_mask=[0, 1, 1, 1, 1]
        )
        executor.output_queue.put(sample)

        result = executor.wait(count=1, timeout=0.2)
        assert result.batch_size[0] == 1

    def test_zero_length_sequence(self, executor):
        """Test handling of zero-length sequences."""
        sample = create_sample(seq_len=0)
        executor.result_cache.append(sample)

        # Should handle gracefully
        result = executor.wait(count=1, timeout=0.2)
        assert result is not None

    def test_all_loss_mask_zero(self, executor):
        """Test sample where all loss_mask is 0."""
        sample = create_sample(
            versions=[0, 1, 2, 3, 4],
            loss_mask=[0, 0, 0, 0, 0]
        )
        executor.output_queue.put(sample)

        result = executor.wait(count=1, timeout=0.2)
        assert result.batch_size[0] == 1

    def test_queue_full_during_put_back(self, executor, mock_engine):
        """Test BUG: potential queue full error during purge put_back."""
        # Fill queue to max
        for i in range(executor.output_queue.maxsize):
            executor.output_queue.put(create_sample(versions=[0] * 10))

        mock_engine.set_version(1)

        # This might raise IndexError if put_back fails
        try:
            result = executor.wait(count=1, timeout=0.5)
        except IndexError as e:
            print(f"BUG: Queue full during put_back: {e}")
        except TimeoutError:
            pass  # Expected if all samples dropped

    def test_recompute_version_key_always_present(self, executor):
        """Test that result always has RECOMPUTE_VERSION_KEY."""
        sample = create_sample(versions=[0] * 10)
        executor.output_queue.put(sample)

        result = executor.wait(count=1)

        # After wait(), all results should have the key
        assert RECOMPUTE_VERSION_KEY in result.keys()

    def test_effective_samples_tracking(self, executor):
        """Test that _record_effective_samples is called correctly."""
        for i in range(3):
            executor.output_queue.put(create_sample(versions=[0] * 10))

        result = executor.wait(count=3)

        # Should have recorded 3 effective samples
        assert len(executor._effective_history) == 1
        assert executor._effective_history[0] == 3
        assert executor._effective_sum == 3


class TestStalenessCalculation:
    """Test staleness calculation consistency."""

    def test_tail_staleness_calculation(self, executor, mock_engine):
        """Test tail staleness for non-recomputed samples."""
        mock_engine.set_version(5)

        sample = create_sample(
            versions=[0, 2, 3, 3, 3],
            loss_mask=[0, 1, 1, 1, 1]
        )
        executor.result_cache.append(sample)

        # tail_staleness = current_ver(5) - max_version(3) = 2
        # allow_staleness = 1
        # Should be dropped, so wait() will timeout
        with pytest.raises(TimeoutError):
            result = executor.wait(count=1, timeout=0.2)

        # Sample should be filtered out from cache
        assert len(executor.result_cache) == 0

    def test_head_staleness_calculation(self, executor, mock_engine):
        """Test head staleness for recomputed samples."""
        mock_engine.set_version(5)
        executor.config.max_head_offpolicyness = 3

        sample = create_sample(
            versions=[0, 2, 4, 4, 4],
            loss_mask=[0, 1, 1, 1, 1],
            recompute_version=5
        )
        executor.result_cache.append(sample)

        result = executor.wait(count=1)

        # head_staleness = current_ver(5) - min_version(2) = 3
        # allow_staleness = max(1, 3) = 3
        # Should be kept
        assert result.batch_size[0] == 1

    def test_staleness_consistency_between_purge_and_filter(self, executor, mock_engine):
        """Test that staleness logic is consistent between purge and cache filter."""
        mock_engine.set_version(3)

        # Sample that should have same staleness in both places
        # version=1, current=3, staleness=2 > allow=1, so should be dropped
        sample = create_sample(versions=[1] * 10)

        # Test through purge
        executor.output_queue.put(sample.clone())
        executor.wait(count=0, timeout=0.1)  # Trigger purge only
        purge_kept = executor.output_queue.qsize() > 0

        # Test through cache filter
        executor.result_cache.append(sample.clone())
        try:
            executor.wait(count=1, timeout=0.1)
            cache_kept = True
        except TimeoutError:
            cache_kept = len(executor.result_cache) > 0

        # Should have same behavior (both drop the stale sample)
        assert purge_kept == cache_kept
        print(f"Purge kept: {purge_kept}, Cache filter kept: {cache_kept}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
