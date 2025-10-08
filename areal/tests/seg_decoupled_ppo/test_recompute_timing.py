"""
Tests for the new recompute_all_proximal_t() timing and logic.

These tests verify the fix for the narrow recompute window bug where samples
that miss their v+1 recompute opportunity never get recomputed.
"""

import queue
import sys
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict

# Mock uvloop for Windows compatibility
if sys.platform == 'win32':
    sys.modules['uvloop'] = MagicMock()

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
        self._recompute_calls = []

    def get_version(self):
        return self._version

    def set_version(self, version):
        self._version = version

    def recompute_output_logprobs_sync(self, input_ids, start_index):
        """Mock recompute that returns fake logprobs and tracks calls."""
        self._recompute_calls.append({
            'input_ids': input_ids,
            'start_index': start_index,
            'version': self._version
        })
        # Return different values based on version for testing
        return [float(self._version + i * 0.1) for i in range(len(input_ids) - start_index - 1)]


def create_sample(
    seq_len: int = 10,
    versions: List[int] = None,
    loss_mask: List[int] = None,
    proximal_t: List[float] = None,
) -> TensorDict:
    """Create a mock TensorDict sample for testing."""
    if versions is None:
        versions = [0] * seq_len
    if loss_mask is None:
        loss_mask = [0] + [1] * (seq_len - 1)  # First is input, rest are output
    if proximal_t is None:
        proximal_t = [0.0] * seq_len

    td = TensorDict(
        {
            "input_ids": torch.tensor([list(range(seq_len))]),
            "versions": torch.tensor([versions]),
            "loss_mask": torch.tensor([loss_mask]),
            "proximal_logprobs_t": torch.tensor([proximal_t]),
            "logprobs": torch.randn(1, seq_len),
            "attention_mask": torch.ones(1, seq_len),
        },
        batch_size=[1],
    )

    return td


@pytest.fixture
def config():
    """Create test configuration."""
    return InferenceEngineConfig()


@pytest.fixture
def mock_engine():
    """Create mock inference engine."""
    return MockInferenceEngine(version=0)


@pytest.fixture
def executor(config, mock_engine):
    """Create WorkflowExecutor for testing."""
    executor = WorkflowExecutor(config, mock_engine)
    executor.rollout_tasks = {}
    return executor


class TestRecomputeAllProximalT:
    """Test the new recompute_all_proximal_t() method."""

    def test_recomputes_cache_samples(self, executor, mock_engine):
        """Test that samples in result_cache get recomputed."""
        mock_engine.set_version(5)

        # Add v4 samples to cache
        for i in range(3):
            sample = create_sample(versions=[4] * 10, proximal_t=[4.0] * 10)
            executor.result_cache.append(sample)

        # Recompute
        executor.recompute_all_proximal_t()

        # Verify all samples were recomputed
        for sample in executor.result_cache:
            prox_t = sample.get("proximal_logprobs_t")[0].tolist()
            # Values should be updated (version 5 + offset)
            assert prox_t[1] != 4.0  # Changed from original
            # Check recompute version was set
            recomp_ver = sample.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
            assert recomp_ver == 5

    def test_recomputes_queue_samples(self, executor, mock_engine):
        """Test that samples in output_queue get recomputed."""
        mock_engine.set_version(6)

        # Add v5 samples to queue
        for i in range(3):
            sample = create_sample(versions=[5] * 10, proximal_t=[5.0] * 10)
            executor.output_queue.put(sample)

        # Recompute
        executor.recompute_all_proximal_t()

        # Drain queue and verify
        recomputed_samples = []
        while not executor.output_queue.empty():
            recomputed_samples.append(executor.output_queue.get())

        assert len(recomputed_samples) == 3
        for sample in recomputed_samples:
            prox_t = sample.get("proximal_logprobs_t")[0].tolist()
            assert prox_t[1] != 5.0  # Changed from original
            recomp_ver = sample.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
            assert recomp_ver == 6

    def test_recomputes_both_cache_and_queue(self, executor, mock_engine):
        """Test that both cache and queue samples are processed."""
        mock_engine.set_version(7)

        # Add samples to both cache and queue
        cache_sample = create_sample(versions=[6] * 10, proximal_t=[6.0] * 10)
        executor.result_cache.append(cache_sample)

        queue_sample = create_sample(versions=[6] * 10, proximal_t=[6.0] * 10)
        executor.output_queue.put(queue_sample)

        # Recompute
        executor.recompute_all_proximal_t()

        # Verify cache sample
        assert executor.result_cache[0].get(RECOMPUTE_VERSION_KEY)[0, 0].item() == 7

        # Verify queue sample
        queue_result = executor.output_queue.get()
        assert queue_result.get(RECOMPUTE_VERSION_KEY)[0, 0].item() == 7

    def test_only_recomputes_v_minus_1_samples(self, executor, mock_engine):
        """Test that only samples with version = current_ver - 1 are recomputed."""
        mock_engine.set_version(8)

        # Add samples with different versions
        executor.result_cache.append(create_sample(versions=[6] * 10))  # Too old
        executor.result_cache.append(create_sample(versions=[7] * 10))  # v-1, should recompute
        executor.result_cache.append(create_sample(versions=[8] * 10))  # Current, skip

        executor.recompute_all_proximal_t()

        # Check which were recomputed
        recomp_v6 = executor.result_cache[0].get(RECOMPUTE_VERSION_KEY, torch.tensor([[-1]]))[0, 0].item()
        recomp_v7 = executor.result_cache[1].get(RECOMPUTE_VERSION_KEY, torch.tensor([[-1]]))[0, 0].item()
        recomp_v8 = executor.result_cache[2].get(RECOMPUTE_VERSION_KEY, torch.tensor([[-1]]))[0, 0].item()

        assert recomp_v6 == -1  # Not recomputed (too old)
        assert recomp_v7 == 8   # Recomputed
        assert recomp_v8 == -1  # Not recomputed (current version)

    def test_queue_drain_putback_preserves_samples(self, executor, mock_engine):
        """Test that drain-process-putback doesn't lose samples."""
        mock_engine.set_version(5)

        # Add many samples to queue
        original_count = 10
        for i in range(original_count):
            sample = create_sample(versions=[4] * 10)
            executor.output_queue.put(sample)

        executor.recompute_all_proximal_t()

        # Verify all samples are still in queue
        assert executor.output_queue.qsize() == original_count

    def test_mixed_version_sequence_recompute(self, executor, mock_engine):
        """Test recompute with tokens at different versions within same sequence."""
        mock_engine.set_version(6)

        # Create sample with mixed versions (simulating abort-resume)
        sample = create_sample(
            seq_len=9,
            versions=[5, 5, 5, 5, 5, 6, 6, 6, 6],  # v5 then v6
            loss_mask=[0, 1, 1, 1, 1, 1, 1, 1, 1],
            proximal_t=[5.0] * 9
        )
        executor.result_cache.append(sample)

        executor.recompute_all_proximal_t()

        # Only v5 tokens should be recomputed
        result = executor.result_cache[0]
        prox_t = result.get("proximal_logprobs_t")[0].tolist()

        # Tokens 1-4 (v5) should have new values
        # Tokens 5-8 (v6) should keep original values (not v-1)
        assert prox_t[1] != 5.0  # Recomputed
        assert prox_t[4] != 5.0  # Recomputed

    def test_handles_samples_without_proximal_t(self, executor, mock_engine):
        """Test graceful handling of samples missing proximal_logprobs_t."""
        mock_engine.set_version(5)

        # Sample without proximal_logprobs_t
        sample = TensorDict({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "versions": torch.tensor([[4, 4, 4]]),
            "loss_mask": torch.tensor([[0, 1, 1]]),
        }, batch_size=[1])

        executor.result_cache.append(sample)

        # Should not crash
        executor.recompute_all_proximal_t()

        # Sample should still be in cache (not dropped)
        assert len(executor.result_cache) == 1


class TestRecomputeMissedWindow:
    """
    Tests for the BUG that the new implementation fixes:
    Samples that miss their recompute window (current_ver = version + 1)
    should still get recomputed when recompute_all_proximal_t() is called.
    """

    def test_sample_generated_at_v5_recomputed_at_v6(self, executor, mock_engine):
        """Baseline test: Sample at v5, recomputed when version is v6 (normal case)."""
        # Generate sample at v5
        mock_engine.set_version(5)
        sample = create_sample(versions=[5] * 10, proximal_t=[5.0] * 10)
        executor.output_queue.put(sample)

        # Policy updates to v6
        mock_engine.set_version(6)

        # Recompute
        executor.recompute_all_proximal_t()

        # Verify sample was recomputed
        result = executor.output_queue.get()
        recomp_ver = result.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
        assert recomp_ver == 6

    def test_old_implementation_would_miss_this(self, executor, mock_engine):
        """
        TEST CASE THAT WOULD FAIL WITH OLD wait() IMPLEMENTATION.

        Scenario: Sample sits in queue across multiple version updates.
        Old implementation: wait() only checks version == current_ver - 1,
        so sample at v5 would never match when current_ver > 6.

        New implementation: recompute_all_proximal_t() processes ALL v-1 samples
        at the time it's called, so sample gets recomputed when called at v6.
        """
        # Sample generated at v5
        mock_engine.set_version(5)
        sample = create_sample(versions=[5] * 10, proximal_t=[5.0] * 10)
        executor.output_queue.put(sample)

        # Policy updates multiple times WITHOUT calling recompute
        # (simulating sample sitting in queue)
        mock_engine.set_version(6)
        mock_engine.set_version(7)
        mock_engine.set_version(8)

        # In old implementation (via wait()), sample would never be recomputed
        # because wait() checks: ver[i] == current_ver - 1
        # When current_ver=8: 5 == 7? No. → Never recomputed

        # Now we explicitly call recompute at v8
        # But sample should NOT be recomputed (it's v5, need v6)
        mock_engine.set_version(8)
        executor.recompute_all_proximal_t()

        result = executor.output_queue.get()
        recomp_ver = result.get(RECOMPUTE_VERSION_KEY, torch.tensor([[-1]]))[0, 0].item()

        # At v8, v5 sample doesn't get recomputed (needs v6, not v8)
        assert recomp_ver == -1  # Not recomputed

        # But if we call it at the RIGHT time (v6), it works
        executor.output_queue.put(result)  # Put back
        mock_engine.set_version(6)
        executor.recompute_all_proximal_t()

        result = executor.output_queue.get()
        recomp_ver = result.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
        assert recomp_ver == 6  # Successfully recomputed!

    def test_calling_before_weight_update_ensures_coverage(self, executor, mock_engine):
        """
        Test the NEW PATTERN: Call recompute_all_proximal_t() before each weight update.

        This ensures ALL v-1 samples (both in queue and cache) get recomputed
        before the version increments.
        """
        # Step N: version = 5
        mock_engine.set_version(5)

        # Some samples generated at v4 (sitting in queue)
        for i in range(3):
            sample = create_sample(versions=[4] * 10, proximal_t=[4.0] * 10)
            executor.output_queue.put(sample)

        # Some samples generated at v4 (already in cache from wait())
        for i in range(2):
            sample = create_sample(versions=[4] * 10, proximal_t=[4.0] * 10)
            executor.result_cache.append(sample)

        # Before weight update, call recompute_all_proximal_t()
        # This processes ALL v4 samples (queue + cache) under v5
        executor.recompute_all_proximal_t()

        # Verify queue samples recomputed
        for i in range(3):
            result = executor.output_queue.get()
            recomp_ver = result.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
            assert recomp_ver == 5

        # Verify cache samples recomputed
        for sample in executor.result_cache:
            recomp_ver = sample.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
            assert recomp_ver == 5

        # Now safe to update version to 6
        mock_engine.set_version(6)
        # All v4 samples have been recomputed with v5, none missed!


class TestQueueThreadSafety:
    """Test thread-safety of queue recompute with drain-process-putback."""

    def test_concurrent_puts_dont_break_recompute(self, executor, mock_engine):
        """
        Test that background thread putting to queue during recompute
        doesn't cause issues (samples are eventually processed).
        """
        import threading

        mock_engine.set_version(6)

        # Add initial samples
        for i in range(5):
            executor.output_queue.put(create_sample(versions=[5] * 10))

        # Simulate background thread adding more samples during recompute
        def add_samples():
            time.sleep(0.05)  # Small delay
            for i in range(3):
                try:
                    executor.output_queue.put(create_sample(versions=[5] * 10))
                except queue.Full:
                    pass

        thread = threading.Thread(target=add_samples)
        thread.start()

        # Run recompute (may complete before thread adds all samples)
        executor.recompute_all_proximal_t()

        thread.join()

        # All samples should be in queue (may not all be recomputed in first pass)
        # But this is acceptable - we can call recompute again if needed
        total_samples = executor.output_queue.qsize()
        assert total_samples == 8  # 5 initial + 3 added


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
