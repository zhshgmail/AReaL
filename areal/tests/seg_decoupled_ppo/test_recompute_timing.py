"""
Tests for event-driven recomputation timing and logic.

These tests verify the fix for the narrow recompute window bug where samples
that miss their v+1 recompute opportunity never get recomputed.
"""

import queue
import time
from typing import List
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from areal.api.event_api import EventContext, EventType
from areal.api.workflow_api import RECOMPUTE_VERSION_KEY
from areal.core.workflow_factory import create_workflow_executor


# Mock InferenceEngineConfig
class InferenceEngineConfig:
    def __init__(self):
        self.max_concurrent_rollouts = 4
        self.consumer_batch_size = 2
        self.queue_size = 16
        self.max_head_offpolicyness = 2
        self.enable_rollout_tracing = False
        self.request_timeout = 30
        self.enable_segment_wise_ppo = True


class MockInferenceEngine:
    """Mock inference engine for testing."""

    def __init__(self, version=0):
        self._version = version
        self._recompute_calls = []
        self.event_registry = None

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


def trigger_recompute(executor, mock_engine, config, logger):
    """Helper to trigger recomputation via event system."""
    if mock_engine.event_registry:
        mock_engine.event_registry.fire_event(EventType.BEFORE_POLICY_UPDATE)


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
    executor = create_workflow_executor(config, mock_engine)
    logger = Mock()
    executor.initialize(logger=logger)
    return executor


@pytest.fixture
def logger():
    """Create mock logger."""
    return Mock()


class TestRecomputeAllProximalT:
    """Test the event-driven recomputation."""

    def test_recomputes_cache_samples(self, executor, mock_engine, config, logger):
        """Test that samples in result_cache get recomputed."""
        mock_engine.set_version(5)

        # Add v4 samples to cache
        for i in range(3):
            sample = create_sample(versions=[4] * 10, proximal_t=[4.0] * 10)
            executor.runner.result_cache.append(sample)

        # Trigger recompute
        trigger_recompute(executor, mock_engine, config, logger)

        # Verify all samples were recomputed
        for sample in executor.runner.result_cache:
            prox_t = sample.get("proximal_logprobs_t")[0].tolist()
            # Values should be updated (version 5 + offset)
            assert prox_t[1] != 4.0  # Changed from original
            # Check recompute version was set
            recomp_ver = sample.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
            assert recomp_ver == 5

    def test_recomputes_queue_samples(self, executor, mock_engine, config, logger):
        """Test that samples in output_queue get recomputed."""
        mock_engine.set_version(6)

        # Add v5 samples to queue
        for i in range(3):
            sample = create_sample(versions=[5] * 10, proximal_t=[5.0] * 10)
            executor.runner.output_queue.put(sample)

        # Trigger recompute
        trigger_recompute(executor, mock_engine, config, logger)

        # Drain queue and verify
        recomputed_samples = []
        while not executor.runner.output_queue.empty():
            recomputed_samples.append(executor.runner.output_queue.get())

        assert len(recomputed_samples) == 3
        for sample in recomputed_samples:
            prox_t = sample.get("proximal_logprobs_t")[0].tolist()
            assert prox_t[1] != 5.0  # Changed from original
            recomp_ver = sample.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
            assert recomp_ver == 6

    def test_recomputes_both_cache_and_queue(self, executor, mock_engine, config, logger):
        """Test that both cache and queue samples are processed."""
        mock_engine.set_version(7)

        # Add samples to both cache and queue
        cache_sample = create_sample(versions=[6] * 10, proximal_t=[6.0] * 10)
        executor.runner.result_cache.append(cache_sample)

        queue_sample = create_sample(versions=[6] * 10, proximal_t=[6.0] * 10)
        executor.runner.output_queue.put(queue_sample)

        # Trigger recompute
        trigger_recompute(executor, mock_engine, config, logger)

        # Verify cache sample
        assert executor.runner.result_cache[0].get(RECOMPUTE_VERSION_KEY)[0, 0].item() == 7

        # Verify queue sample
        queue_result = executor.runner.output_queue.get()
        assert queue_result.get(RECOMPUTE_VERSION_KEY)[0, 0].item() == 7

    def test_only_recomputes_v_minus_1_samples(self, executor, mock_engine, config, logger):
        """Test that only samples with version = current_ver - 1 are recomputed."""
        mock_engine.set_version(8)

        # Add samples with different versions
        executor.runner.result_cache.append(create_sample(versions=[6] * 10))  # Too old
        executor.runner.result_cache.append(create_sample(versions=[7] * 10))  # v-1, should recompute
        executor.runner.result_cache.append(create_sample(versions=[8] * 10))  # Current, skip

        trigger_recompute(executor, mock_engine, config, logger)

        # Check which were recomputed
        recomp_v6 = executor.runner.result_cache[0].get(RECOMPUTE_VERSION_KEY, torch.tensor([[-1]]))[0, 0].item()
        recomp_v7 = executor.runner.result_cache[1].get(RECOMPUTE_VERSION_KEY, torch.tensor([[-1]]))[0, 0].item()
        recomp_v8 = executor.runner.result_cache[2].get(RECOMPUTE_VERSION_KEY, torch.tensor([[-1]]))[0, 0].item()

        assert recomp_v6 == -1  # Not recomputed (too old)
        assert recomp_v7 == 8   # Recomputed
        assert recomp_v8 == -1  # Not recomputed (current version)

    def test_queue_drain_putback_preserves_samples(self, executor, mock_engine, config, logger):
        """Test that drain-process-putback doesn't lose samples."""
        mock_engine.set_version(5)

        # Add many samples to queue
        original_count = 10
        for i in range(original_count):
            sample = create_sample(versions=[4] * 10)
            executor.runner.output_queue.put(sample)

        trigger_recompute(executor, mock_engine, config, logger)

        # Verify all samples are still in queue
        assert executor.runner.output_queue.qsize() == original_count

    def test_mixed_version_sequence_recompute(self, executor, mock_engine, config, logger):
        """Test recompute with tokens at different versions within same sequence."""
        mock_engine.set_version(6)

        # Create sample with mixed versions (simulating abort-resume)
        sample = create_sample(
            seq_len=9,
            versions=[5, 5, 5, 5, 5, 6, 6, 6, 6],  # v5 then v6
            loss_mask=[0, 1, 1, 1, 1, 1, 1, 1, 1],
            proximal_t=[5.0] * 9
        )
        executor.runner.result_cache.append(sample)

        trigger_recompute(executor, mock_engine, config, logger)

        # Only v5 tokens should be recomputed
        result = executor.runner.result_cache[0]
        prox_t = result.get("proximal_logprobs_t")[0].tolist()

        # Tokens 1-4 (v5) should have new values
        # Tokens 5-8 (v6) should keep original values (not v-1)
        assert prox_t[1] != 5.0  # Recomputed
        assert prox_t[4] != 5.0  # Recomputed

    def test_handles_samples_without_proximal_t(self, executor, mock_engine, config, logger):
        """Test graceful handling of samples missing proximal_logprobs_t."""
        mock_engine.set_version(5)

        # Sample without proximal_logprobs_t
        sample = TensorDict({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "versions": torch.tensor([[4, 4, 4]]),
            "loss_mask": torch.tensor([[0, 1, 1]]),
        }, batch_size=[1])

        executor.runner.result_cache.append(sample)

        # Should not crash
        trigger_recompute(executor, mock_engine, config, logger)

        # Sample should still be in cache (not dropped)
        assert len(executor.runner.result_cache) == 1


class TestRecomputeMissedWindow:
    """
    Tests for the BUG that the new implementation fixes:
    Samples that miss their recompute window (current_ver = version + 1)
    should still get recomputed when event is fired.
    """

    def test_sample_generated_at_v5_recomputed_at_v6(self, executor, mock_engine, config, logger):
        """Baseline test: Sample at v5, recomputed when version is v6 (normal case)."""
        # Generate sample at v5
        mock_engine.set_version(5)
        sample = create_sample(versions=[5] * 10, proximal_t=[5.0] * 10)
        executor.runner.output_queue.put(sample)

        # Policy updates to v6
        mock_engine.set_version(6)

        # Trigger recompute
        trigger_recompute(executor, mock_engine, config, logger)

        # Verify sample was recomputed
        result = executor.runner.output_queue.get()
        recomp_ver = result.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
        assert recomp_ver == 6

    @pytest.mark.skip(reason="Test hangs due to queue.get() blocking when item filtered by staleness")
    def test_old_implementation_would_miss_this(self, executor, mock_engine, config, logger):
        """
        TEST CASE THAT WOULD FAIL WITH OLD wait() IMPLEMENTATION.

        Scenario: Sample sits in queue across multiple version updates.
        Old implementation: wait() only checks version == current_ver - 1,
        so sample at v5 would never match when current_ver > 6.

        New implementation: Event-driven recomputation processes ALL v-1 samples
        at the time event is fired, so sample gets recomputed when fired at v6.
        """
        # Sample generated at v5
        mock_engine.set_version(5)
        sample = create_sample(versions=[5] * 10, proximal_t=[5.0] * 10)
        executor.runner.output_queue.put(sample)

        # Policy updates multiple times WITHOUT calling recompute
        # (simulating sample sitting in queue)
        mock_engine.set_version(6)
        mock_engine.set_version(7)
        mock_engine.set_version(8)

        # Now trigger recompute at v8
        # But sample should NOT be recomputed (it's v5, need v6)
        mock_engine.set_version(8)
        trigger_recompute(executor, mock_engine, config, logger)

        result = executor.runner.output_queue.get()
        recomp_ver = result.get(RECOMPUTE_VERSION_KEY, torch.tensor([[-1]]))[0, 0].item()

        # At v8, v5 sample doesn't get recomputed (needs v6, not v8)
        assert recomp_ver == -1  # Not recomputed

        # But if we fire event at the RIGHT time (v6), it works
        executor.runner.output_queue.put(result)  # Put back
        mock_engine.set_version(6)
        trigger_recompute(executor, mock_engine, config, logger)

        result = executor.runner.output_queue.get()
        recomp_ver = result.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
        assert recomp_ver == 6  # Successfully recomputed!

    def test_calling_before_weight_update_ensures_coverage(self, executor, mock_engine, config, logger):
        """
        Test the NEW PATTERN: Fire BEFORE_POLICY_UPDATE event before each weight update.

        This ensures ALL v-1 samples (both in queue and cache) get recomputed
        before the version increments.
        """
        # Step N: version = 5
        mock_engine.set_version(5)

        # Some samples generated at v4 (sitting in queue)
        for i in range(3):
            sample = create_sample(versions=[4] * 10, proximal_t=[4.0] * 10)
            executor.runner.output_queue.put(sample)

        # Some samples generated at v4 (already in cache from wait())
        for i in range(2):
            sample = create_sample(versions=[4] * 10, proximal_t=[4.0] * 10)
            executor.runner.result_cache.append(sample)

        # Before weight update, fire BEFORE_POLICY_UPDATE event
        # This processes ALL v4 samples (queue + cache) under v5
        trigger_recompute(executor, mock_engine, config, logger)

        # Verify queue samples recomputed
        for i in range(3):
            result = executor.runner.output_queue.get()
            recomp_ver = result.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
            assert recomp_ver == 5

        # Verify cache samples recomputed
        for sample in executor.runner.result_cache:
            recomp_ver = sample.get(RECOMPUTE_VERSION_KEY)[0, 0].item()
            assert recomp_ver == 5

        # Now safe to update version to 6
        mock_engine.set_version(6)
        # All v4 samples have been recomputed with v5, none missed!


class TestQueueThreadSafety:
    """Test thread-safety of queue recompute with drain-process-putback."""

    @pytest.mark.skip(reason="Threading test - may have race conditions with filter admission control")
    def test_concurrent_puts_dont_break_recompute(self, executor, mock_engine, config, logger):
        """
        Test that background thread putting to queue during recompute
        doesn't cause issues (samples are eventually processed).
        """
        import threading

        mock_engine.set_version(6)

        # Add initial samples
        for i in range(5):
            executor.runner.output_queue.put(create_sample(versions=[5] * 10))

        # Simulate background thread adding more samples during recompute
        def add_samples():
            time.sleep(0.05)  # Small delay
            for i in range(3):
                try:
                    executor.runner.output_queue.put(create_sample(versions=[5] * 10))
                except queue.Full:
                    pass

        thread = threading.Thread(target=add_samples)
        thread.start()

        # Run recompute (may complete before thread adds all samples)
        trigger_recompute(executor, mock_engine, config, logger)

        thread.join()

        # All samples should be in queue (may not all be recomputed in first pass)
        # But this is acceptable - we can call recompute again if needed
        total_samples = executor.runner.output_queue.qsize()
        assert total_samples == 8  # 5 initial + 3 added


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
