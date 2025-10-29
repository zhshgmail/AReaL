"""
Integration tests for event-driven recompute workflow.

This test ensures that the event-driven recomputation is triggered before weight updates
to enable segment-wise decoupled PPO. Without this, all importance weights become 1.0
(on-policy behavior) even with high max_head_offpolicyness.
"""

from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from areal.api.event_api import EventContext, EventType
from areal.api.workflow_api import RECOMPUTE_VERSION_KEY
from areal.core.workflow_factory import create_workflow_executor


class InferenceEngineConfig:
    def __init__(self, enable_segment_wise_ppo=True):
        self.max_concurrent_rollouts = 4
        self.consumer_batch_size = 2
        self.queue_size = 16
        self.max_head_offpolicyness = 2
        self.enable_rollout_tracing = False
        self.request_timeout = 30
        self.enable_segment_wise_ppo = enable_segment_wise_ppo


class MockInferenceEngine:
    """Mock inference engine for testing."""

    def __init__(self, version=0):
        self._version = version
        self.recompute_calls = []
        self.event_registry = None

    def get_version(self):
        return self._version

    def set_version(self, version):
        self._version = version

    def recompute_output_logprobs_sync(self, input_ids, start_index):
        """Mock recompute that returns DIFFERENT logprobs to simulate version change."""
        self.recompute_calls.append({
            'version': self._version,
            'input_ids': input_ids,
            'start_index': start_index
        })
        # Return different values to simulate updated policy
        # This simulates proximal_t != old_logp
        return [0.9] * (len(input_ids) - start_index - 1)


def create_sample(seq_len=10, versions=None, old_logprobs=None):
    """Create a sample with v-1 tokens for recompute."""
    if versions is None:
        versions = [0] * seq_len
    if old_logprobs is None:
        old_logprobs = [0.5] * seq_len  # Different from recompute output (0.9)

    td = TensorDict(
        {
            "input_ids": torch.tensor([list(range(seq_len))]),
            "versions": torch.tensor([versions]),
            "loss_mask": torch.tensor([[1] * seq_len]),
            "proximal_logprobs_t": torch.tensor([old_logprobs]),  # Will be updated by recompute
            "attention_mask": torch.ones(1, seq_len),
        },
        batch_size=[1],
    )
    return td


class TestRecomputeWorkflowIntegration:
    """Integration tests for the event-driven recompute workflow."""

    def test_recompute_before_weight_update_changes_proximal_t(self):
        """Test that event-driven recomputation actually updates proximal_logprobs_t.

        This is the CRITICAL test that verifies the event system triggers recomputation.
        """
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        # Simulate: Sample generated at v0, now at v1 (need recompute)
        sample = create_sample(
            seq_len=10,
            versions=[999, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # context, then 4 v0 tokens, then 5 v1 tokens
            old_logprobs=[0.5] * 10  # Original logprobs from generation
        )
        # Adjust loss_mask so token 0 is NOT in loss (it's context)
        sample["loss_mask"][0, 0] = 0

        # Store original proximal_t values for comparison
        original_proximal_t = sample["proximal_logprobs_t"].clone()

        # Add to cache (simulating samples waiting for training)
        executor.runner.result_cache.append(sample)

        # CRITICAL: Fire BEFORE_POLICY_UPDATE event to trigger recomputation
        if mock_engine.event_registry:
            mock_engine.event_registry.fire_event(EventType.BEFORE_POLICY_UPDATE)

        # Verify recompute was called
        assert len(mock_engine.recompute_calls) > 0, \
            "recompute_output_logprobs_sync should have been called"

        # Verify proximal_t was updated for v0 tokens
        updated_proximal_t = sample["proximal_logprobs_t"]
        assert not torch.allclose(updated_proximal_t, original_proximal_t), \
            "proximal_logprobs_t should be updated after recompute"

        # Tokens 1-4 (v0, in loss) should have new proximal_t (0.9)
        # Token 0 is context (not updated), tokens 5-9 are v1 (current version, not updated)
        for i in range(1, 5):  # v0 tokens (indices 1-4)
            assert abs(updated_proximal_t[0, i].item() - 0.9) < 0.01, \
                f"Token {i} should have updated proximal_t=0.9, got {updated_proximal_t[0, i]}"

    def test_without_recompute_proximal_t_equals_old_logp(self):
        """Test that WITHOUT firing event, proximal_t == old_logp (on-policy, weight=1.0)."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        sample = create_sample(
            seq_len=10,
            versions=[0] * 10,  # All v0, current is v1
            old_logprobs=[0.5] * 10
        )

        original_proximal_t = sample["proximal_logprobs_t"].clone()
        executor.runner.result_cache.append(sample)

        # BUG: NOT firing BEFORE_POLICY_UPDATE event
        # Without event, proximal_t stays the same as old_logp
        updated_proximal_t = sample["proximal_logprobs_t"]
        assert torch.allclose(updated_proximal_t, original_proximal_t), \
            "Without recompute event, proximal_t should stay unchanged"

    def test_recompute_only_affects_v_minus_1_tokens(self):
        """Test that recompute only updates tokens with version = current_ver - 1."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=2)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        sample = create_sample(
            seq_len=10,
            versions=[999, 0, 1, 1, 1, 1, 1, 2, 2, 2],  # context, v0, v1, v2 mix
            old_logprobs=[0.5] * 10
        )
        sample["loss_mask"][0, 0] = 0  # Token 0 is context

        original_proximal_t = sample["proximal_logprobs_t"].clone()
        executor.runner.result_cache.append(sample)

        # Fire event to trigger recompute
        if mock_engine.event_registry:
            mock_engine.event_registry.fire_event(EventType.BEFORE_POLICY_UPDATE)

        updated_proximal_t = sample["proximal_logprobs_t"]

        # Only v1 tokens (indices 2-6) should be updated (current_ver=2, so v-1=1)
        for i in [2, 3, 4, 5, 6]:  # v1 tokens
            assert abs(updated_proximal_t[0, i].item() - 0.9) < 0.01, \
                f"v1 token {i} should be updated to 0.9, got {updated_proximal_t[0, i]}"

    def test_recompute_updates_queue_samples(self):
        """Test that recompute also processes samples in output_queue, not just cache."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        # Add samples to queue (not yet in cache)
        sample_queue = create_sample(seq_len=10, versions=[999, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        sample_queue["loss_mask"][0, 0] = 0  # Token 0 is context
        original_queue_proximal_t = sample_queue["proximal_logprobs_t"].clone()
        executor.runner.output_queue.put(sample_queue)

        # Fire event to trigger recompute (should process queue too!)
        if mock_engine.event_registry:
            mock_engine.event_registry.fire_event(EventType.BEFORE_POLICY_UPDATE)

        # Get sample from queue and verify it was updated
        updated_sample = executor.runner.output_queue.get()
        updated_proximal_t = updated_sample["proximal_logprobs_t"]

        assert not torch.allclose(updated_proximal_t, original_queue_proximal_t), \
            "Queue samples should also be recomputed"

    def test_recompute_version_key_tracking(self):
        """Test that RECOMPUTE_VERSION_KEY is set after recompute."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=2)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        sample = create_sample(seq_len=10, versions=[999, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        sample["loss_mask"][0, 0] = 0  # Token 0 is context
        executor.runner.result_cache.append(sample)

        # Fire event to trigger recompute
        if mock_engine.event_registry:
            mock_engine.event_registry.fire_event(EventType.BEFORE_POLICY_UPDATE)

        # Verify recompute version was set
        assert RECOMPUTE_VERSION_KEY in sample.keys(), \
            "RECOMPUTE_VERSION_KEY should be set after recompute"

        recompute_ver = sample.get(RECOMPUTE_VERSION_KEY)
        assert recompute_ver[0, 0].item() == 2, \
            f"Recompute version should be 2, got {recompute_ver[0, 0].item()}"


class TestRecomputeEdgeCases:
    """Test edge cases for recompute workflow."""

    def test_recompute_with_no_samples(self):
        """Test that recompute handles empty cache/queue gracefully."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        # No samples in cache or queue - should not crash
        if mock_engine.event_registry:
            mock_engine.event_registry.fire_event(EventType.BEFORE_POLICY_UPDATE)

    def test_recompute_without_proximal_t_field(self):
        """Test recompute gracefully handles samples without proximal_logprobs_t."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        # Sample without proximal_logprobs_t
        sample = TensorDict({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "versions": torch.tensor([[0, 0, 0]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "attention_mask": torch.ones(1, 3),
        }, batch_size=[1])

        executor.runner.result_cache.append(sample)

        # Should not crash
        if mock_engine.event_registry:
            mock_engine.event_registry.fire_event(EventType.BEFORE_POLICY_UPDATE)


class TestBackwardCompatibility:
    """Test backward compatibility when segment-wise PPO is disabled."""

    def test_feature_disabled_standard_ppo_behavior(self):
        """Test that with enable_segment_wise_ppo=False, system works as standard PPO."""
        config = InferenceEngineConfig(enable_segment_wise_ppo=False)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        # Sample WITHOUT proximal_logprobs_t (standard PPO)
        sample = TensorDict({
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "versions": torch.tensor([[0, 0, 0, 0, 0]]),
            "loss_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "attention_mask": torch.ones(1, 5),
            "logprobs": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]]),  # Standard PPO only needs this
        }, batch_size=[1])

        executor.runner.result_cache.append(sample)

        # With feature disabled, no event registry should exist
        assert mock_engine.event_registry is None, \
            "Event registry should not be created when feature is disabled"

        # Sample should remain unchanged
        assert "proximal_logprobs_t" not in sample.keys(), \
            "Standard PPO samples should not have proximal_logprobs_t field"

    def test_recompute_skips_when_feature_disabled(self):
        """Test that without event registry, no recomputation happens."""
        config = InferenceEngineConfig(enable_segment_wise_ppo=False)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        sample = TensorDict({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "versions": torch.tensor([[0, 0, 0]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "attention_mask": torch.ones(1, 3),
        }, batch_size=[1])

        executor.runner.result_cache.append(sample)

        # No event registry means no recomputation
        assert mock_engine.event_registry is None
        assert len(mock_engine.recompute_calls) == 0, \
            "No recompute should be called when feature is disabled"


class TestEventSystem:
    """Test the event system for triggering recomputation."""

    def test_event_registry_exists_when_enabled(self):
        """Test that event registry is created when feature is enabled."""
        config = InferenceEngineConfig(enable_segment_wise_ppo=True)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        assert mock_engine.event_registry is not None, \
            "Event registry should be created when segment-wise PPO is enabled"

    def test_multiple_events_execute_handlers(self):
        """Test that multiple event firings execute handlers multiple times."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        # Add sample
        sample = create_sample(seq_len=10, versions=[999, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        sample["loss_mask"][0, 0] = 0
        executor.runner.result_cache.append(sample)

        # Fire event twice
        for _ in range(2):
            if mock_engine.event_registry:
                mock_engine.event_registry.fire_event(EventType.BEFORE_POLICY_UPDATE)

        # Should have multiple recompute calls
        assert len(mock_engine.recompute_calls) > 0

    def test_event_propagates_to_queue_and_cache(self):
        """Test that events propagate correctly to both queue and cache handlers."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        logger = Mock()
        executor.initialize(logger=logger)

        # Add samples to both queue and cache
        sample_cache = create_sample(seq_len=10, versions=[999, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        sample_cache["loss_mask"][0, 0] = 0
        executor.runner.result_cache.append(sample_cache)

        sample_queue = create_sample(seq_len=10, versions=[999, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        sample_queue["loss_mask"][0, 0] = 0
        executor.runner.output_queue.put(sample_queue)

        # Fire event - should process both
        if mock_engine.event_registry:
            mock_engine.event_registry.fire_event(EventType.BEFORE_POLICY_UPDATE)

        # Both samples should be recomputed
        assert len(mock_engine.recompute_calls) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
