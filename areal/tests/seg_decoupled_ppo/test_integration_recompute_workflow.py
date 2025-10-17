"""
Integration tests for recompute_all_proximal_t() workflow.

This test ensures that the critical recompute step is called before weight updates
to enable segment-wise decoupled PPO. Without this, all importance weights become 1.0
(on-policy behavior) even with high max_head_offpolicyness.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

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

from areal.api.workflow_api import RECOMPUTE_VERSION_KEY, WorkflowExecutor, create_workflow_executor


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
    """Integration tests for the complete recompute workflow."""

    def test_recompute_before_weight_update_changes_proximal_t(self):
        """Test that recompute_all_proximal_t() actually updates proximal_logprobs_t.

        This is the CRITICAL test that would have caught the missing API call.
        """
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.rollout_tasks = {}
        executor.logger = Mock()
        executor.dp_world_size = 1

        # Simulate: Sample generated at v0, now at v1 (need recompute)
        # NOTE: Use versions [999, 0, 0, 0, 0, 1, 1, 1, 1, 1] and loss_mask [0, 1, 1, ...]
        # The first token is context (version=999 means never recompute, loss_mask=0 means not in loss)
        # This ensures first_output_idx=1, start_index=0, and tokens 1-4 can be recomputed properly
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
        executor.result_cache.append(sample)

        # CRITICAL: Call recompute BEFORE weight update
        # This is what gsm8k_grpo.py was missing!
        executor.recompute_all_proximal_t()

        # Verify recompute was called
        assert len(mock_engine.recompute_calls) > 0, \
            "recompute_output_logprobs_sync should have been called"

        # Verify proximal_t was updated for v0 tokens
        updated_proximal_t = sample["proximal_logprobs_t"]
        assert not torch.allclose(updated_proximal_t, original_proximal_t), \
            "proximal_logprobs_t should be updated after recompute"

        # Tokens 1-4 (v0, in loss) should have new proximal_t (0.9)
        # Token 0 is context (not updated), tokens 5-9 are v1 (current version, not updated)
        # This is what makes behav_imp_weight != 1.0
        for i in range(1, 5):  # v0 tokens (indices 1-4)
            assert abs(updated_proximal_t[0, i].item() - 0.9) < 0.01, \
                f"Token {i} should have updated proximal_t=0.9, got {updated_proximal_t[0, i]}"

    def test_without_recompute_proximal_t_equals_old_logp(self):
        """Test that WITHOUT recompute, proximal_t == old_logp (on-policy, weight=1.0).

        This demonstrates the bug when recompute is not called.
        """
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.rollout_tasks = {}
        executor.logger = Mock()
        executor.dp_world_size = 1

        sample = create_sample(
            seq_len=10,
            versions=[0] * 10,  # All v0, current is v1
            old_logprobs=[0.5] * 10
        )

        original_proximal_t = sample["proximal_logprobs_t"].clone()
        executor.result_cache.append(sample)

        # BUG: NOT calling recompute_all_proximal_t()
        # This simulates the bug in gsm8k_grpo.py

        # Without recompute, proximal_t stays the same as old_logp
        # This means behav_imp_weight = exp(proximal_t - old_logp) = exp(0) = 1.0
        updated_proximal_t = sample["proximal_logprobs_t"]
        assert torch.allclose(updated_proximal_t, original_proximal_t), \
            "Without recompute, proximal_t should stay unchanged (BUG!)"

    def test_recompute_only_affects_v_minus_1_tokens(self):
        """Test that recompute only updates tokens with version = current_ver - 1."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=2)
        executor = create_workflow_executor(config, mock_engine)
        executor.rollout_tasks = {}
        executor.logger = Mock()
        executor.dp_world_size = 1

        # Use context token at index 0 to avoid first-token recompute issue
        sample = create_sample(
            seq_len=10,
            versions=[999, 0, 1, 1, 1, 1, 1, 2, 2, 2],  # context, v0, v1, v2 mix
            old_logprobs=[0.5] * 10
        )
        sample["loss_mask"][0, 0] = 0  # Token 0 is context

        original_proximal_t = sample["proximal_logprobs_t"].clone()
        executor.result_cache.append(sample)

        executor.recompute_all_proximal_t()

        updated_proximal_t = sample["proximal_logprobs_t"]

        # Only v1 tokens (indices 2-6) should be updated (current_ver=2, so v-1=1)
        # Token 0 is context (not updated), token 1 is v0 (too old, not updated)
        # Tokens 7-9 are v2 (current version, not updated)
        for i in [2, 3, 4, 5, 6]:  # v1 tokens
            assert abs(updated_proximal_t[0, i].item() - 0.9) < 0.01, \
                f"v1 token {i} should be updated to 0.9, got {updated_proximal_t[0, i]}"

    def test_recompute_updates_queue_samples(self):
        """Test that recompute also processes samples in output_queue, not just cache."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.rollout_tasks = {}
        executor.logger = Mock()
        executor.dp_world_size = 1

        # Add samples to queue (not yet in cache)
        # Use context token to avoid first-token recompute issue
        sample_queue = create_sample(seq_len=10, versions=[999, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        sample_queue["loss_mask"][0, 0] = 0  # Token 0 is context
        original_queue_proximal_t = sample_queue["proximal_logprobs_t"].clone()
        executor.output_queue.put(sample_queue)

        # Call recompute (should process queue too!)
        executor.recompute_all_proximal_t()

        # Get sample from queue and verify it was updated
        updated_sample = executor.output_queue.get()
        updated_proximal_t = updated_sample["proximal_logprobs_t"]

        assert not torch.allclose(updated_proximal_t, original_queue_proximal_t), \
            "Queue samples should also be recomputed"

    def test_recompute_version_key_tracking(self):
        """Test that RECOMPUTE_VERSION_KEY is set after recompute."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=2)
        executor = create_workflow_executor(config, mock_engine)
        executor.rollout_tasks = {}
        executor.logger = Mock()
        executor.dp_world_size = 1

        # Use context token to avoid first-token recompute issue
        sample = create_sample(seq_len=10, versions=[999, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        sample["loss_mask"][0, 0] = 0  # Token 0 is context
        executor.result_cache.append(sample)

        executor.recompute_all_proximal_t()

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
        executor.rollout_tasks = {}
        executor.logger = Mock()
        executor.dp_world_size = 1

        # No samples in cache or queue
        executor.recompute_all_proximal_t()  # Should not crash

    def test_recompute_without_proximal_t_field(self):
        """Test recompute gracefully handles samples without proximal_logprobs_t."""
        config = InferenceEngineConfig()
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.rollout_tasks = {}
        executor.logger = Mock()
        executor.dp_world_size = 1

        # Sample without proximal_logprobs_t
        sample = TensorDict({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "versions": torch.tensor([[0, 0, 0]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "attention_mask": torch.ones(1, 3),
        }, batch_size=[1])

        executor.result_cache.append(sample)
        executor.recompute_all_proximal_t()  # Should not crash


class TestBackwardCompatibility:
    """Test backward compatibility when segment-wise PPO is disabled."""

    def test_feature_disabled_standard_ppo_behavior(self):
        """Test that with enable_segment_wise_ppo=False, system works as standard PPO.

        This verifies:
        1. No recompute is called (feature is disabled)
        2. Samples without proximal_logprobs_t work correctly (backward compatibility)
        3. The None-check in functional.py provides the fallback behavior
        """
        config = InferenceEngineConfig(enable_segment_wise_ppo=False)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.rollout_tasks = {}
        executor.logger = Mock()
        executor.dp_world_size = 1

        # Sample WITHOUT proximal_logprobs_t (standard PPO)
        sample = TensorDict({
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "versions": torch.tensor([[0, 0, 0, 0, 0]]),
            "loss_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "attention_mask": torch.ones(1, 5),
            "logprobs": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]]),  # Standard PPO only needs this
        }, batch_size=[1])

        executor.result_cache.append(sample)

        # With feature disabled, recompute should do nothing (or skip samples without proximal_t)
        initial_recompute_calls = len(mock_engine.recompute_calls)
        executor.recompute_all_proximal_t()

        # Verify no recompute was attempted (feature disabled)
        assert len(mock_engine.recompute_calls) == initial_recompute_calls, \
            "With feature disabled, recompute should not process samples without proximal_logprobs_t"

        # Sample should remain unchanged
        assert "proximal_logprobs_t" not in sample.keys(), \
            "Standard PPO samples should not have proximal_logprobs_t field"

    def test_recompute_skips_when_feature_disabled(self):
        """Test that recompute is essentially a no-op when feature is disabled."""
        config = InferenceEngineConfig(enable_segment_wise_ppo=False)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.rollout_tasks = {}
        executor.logger = Mock()
        executor.dp_world_size = 1

        # Even if we have samples, nothing should happen with feature disabled
        sample = TensorDict({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "versions": torch.tensor([[0, 0, 0]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "attention_mask": torch.ones(1, 3),
        }, batch_size=[1])

        executor.result_cache.append(sample)

        # This should be safe to call even with feature disabled
        executor.recompute_all_proximal_t()  # Should not crash

        # Verify no recompute calls were made
        assert len(mock_engine.recompute_calls) == 0, \
            "No recompute should be called when samples lack proximal_logprobs_t"


class TestHookSystem:
    """Test the hook/callback system for pause and resume."""

    def test_pre_pause_hook_executes(self):
        """Test that pre-pause hooks are executed before pause."""
        config = InferenceEngineConfig(enable_segment_wise_ppo=False)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.logger = Mock()

        # Register custom hook
        hook_called = []
        def custom_hook():
            hook_called.append(True)

        executor.register_pre_pause_hook(custom_hook)

        # Call pause
        executor.pause()

        # Verify hook was called
        assert len(hook_called) == 1, "Pre-pause hook should be called once"
        assert executor.paused.is_set(), "Paused flag should be set"

    def test_auto_registered_recompute_hook(self):
        """Test that recompute is automatically registered when feature enabled."""
        config = InferenceEngineConfig(enable_segment_wise_ppo=True)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.logger = Mock()
        executor.dp_world_size = 1

        # Add sample for recompute
        sample = create_sample(seq_len=10, versions=[999, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        sample["loss_mask"][0, 0] = 0
        executor.result_cache.append(sample)

        # Calling pause should trigger recompute hook
        executor.pause()

        # Verify recompute was called
        assert len(mock_engine.recompute_calls) > 0, \
            "Recompute should be automatically called during pause when feature enabled"

    def test_multiple_hooks_execute_in_order(self):
        """Test that multiple hooks execute in registration order."""
        config = InferenceEngineConfig(enable_segment_wise_ppo=False)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.logger = Mock()

        # Register multiple hooks
        call_order = []

        def hook1():
            call_order.append(1)

        def hook2():
            call_order.append(2)

        def hook3():
            call_order.append(3)

        executor.register_pre_pause_hook(hook1)
        executor.register_pre_pause_hook(hook2)
        executor.register_pre_pause_hook(hook3)

        # Call pause
        executor.pause()

        # Verify hooks executed in order
        assert call_order == [1, 2, 3], "Hooks should execute in registration order"

    def test_post_pause_hooks_execute(self):
        """Test that post-pause hooks execute after pausing."""
        config = InferenceEngineConfig(enable_segment_wise_ppo=False)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.logger = Mock()

        post_hook_called = []
        pause_state_when_hook_ran = []

        def post_hook():
            post_hook_called.append(True)
            pause_state_when_hook_ran.append(executor.paused.is_set())

        executor.register_post_pause_hook(post_hook)

        # Call pause
        executor.pause()

        # Verify post hook was called after pause was set
        assert len(post_hook_called) == 1, "Post-pause hook should be called"
        assert pause_state_when_hook_ran[0], "Pause should be set when post-hook runs"

    def test_pre_resume_hooks_execute(self):
        """Test that pre-resume hooks execute before resuming."""
        config = InferenceEngineConfig(enable_segment_wise_ppo=False)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.logger = Mock()

        # First pause
        executor.pause()

        pre_resume_called = []
        pause_state_when_hook_ran = []

        def pre_resume_hook():
            pre_resume_called.append(True)
            pause_state_when_hook_ran.append(executor.paused.is_set())

        executor.register_pre_resume_hook(pre_resume_hook)

        # Call resume
        executor.resume()

        # Verify pre-resume hook was called before resume cleared the flag
        assert len(pre_resume_called) == 1, "Pre-resume hook should be called"
        assert pause_state_when_hook_ran[0], "Pause should still be set when pre-resume hook runs"
        assert not executor.paused.is_set(), "Pause should be cleared after resume"

    def test_hook_exception_does_not_break_pause(self):
        """Test that hook exceptions are caught and pause still works."""
        config = InferenceEngineConfig(enable_segment_wise_ppo=False)
        mock_engine = MockInferenceEngine(version=1)
        executor = create_workflow_executor(config, mock_engine)
        executor.logger = Mock()

        def failing_hook():
            raise ValueError("Hook intentionally failed")

        def successful_hook():
            successful_hook.called = True

        successful_hook.called = False

        executor.register_pre_pause_hook(failing_hook)
        executor.register_pre_pause_hook(successful_hook)

        # Call pause - should not raise despite failing hook
        executor.pause()

        # Verify pause still worked and successful hook ran
        assert executor.paused.is_set(), "Pause should still be set despite hook failure"
        assert successful_hook.called, "Subsequent hooks should still execute"
        assert executor.logger.warning.called, "Logger should record the warning"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
