"""Critical tests for segment-wise decoupled PPO's core distinction.

This test file guards the CRITICAL invariant of SDP:
- PPO clipping ratio ALWAYS uses proximal_logprobs (never proximal_logprobs_t)
- Behavioral importance weight uses proximal_logprobs_t (when SDP enabled)

If these tests fail, the algorithm is fundamentally broken.
"""

import pytest
import torch
from areal.utils.functional import ppo_actor_loss_fn


class TestSDPCriticalDistinction:
    """Test the critical distinction between SDP and standard decoupled PPO.

    SDP's key innovation:
    - Replace π_proximal with π_proximal_t ONLY in behavioral importance weight
    - PPO clipping ratio still uses π_proximal (current policy)

    This is the core of what makes SDP work - it reduces variance in behavioral
    importance weights while keeping the PPO objective unchanged.
    """

    def test_ppo_ratio_never_uses_proximal_t(self):
        """CRITICAL: Verify PPO ratio is computed from proximal_logprobs, not proximal_logprobs_t.

        This is the most important test for SDP. The PPO clipping ratio must always
        be computed as ratio = π_current / π_proximal, where π_proximal = current policy.

        Even when proximal_logprobs_t is provided, the PPO ratio computation must
        ignore it and use only proximal_logprobs.
        """
        batch_size, seq_len = 2, 4

        # Create distinct values for each logprob type
        logprobs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        proximal_logprobs = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]])

        # Make proximal_t VERY different to ensure it's not being used
        proximal_logprobs_t = torch.tensor([[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]])

        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Compute loss with SDP enabled
        loss, stats = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            proximal_logprobs_t=proximal_logprobs_t,
        )

        # CRITICAL ASSERTION: importance_weight must be exp(logprobs - proximal_logprobs)
        # NOT exp(logprobs - proximal_logprobs_t)
        expected_ppo_ratio = torch.exp(logprobs - proximal_logprobs)
        actual_ppo_ratio = stats["importance_weight"]

        assert torch.allclose(actual_ppo_ratio, expected_ppo_ratio, atol=1e-5), \
            f"PPO ratio MUST be computed from proximal_logprobs, not proximal_logprobs_t!\n" \
            f"Expected: {expected_ppo_ratio}\nActual: {actual_ppo_ratio}"

        # Verify proximal_t is NOT accidentally used
        wrong_ratio = torch.exp(logprobs - proximal_logprobs_t)
        assert not torch.allclose(actual_ppo_ratio, wrong_ratio, atol=1e-5), \
            "PPO ratio appears to be using proximal_logprobs_t - this breaks SDP!"

    def test_behav_imp_weight_uses_proximal_t(self):
        """CRITICAL: Verify behavioral importance weight uses proximal_t when provided.

        This is the other half of SDP - the behavioral importance weight MUST use
        proximal_logprobs_t (π_proximal_t / π_behavior) instead of proximal_logprobs.

        This is what reduces variance in SDP.
        """
        batch_size, seq_len = 1, 3

        logprobs = torch.tensor([[1.0, 2.0, 3.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        proximal_logprobs = torch.tensor([[0.5, 1.0, 1.5]])
        proximal_logprobs_t = torch.tensor([[0.3, 0.6, 0.9]])  # Different from proximal

        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # With proximal_t (SDP)
        loss_sdp, stats_sdp = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            proximal_logprobs_t=proximal_logprobs_t,
        )

        # CRITICAL ASSERTION: behav_imp_weight must use proximal_t
        expected_behav_weight = torch.exp(proximal_logprobs_t - old_logprobs)
        actual_behav_weight = stats_sdp["behave_imp_weight"]

        assert torch.allclose(actual_behav_weight, expected_behav_weight, atol=1e-5), \
            f"Behavioral importance weight MUST use proximal_logprobs_t!\n" \
            f"Expected: {expected_behav_weight}\nActual: {actual_behav_weight}"

        # Without proximal_t (standard decoupled), it should use proximal_logprobs
        loss_dec, stats_dec = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            proximal_logprobs_t=None,
        )

        expected_behav_weight_dec = torch.exp(proximal_logprobs - old_logprobs)
        actual_behav_weight_dec = stats_dec["behave_imp_weight"]

        assert torch.allclose(actual_behav_weight_dec, expected_behav_weight_dec, atol=1e-5), \
            "Without proximal_t, behav_imp_weight should use proximal_logprobs"

    def test_sdp_selective_replacement(self):
        """Verify SDP only replaces π_proximal in behav_imp_weight, not in PPO ratio.

        This test ensures the complete correctness of SDP's selective replacement:
        1. PPO ratio = exp(logprobs - proximal_logprobs)  [UNCHANGED]
        2. behav_imp_weight = exp(proximal_t - old_logprobs)  [CHANGED]

        The final loss is: clipped_ppo_loss * behav_imp_weight
        """
        batch_size, seq_len = 2, 3

        # Use simple values for easy verification
        logprobs = torch.log(torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
        old_logprobs = torch.log(torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
        proximal_logprobs = torch.log(torch.tensor([[1.8, 2.8, 3.8], [4.8, 5.8, 6.8]]))
        proximal_logprobs_t = torch.log(torch.tensor([[1.2, 1.5, 1.8], [2.0, 2.5, 3.0]]))

        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stats = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            proximal_logprobs_t=proximal_logprobs_t,
        )

        # Verify PPO ratio
        expected_ppo_ratio = torch.exp(logprobs - proximal_logprobs)
        assert torch.allclose(stats["importance_weight"], expected_ppo_ratio, atol=1e-5)

        # Verify behavioral importance weight
        expected_behav_weight = torch.exp(proximal_logprobs_t - old_logprobs)
        assert torch.allclose(stats["behave_imp_weight"], expected_behav_weight, atol=1e-5)

        # Verify they are different (selective replacement worked)
        assert not torch.allclose(
            stats["importance_weight"],
            stats["behave_imp_weight"],
            atol=1e-5
        ), "PPO ratio and behav_imp_weight should be different with different proximal values"

    def test_backward_compatibility_without_proximal_t(self):
        """Verify backward compatibility: when proximal_t=None, behav_imp_weight uses proximal.

        This ensures that disabling SDP (enable_segment_wise_ppo=False) reverts to
        standard decoupled PPO behavior.
        """
        batch_size, seq_len = 1, 4

        logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Without proximal_t (backward compatibility)
        loss, stats = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            proximal_logprobs_t=None,
        )

        # Both PPO ratio and behav_imp_weight should use proximal_logprobs
        expected_ppo_ratio = torch.exp(logprobs - proximal_logprobs)
        expected_behav_weight = torch.exp(proximal_logprobs - old_logprobs)

        assert torch.allclose(stats["importance_weight"], expected_ppo_ratio, atol=1e-5)
        assert torch.allclose(stats["behave_imp_weight"], expected_behav_weight, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
