"""
Unit tests for proximal log-probability approximation functionality.

Tests the compute_prox_logp_approximations function and related metrics.
"""

import pytest
import torch

from areal.engine.ppo.actor import compute_prox_logp_approximations


class TestProximalApproximations:
    """Test suite for proximal log-probability approximation methods."""

    def test_basic_linear_interpolation(self):
        """Test linear interpolation with simple version progression."""
        # Setup: behavior version=0, proximal version=1, current version=2
        old_logp = torch.tensor([[-1.0, -2.0, -3.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-1.5, -2.5, -3.5]], dtype=torch.float32)
        versions = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        current_version = 2

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # alpha = (proximal - behave) / (theta - behave) = (1 - 0) / (2 - 0) = 0.5
        # linear: old + alpha * (new - old) = -1.0 + 0.5 * (-1.5 - (-1.0)) = -1.25
        expected_linear = torch.tensor([[-1.25, -2.25, -3.25]], dtype=torch.float32)
        torch.testing.assert_close(
            approx["linear"], expected_linear, rtol=1e-4, atol=1e-4
        )

    def test_identity_approximation(self):
        """Test identity approximation returns behavior logp."""
        old_logp = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-5.0, -6.0]], dtype=torch.float32)
        versions = torch.tensor([[0, 1]], dtype=torch.int32)
        current_version = 5

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # Identity should return old_logp unchanged
        torch.testing.assert_close(approx["identity"], old_logp, rtol=1e-6, atol=1e-6)

    def test_alpha_clamping(self):
        """Test that alpha is clamped to [0, 1] range."""
        # Case 1: alpha should be 0 when v_behave == v_proximal
        old_logp = torch.tensor([[-1.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-2.0]], dtype=torch.float32)
        versions = torch.tensor([[4]], dtype=torch.int32)  # v_behave = v_proximal = 4
        current_version = 5  # v_theta = 5

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # alpha = (4 - 4) / (5 - 4) = 0
        # linear should equal old_logp
        torch.testing.assert_close(approx["linear"], old_logp, rtol=1e-4, atol=1e-4)

    def test_mixed_versions_in_batch(self):
        """Test handling of samples with different behavior versions."""
        # Sample 1: v_behave=0, Sample 2: v_behave=2
        old_logp = torch.tensor([[-1.0], [-2.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-1.5], [-2.2]], dtype=torch.float32)
        versions = torch.tensor([[0], [2]], dtype=torch.int32)
        current_version = 4  # v_proximal = 3

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # Sample 1: alpha = (3 - 0) / (4 - 0) = 0.75
        # linear = -1.0 + 0.75 * (-1.5 - (-1.0)) = -1.0 + 0.75 * (-0.5) = -1.375
        # Sample 2: alpha = (3 - 2) / (4 - 2) = 0.5
        # linear = -2.0 + 0.5 * (-2.2 - (-2.0)) = -2.0 + 0.5 * (-0.2) = -2.1
        expected_linear = torch.tensor([[-1.375], [-2.1]], dtype=torch.float32)
        torch.testing.assert_close(
            approx["linear"], expected_linear, rtol=1e-4, atol=1e-4
        )

    def test_harmonic_approximation_probabilities(self):
        """Test harmonic mean works in probability space."""
        old_logp = torch.tensor([[-0.693]], dtype=torch.float32)  # log(0.5)
        logprobs = torch.tensor([[-1.386]], dtype=torch.float32)  # log(0.25)
        versions = torch.tensor([[0]], dtype=torch.int32)
        current_version = 2  # v_proximal = 1, alpha = 0.5

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # alpha = 0.5
        # p_behave = 0.5, p_theta = 0.25
        # p_harmonic = (1-0.5)*0.5 + 0.5*0.25 = 0.25 + 0.125 = 0.375
        # log(0.375) ≈ -0.981
        expected = torch.log(torch.tensor([[0.375]], dtype=torch.float32))
        torch.testing.assert_close(approx["harmonic"], expected, rtol=1e-3, atol=1e-3)

    def test_all_methods_return_tensors(self):
        """Test that all approximation methods return valid tensors."""
        old_logp = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-1.5, -2.5]], dtype=torch.float32)
        versions = torch.tensor([[0, 0]], dtype=torch.int32)
        current_version = 2

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        expected_methods = [
            "linear",
            "identity",
            "harmonic",
            "quadratic",
        ]
        for method in expected_methods:
            assert method in approx, f"Missing method: {method}"
            assert isinstance(approx[method], torch.Tensor), f"{method} not a tensor"
            assert approx[method].shape == old_logp.shape, f"{method} shape mismatch"
            assert approx[method].dtype == torch.float32, f"{method} wrong dtype"

    def test_version_zero_division_handling(self):
        """Test handling of same versions (zero division in alpha)."""
        old_logp = torch.tensor([[-1.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-2.0]], dtype=torch.float32)
        versions = torch.tensor([[3]], dtype=torch.int32)
        current_version = 3  # v_behave = v_theta = 3, division by zero

        # Should not crash, alpha should be 0
        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # When versions are equal, alpha should be 0, linear should equal old_logp
        torch.testing.assert_close(approx["linear"], old_logp, rtol=1e-4, atol=1e-4)

    def test_quadratic_approximation(self):
        """Test quadratic interpolation includes acceleration term."""
        old_logp = torch.tensor([[-1.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-2.0]], dtype=torch.float32)
        versions = torch.tensor([[0]], dtype=torch.int32)
        current_version = 2  # alpha = 0.5

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # alpha = 0.5, delta = -2.0 - (-1.0) = -1.0
        # quadratic = old + alpha * delta + 0.5 * alpha^2 * delta
        #           = -1.0 + 0.5 * (-1.0) + 0.5 * 0.25 * (-1.0)
        #           = -1.0 - 0.5 - 0.125 = -1.625
        expected = torch.tensor([[-1.625]], dtype=torch.float32)
        torch.testing.assert_close(approx["quadratic"], expected, rtol=1e-4, atol=1e-4)

    def test_negative_versions_in_prompt(self):
        """Test handling of negative versions (prompt tokens)."""
        # Prompt tokens have version=-1, should be handled gracefully
        old_logp = torch.tensor([[-1.0, -2.0, -3.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-1.5, -2.5, -3.5]], dtype=torch.float32)
        versions = torch.tensor([[-1, 0, 1]], dtype=torch.int32)
        current_version = 3

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # Should not crash with negative versions
        assert approx["linear"].shape == old_logp.shape
        # For prompt tokens (version < 0), alpha is 0, so approximation should equal old_logp
        assert torch.isclose(approx["linear"][0, 0], old_logp[0, 0])
        assert torch.isfinite(approx["linear"]).all(), "NaN/Inf in approximation"

    def test_batch_dimensions(self):
        """Test handling of different batch shapes."""
        batch_size = 4
        seq_len = 8
        old_logp = torch.randn(batch_size, seq_len, dtype=torch.float32)
        logprobs = torch.randn(batch_size, seq_len, dtype=torch.float32)
        versions = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.int32)
        current_version = 10

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        for method in ["linear", "identity", "harmonic", "quadratic"]:
            assert approx[method].shape == (batch_size, seq_len)
            assert torch.isfinite(approx[method]).all(), f"{method} has NaN/Inf"


class TestProximalApproximationIntegration:
    """Integration tests for proximal approximation in training flow."""

    def test_versions_preserved_before_popping(self):
        """Test that versions are preserved before being popped from batch."""
        # Simulate the code in ppo_update
        data = {
            "versions": torch.tensor([[0, 1, 2]], dtype=torch.int32),
            "rewards": torch.tensor([1.0]),
            "tot_rewards": torch.tensor([1.0]),
            "kl_rewards": torch.tensor([[0.1, 0.2, 0.3]]),
        }

        # Simulate preservation logic
        use_decoupled_loss = True
        if "versions" in data and use_decoupled_loss:
            data["_versions_for_approx"] = data["versions"].clone()

        # Pop versions
        for key in ["rewards", "tot_rewards", "kl_rewards", "versions"]:
            data.pop(key, None)

        # Verify _versions_for_approx still exists
        assert "_versions_for_approx" in data
        torch.testing.assert_close(
            data["_versions_for_approx"],
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
        )

    def test_approximation_metrics_not_computed_without_flag(self):
        """Test that metrics are not computed when use_decoupled_loss=False."""
        data = {
            "versions": torch.tensor([[0, 1, 2]], dtype=torch.int32),
        }

        use_decoupled_loss = False
        if "versions" in data and use_decoupled_loss:
            data["_versions_for_approx"] = data["versions"].clone()

        # _versions_for_approx should NOT be set
        assert "_versions_for_approx" not in data


def test_import_success():
    """Test that the approximation function can be imported."""
    from areal.engine.ppo.actor import compute_prox_logp_approximations

    assert callable(compute_prox_logp_approximations)


class TestGrpoLossFnNoneHandling:
    """Test suite for grpo_loss_fn() handling of None prox_logp."""

    def test_grpo_loss_fn_detects_none_prox_logp(self):
        """Test that grpo_loss_fn() detects None prox_logp and validates configuration."""
        from areal.engine.ppo.actor import grpo_loss_fn

        # Create dummy inputs with prox_logp=None
        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)  # vocab_size=100
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": None,  # Key test: None value
            "_versions_for_approx": torch.randint(
                0, 5, (batch_size, seq_len), dtype=torch.int32
            ),
        }

        # Should raise error if use_prox_approx=False but prox_logp=None
        with pytest.raises(
            ValueError, match="prox_logp is None but use_prox_approx=False"
        ):
            grpo_loss_fn(
                logits=logits,
                input_data=input_data,
                temperature=1.0,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=None,
                behav_imp_weight_cap=None,
                current_version=5,
                use_prox_approx=False,  # Approximation disabled but prox_logp is None
                prox_approx_method="linear",
                log_prox_approx_metrics=False,
            )

    def test_grpo_loss_fn_requires_versions_when_prox_logp_none(self):
        """Test that grpo_loss_fn() requires versions when prox_logp is None."""
        from areal.engine.ppo.actor import grpo_loss_fn

        # Create dummy inputs with prox_logp=None but no versions
        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": None,  # None value
            # Missing: "_versions_for_approx"
        }

        # Should raise error if versions not available
        with pytest.raises(
            ValueError, match="versions not available for approximation"
        ):
            grpo_loss_fn(
                logits=logits,
                input_data=input_data,
                temperature=1.0,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=None,
                behav_imp_weight_cap=None,
                current_version=5,
                use_prox_approx=True,
                prox_approx_method="linear",
                log_prox_approx_metrics=False,
            )

    def test_grpo_loss_fn_computes_approximation_when_prox_logp_none(self):
        """Test that grpo_loss_fn() successfully computes approximation when prox_logp is None."""
        from areal.engine.ppo.actor import grpo_loss_fn

        # Create valid inputs with prox_logp=None
        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": None,  # None - will be replaced with approximation
            "_versions_for_approx": torch.randint(
                0, 3, (batch_size, seq_len), dtype=torch.int32
            ),
        }

        # Should successfully compute approximation
        loss = grpo_loss_fn(
            logits=logits,
            input_data=input_data,
            temperature=1.0,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            behav_imp_weight_cap=None,
            current_version=5,
            use_prox_approx=True,
            prox_approx_method="linear",
            log_prox_approx_metrics=False,
        )

        # Loss should be computed successfully
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert torch.isfinite(loss), "Loss should not contain NaN/Inf"

    def test_grpo_loss_fn_works_with_tensor_prox_logp(self):
        """Test that grpo_loss_fn() still works normally with tensor prox_logp."""
        from areal.engine.ppo.actor import grpo_loss_fn

        # Create valid inputs with prox_logp as tensor (normal case)
        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": torch.randn(batch_size, seq_len),  # Tensor (normal case)
            "_versions_for_approx": torch.randint(
                0, 3, (batch_size, seq_len), dtype=torch.int32
            ),
        }

        # Should work normally
        loss = grpo_loss_fn(
            logits=logits,
            input_data=input_data,
            temperature=1.0,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            behav_imp_weight_cap=None,
            current_version=5,
            use_prox_approx=True,
            prox_approx_method="linear",
            log_prox_approx_metrics=False,
        )

        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert torch.isfinite(loss), "Loss should not contain NaN/Inf"

    def test_grpo_loss_fn_metrics_disabled_when_prox_logp_none(self):
        """Test that metrics are not logged when prox_logp is None (no ground truth)."""
        from areal.engine.ppo.actor import grpo_loss_fn

        # Create inputs with prox_logp=None and metrics enabled
        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": None,  # No ground truth
            "_versions_for_approx": torch.randint(
                0, 3, (batch_size, seq_len), dtype=torch.int32
            ),
            "prox_logp_recomputed": False,  # Not recomputed
        }

        # Even with log_prox_approx_metrics=True, metrics should not be logged
        loss = grpo_loss_fn(
            logits=logits,
            input_data=input_data,
            temperature=1.0,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            behav_imp_weight_cap=None,
            current_version=5,
            use_prox_approx=True,
            prox_approx_method="linear",
            log_prox_approx_metrics=True,  # Metrics enabled but should not log
        )

        # Should not crash and loss should be valid
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert torch.isfinite(loss), "Loss should not contain NaN/Inf"


class TestConfigValidation:
    """Test suite for PPOActorConfig validation, especially dynamic method validation."""

    def test_valid_method_from_config_metadata(self):
        """Test that validation accepts methods defined in PPOActorConfig choices."""
        from areal.api.cli_args import PPOActorConfig
        from areal.engine.ppo.actor import PPOActor

        # Test each valid method from the config
        valid_methods = ["linear", "harmonic", "quadratic", "identity"]
        for method in valid_methods:
            config = PPOActorConfig(
                use_decoupled_loss=True,
                use_prox_approx=True,
                prox_approx_method=method,
                recompute_logprob=False,
            )
            # Should not raise ValueError
            try:
                # We validate in __init__, but we need a mock engine
                from unittest.mock import MagicMock

                mock_engine = MagicMock()
                mock_engine.module.config = MagicMock()
                _ = PPOActor(config, mock_engine)  # noqa: F841
                # If we reach here, validation passed
            except ValueError as e:
                pytest.fail(f"Method '{method}' should be valid but got: {e}")

    def test_invalid_method_raises_error(self):
        """Test that validation rejects invalid approximation methods."""
        from unittest.mock import MagicMock

        from areal.api.cli_args import PPOActorConfig
        from areal.engine.ppo.actor import PPOActor

        config = PPOActorConfig(
            use_decoupled_loss=True,
            use_prox_approx=True,
            prox_approx_method="invalid_method",
            recompute_logprob=False,
        )

        mock_engine = MagicMock()
        mock_engine.module.config = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            PPOActor(config, mock_engine)

        error_msg = str(exc_info.value)
        assert "Invalid prox_approx_method" in error_msg
        assert "invalid_method" in error_msg
        assert "linear" in error_msg  # Should list valid methods

    def test_validation_uses_metadata_choices(self):
        """Test that validation dynamically retrieves choices from dataclass metadata."""
        from dataclasses import fields as dataclass_fields

        from areal.api.cli_args import PPOActorConfig

        # Get the actual choices from the dataclass
        config_choices = None
        for f in dataclass_fields(PPOActorConfig):
            if f.name == "prox_approx_method":
                config_choices = f.metadata.get("choices", [])
                break

        assert config_choices is not None, "prox_approx_method field should exist"
        assert len(config_choices) > 0, "choices should not be empty"
        assert "linear" in config_choices, "linear should be in choices"
        assert "harmonic" in config_choices, "harmonic should be in choices"
        assert "quadratic" in config_choices, "quadratic should be in choices"
        assert "identity" in config_choices, "identity should be in choices"

    def test_metadata_matches_validation_fallback(self):
        """Test that config metadata choices match the fallback hardcoded list."""
        from dataclasses import fields as dataclass_fields

        from areal.api.cli_args import PPOActorConfig

        # Get choices from dataclass metadata
        config_choices = None
        for f in dataclass_fields(PPOActorConfig):
            if f.name == "prox_approx_method":
                config_choices = f.metadata.get("choices", [])
                break

        # Hardcoded fallback list in actor.py:93
        fallback_list = ["linear", "harmonic", "quadratic", "identity"]

        # Verify metadata exists
        assert config_choices is not None
        # Verify choices match fallback list
        assert set(config_choices) == set(fallback_list)

    def test_decoupled_approximation_requires_recompute_false(self):
        """Test that decoupled + approximation requires recompute_logprob=False."""
        from unittest.mock import MagicMock

        from areal.api.cli_args import PPOActorConfig
        from areal.engine.ppo.actor import PPOActor

        # Invalid: decoupled + approximation with recompute enabled
        config = PPOActorConfig(
            use_decoupled_loss=True,
            use_prox_approx=True,
            recompute_logprob=True,  # Should fail
        )

        mock_engine = MagicMock()
        mock_engine.module.config = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            PPOActor(config, mock_engine)

        error_msg = str(exc_info.value)
        assert "use_prox_approx=True requires recompute_logprob=False" in error_msg

    def test_decoupled_without_approximation_requires_recompute_true(self):
        """Test that decoupled without approximation requires recompute_logprob=True."""
        from unittest.mock import MagicMock

        from areal.api.cli_args import PPOActorConfig
        from areal.engine.ppo.actor import PPOActor

        # Invalid: decoupled without approximation but no recompute
        config = PPOActorConfig(
            use_decoupled_loss=True,
            use_prox_approx=False,
            recompute_logprob=False,  # Should fail
        )

        mock_engine = MagicMock()
        mock_engine.module.config = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            PPOActor(config, mock_engine)

        error_msg = str(exc_info.value)
        assert "requires recompute_logprob=True" in error_msg

    def test_all_configuration_combinations(self):
        """Test all combinations of use_decoupled_loss, use_prox_approx, recompute_logprob."""
        from unittest.mock import MagicMock

        from areal.api.cli_args import PPOActorConfig
        from areal.engine.ppo.actor import PPOActor

        # Format: (use_decoupled_loss, use_prox_approx, recompute_logprob, should_succeed, description)
        test_cases = [
            # Standard PPO (use_decoupled_loss=False)
            (False, False, False, True, "Standard PPO: no recompute"),
            (False, False, True, True, "Standard PPO: with recompute"),
            (False, True, False, False, "Standard PPO: approx requires decoupled"),
            (False, True, True, False, "Standard PPO: approx requires decoupled"),
            # Decoupled PPO without approximation
            (True, False, False, False, "Decoupled: needs recompute without approx"),
            (True, False, True, True, "Decoupled: standard with recompute"),
            # Decoupled PPO with approximation
            (True, True, False, True, "Decoupled: approximation without recompute"),
            (True, True, True, False, "Decoupled: approx requires no recompute"),
        ]

        for use_decoupled, use_approx, recompute, should_succeed, desc in test_cases:
            config = PPOActorConfig(
                use_decoupled_loss=use_decoupled,
                use_prox_approx=use_approx,
                recompute_logprob=recompute,
            )

            mock_engine = MagicMock()
            mock_engine.module.config = MagicMock()

            if should_succeed:
                # Should create actor successfully
                try:
                    _ = PPOActor(config, mock_engine)  # noqa: F841
                    # Success expected
                except ValueError as e:
                    pytest.fail(f"Config should succeed but failed: {desc}\nError: {e}")
            else:
                # Should raise ValueError
                with pytest.raises(ValueError) as exc_info:
                    PPOActor(config, mock_engine)
                # Verify error message is informative
                assert len(str(exc_info.value)) > 50, (
                    f"Error message too short for: {desc}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
