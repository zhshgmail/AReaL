"""Unit tests for ProximalRecomputer.

This module provides comprehensive test coverage for the ProximalRecomputer class,
which handles proximal logprob recomputation for v-1 samples in segment-wise PPO.
All tests run without GPU using mocked inference engines.
"""

from unittest.mock import Mock, call, patch

import pytest

from areal.api.proximal_recomputer import ProximalRecomputer


class TestProximalRecomputerInitialization:
    """Test ProximalRecomputer initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        mock_engine = Mock()
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        assert recomputer.inference_engine is mock_engine
        assert recomputer.logger is mock_logger

    def test_initialization_with_none_engine(self):
        """Test initialization with None engine (should work for testing)."""
        mock_logger = Mock()

        recomputer = ProximalRecomputer(None, mock_logger)

        assert recomputer.inference_engine is None
        assert recomputer.logger is mock_logger


class TestRecomputeForSample:
    """Test recompute_for_sample method."""

    def test_recomputes_proximal_for_v_minus_1_sample(self):
        """Test that v-1 samples get recomputed."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_engine.recompute_output_logprobs_sync.return_value = [0.1, 0.2, 0.3]
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        trajectory = {
            "input_ids": [1, 2, 3, 4, 5, 6],  # 3 prompt + 3 generated
            "output_version": [4, 4, 4],  # v-1 sample
            "proximal_logprobs_t": [0.9, 0.8, 0.7],  # Old values
        }
        prompt_len = 3

        recomputer.recompute_for_sample(trajectory, prompt_len)

        # Should call recompute with correct arguments
        mock_engine.recompute_output_logprobs_sync.assert_called_once_with(
            input_ids=[1, 2, 3, 4, 5, 6],
            start_index=prompt_len,
        )
        # Should update proximal_logprobs_t
        assert trajectory["proximal_logprobs_t"] == [0.1, 0.2, 0.3]

    def test_skips_current_version_samples(self):
        """Test that current version samples are skipped."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        trajectory = {
            "input_ids": [1, 2, 3, 4, 5, 6],
            "output_version": [5, 5, 5],  # Current version
            "proximal_logprobs_t": [0.1, 0.2, 0.3],
        }
        prompt_len = 3

        recomputer.recompute_for_sample(trajectory, prompt_len)

        # Should NOT call recompute
        mock_engine.recompute_output_logprobs_sync.assert_not_called()
        # proximal_logprobs_t should remain unchanged
        assert trajectory["proximal_logprobs_t"] == [0.1, 0.2, 0.3]

    def test_skips_v_minus_2_samples(self):
        """Test that v-2 samples are skipped (too stale)."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        trajectory = {
            "input_ids": [1, 2, 3, 4, 5, 6],
            "output_version": [3, 3, 3],  # v-2 (too stale)
            "proximal_logprobs_t": [0.1, 0.2, 0.3],
        }
        prompt_len = 3

        recomputer.recompute_for_sample(trajectory, prompt_len)

        # Should NOT call recompute
        mock_engine.recompute_output_logprobs_sync.assert_not_called()

    def test_handles_missing_output_version(self):
        """Test handling when output_version is missing."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        trajectory = {
            "input_ids": [1, 2, 3, 4, 5, 6],
            # output_version missing
            "proximal_logprobs_t": [0.1, 0.2, 0.3],
        }
        prompt_len = 3

        # Should not crash
        recomputer.recompute_for_sample(trajectory, prompt_len)

        # Should not call recompute
        mock_engine.recompute_output_logprobs_sync.assert_not_called()

    def test_handles_empty_output_version(self):
        """Test handling when output_version is empty."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        trajectory = {
            "input_ids": [1, 2, 3],  # Only prompt
            "output_version": [],  # Empty
            "proximal_logprobs_t": [],
        }
        prompt_len = 3

        # Should not crash
        recomputer.recompute_for_sample(trajectory, prompt_len)

        # Should not call recompute
        mock_engine.recompute_output_logprobs_sync.assert_not_called()

    def test_uses_correct_prompt_length(self):
        """Test that prompt length is correctly used for recomputation."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_engine.recompute_output_logprobs_sync.return_value = [0.1, 0.2]
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        trajectory = {
            "input_ids": [1, 2, 3, 4, 5],  # 3 prompt + 2 generated
            "output_version": [4, 4],  # v-1
            "proximal_logprobs_t": [0.9, 0.8],
        }
        prompt_len = 3

        recomputer.recompute_for_sample(trajectory, prompt_len)

        # start_index should be prompt_len
        mock_engine.recompute_output_logprobs_sync.assert_called_once_with(
            input_ids=[1, 2, 3, 4, 5],
            start_index=3,
        )


class TestRecomputeBatch:
    """Test recompute_batch method."""

    def test_recomputes_multiple_v_minus_1_samples(self):
        """Test batch recomputation with multiple v-1 samples."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_engine.recompute_output_logprobs_sync.side_effect = [
            [0.1, 0.2],  # First sample
            [0.3, 0.4],  # Second sample
        ]
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        batch = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "output_version": [4, 4],  # v-1
                "proximal_logprobs_t": [0.9, 0.8],
            },
            {
                "input_ids": [6, 7, 8, 9, 10],
                "output_version": [4, 4],  # v-1
                "proximal_logprobs_t": [0.7, 0.6],
            },
        ]
        prompt_len = 3

        recomputer.recompute_batch(batch, prompt_len)

        # Should call recompute twice
        assert mock_engine.recompute_output_logprobs_sync.call_count == 2
        # Should update both samples
        assert batch[0]["proximal_logprobs_t"] == [0.1, 0.2]
        assert batch[1]["proximal_logprobs_t"] == [0.3, 0.4]

    def test_recomputes_only_v_minus_1_in_mixed_batch(self):
        """Test that only v-1 samples are recomputed in mixed batch."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_engine.recompute_output_logprobs_sync.return_value = [0.1, 0.2]
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        batch = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "output_version": [5, 5],  # Current version
                "proximal_logprobs_t": [0.9, 0.8],
            },
            {
                "input_ids": [6, 7, 8, 9, 10],
                "output_version": [4, 4],  # v-1
                "proximal_logprobs_t": [0.7, 0.6],
            },
            {
                "input_ids": [11, 12, 13, 14, 15],
                "output_version": [3, 3],  # v-2
                "proximal_logprobs_t": [0.5, 0.4],
            },
        ]
        prompt_len = 3

        recomputer.recompute_batch(batch, prompt_len)

        # Should call recompute only once (for v-1 sample)
        assert mock_engine.recompute_output_logprobs_sync.call_count == 1
        # Only second sample should be updated
        assert batch[0]["proximal_logprobs_t"] == [0.9, 0.8]  # Unchanged
        assert batch[1]["proximal_logprobs_t"] == [0.1, 0.2]  # Updated
        assert batch[2]["proximal_logprobs_t"] == [0.5, 0.4]  # Unchanged

    def test_handles_empty_batch(self):
        """Test handling of empty batch."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        batch = []

        # Should not crash
        recomputer.recompute_batch(batch, prompt_len=3)

        # Should not call recompute
        mock_engine.recompute_output_logprobs_sync.assert_not_called()

    def test_handles_all_current_version_batch(self):
        """Test batch with all current version samples."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        batch = [
            {"input_ids": [1, 2, 3, 4], "output_version": [5], "proximal_logprobs_t": [0.9]},
            {"input_ids": [5, 6, 7, 8], "output_version": [5], "proximal_logprobs_t": [0.8]},
        ]

        recomputer.recompute_batch(batch, prompt_len=3)

        # Should not call recompute
        mock_engine.recompute_output_logprobs_sync.assert_not_called()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_recomputation_failure(self):
        """Test handling when recomputation fails."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        mock_engine.recompute_output_logprobs_sync.side_effect = RuntimeError("Server error")
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        trajectory = {
            "input_ids": [1, 2, 3, 4, 5],
            "output_version": [4, 4],  # v-1
            "proximal_logprobs_t": [0.9, 0.8],
        }
        prompt_len = 3

        # Should raise the exception
        with pytest.raises(RuntimeError, match="Server error"):
            recomputer.recompute_for_sample(trajectory, prompt_len)

    def test_handles_version_check_failure(self):
        """Test handling when get_version fails."""
        mock_engine = Mock()
        mock_engine.get_version.side_effect = RuntimeError("Cannot get version")
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        trajectory = {
            "input_ids": [1, 2, 3, 4, 5],
            "output_version": [4, 4],
            "proximal_logprobs_t": [0.9, 0.8],
        }

        # Should raise the exception
        with pytest.raises(RuntimeError, match="Cannot get version"):
            recomputer.recompute_for_sample(trajectory, prompt_len=3)

    def test_handles_mismatched_lengths(self):
        """Test handling when returned logprobs have wrong length."""
        mock_engine = Mock()
        mock_engine.get_version.return_value = 5
        # Return wrong number of logprobs
        mock_engine.recompute_output_logprobs_sync.return_value = [0.1]  # Should be 2
        mock_logger = Mock()

        recomputer = ProximalRecomputer(mock_engine, mock_logger)

        trajectory = {
            "input_ids": [1, 2, 3, 4, 5],
            "output_version": [4, 4],  # 2 tokens
            "proximal_logprobs_t": [0.9, 0.8],
        }
        prompt_len = 3

        # Should update despite length mismatch (engine is authoritative)
        recomputer.recompute_for_sample(trajectory, prompt_len)

        assert trajectory["proximal_logprobs_t"] == [0.1]


# Parametrized tests
@pytest.mark.parametrize("current_version,sample_version,should_recompute", [
    (5, 5, False),  # Current version - skip
    (5, 4, True),   # v-1 - recompute
    (5, 3, False),  # v-2 - skip
    (5, 2, False),  # v-3 - skip
    (10, 9, True),  # v-1 - recompute
    (10, 8, False), # v-2 - skip
])
def test_parametrized_recomputation_logic(current_version, sample_version, should_recompute):
    """Test recomputation logic with various version combinations."""
    mock_engine = Mock()
    mock_engine.get_version.return_value = current_version
    mock_engine.recompute_output_logprobs_sync.return_value = [0.1, 0.2]
    mock_logger = Mock()

    recomputer = ProximalRecomputer(mock_engine, mock_logger)

    trajectory = {
        "input_ids": [1, 2, 3, 4, 5],
        "output_version": [sample_version, sample_version],
        "proximal_logprobs_t": [0.9, 0.8],
    }

    recomputer.recompute_for_sample(trajectory, prompt_len=3)

    if should_recompute:
        mock_engine.recompute_output_logprobs_sync.assert_called_once()
        assert trajectory["proximal_logprobs_t"] == [0.1, 0.2]
    else:
        mock_engine.recompute_output_logprobs_sync.assert_not_called()
        assert trajectory["proximal_logprobs_t"] == [0.9, 0.8]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=areal.api.proximal_recomputer"])
