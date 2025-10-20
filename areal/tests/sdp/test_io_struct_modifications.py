"""Unit tests for ModelResponse modifications (segment-wise PPO feature).

This module provides test coverage for the proximal_logprobs_t field added
to ModelResponse for segment-wise decoupled PPO.
All tests run without GPU.
"""

import pytest

from areal.api.io_struct import ModelResponse


class TestModelResponseProximalLogprobsField:
    """Test proximal_logprobs_t field in ModelResponse."""

    def test_proximal_logprobs_t_field_exists(self):
        """Test that proximal_logprobs_t field exists."""
        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
        )

        assert hasattr(response, 'proximal_logprobs_t')

    def test_proximal_logprobs_t_defaults_to_empty_list(self):
        """Test that proximal_logprobs_t defaults to empty list."""
        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
        )

        assert response.proximal_logprobs_t == []

    def test_proximal_logprobs_t_can_be_set_in_constructor(self):
        """Test that proximal_logprobs_t can be set via constructor."""
        proximal_logprobs = [0.1, 0.2, 0.3]

        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
            proximal_logprobs_t=proximal_logprobs,
        )

        assert response.proximal_logprobs_t == proximal_logprobs

    def test_proximal_logprobs_t_can_be_modified_after_creation(self):
        """Test that proximal_logprobs_t can be modified after creation."""
        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
        )

        response.proximal_logprobs_t = [0.1, 0.2, 0.3]

        assert response.proximal_logprobs_t == [0.1, 0.2, 0.3]

    def test_proximal_logprobs_t_with_empty_list(self):
        """Test setting proximal_logprobs_t to empty list."""
        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
            proximal_logprobs_t=[],
        )

        assert response.proximal_logprobs_t == []

    def test_proximal_logprobs_t_length_independent_of_outputs(self):
        """Test that proximal_logprobs_t length is independent of output_tokens."""
        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],  # 3 tokens
            proximal_logprobs_t=[0.1, 0.2],  # 2 logprobs (mismatched - allowed)
        )

        # Should allow different lengths (up to application logic to validate)
        assert len(response.proximal_logprobs_t) == 2
        assert len(response.output_tokens) == 3

    def test_proximal_logprobs_t_with_output_logprobs(self):
        """Test that proximal_logprobs_t coexists with output_logprobs."""
        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
            output_logprobs=[0.5, 0.6, 0.7],
            proximal_logprobs_t=[0.1, 0.2, 0.3],
        )

        # Both fields should be independent
        assert response.output_logprobs == [0.5, 0.6, 0.7]
        assert response.proximal_logprobs_t == [0.1, 0.2, 0.3]

    def test_proximal_logprobs_t_with_output_versions(self):
        """Test that proximal_logprobs_t works with output_versions."""
        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
            output_versions=[1, 1, 1],
            proximal_logprobs_t=[0.1, 0.2, 0.3],
        )

        # All segment-wise PPO fields should work together
        assert response.output_versions == [1, 1, 1]
        assert response.proximal_logprobs_t == [0.1, 0.2, 0.3]


class TestModelResponseBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_can_create_response_without_proximal_logprobs_t(self):
        """Test that ModelResponse works without specifying proximal_logprobs_t."""
        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
            output_logprobs=[0.5, 0.6, 0.7],
        )

        # Should work without error
        assert response.output_tokens == [4, 5, 6]
        assert response.proximal_logprobs_t == []  # Defaults to empty

    def test_existing_fields_unchanged(self):
        """Test that existing fields still work as expected."""
        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
            output_logprobs=[0.5, 0.6, 0.7],
            output_versions=[1, 1, 1],
            stop_reason="length",
        )

        # Existing fields should all work
        assert response.input_tokens == [1, 2, 3]
        assert response.output_tokens == [4, 5, 6]
        assert response.output_logprobs == [0.5, 0.6, 0.7]
        assert response.output_versions == [1, 1, 1]
        assert response.stop_reason == "length"


class TestModelResponseDataStructure:
    """Test ModelResponse as a data structure."""

    def test_proximal_logprobs_t_is_list_type(self):
        """Test that proximal_logprobs_t is a list."""
        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
        )

        assert isinstance(response.proximal_logprobs_t, list)

    def test_proximal_logprobs_t_can_hold_floats(self):
        """Test that proximal_logprobs_t can hold float values."""
        logprobs = [0.123, -1.456, 2.789]

        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
            proximal_logprobs_t=logprobs,
        )

        assert response.proximal_logprobs_t == logprobs
        for logprob in response.proximal_logprobs_t:
            assert isinstance(logprob, float)

    def test_proximal_logprobs_t_with_negative_values(self):
        """Test proximal_logprobs_t with negative values (common for log probs)."""
        logprobs = [-0.5, -1.2, -0.8]

        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5, 6],
            proximal_logprobs_t=logprobs,
        )

        assert response.proximal_logprobs_t == logprobs

    def test_proximal_logprobs_t_with_large_list(self):
        """Test proximal_logprobs_t with large list."""
        logprobs = [0.1 * i for i in range(1000)]

        response = ModelResponse(
            input_tokens=[1, 2, 3],
            output_tokens=list(range(1000)),
            proximal_logprobs_t=logprobs,
        )

        assert len(response.proximal_logprobs_t) == 1000


# Parametrized tests
@pytest.mark.parametrize("logprobs", [
    [],
    [0.1],
    [0.1, 0.2],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [-0.5, -1.2, -0.8],
    [0.0, 0.0, 0.0],
])
def test_parametrized_proximal_logprobs_values(logprobs):
    """Test proximal_logprobs_t with various values."""
    response = ModelResponse(
        input_tokens=[1, 2, 3],
        output_tokens=[4, 5, 6],
        proximal_logprobs_t=logprobs,
    )

    assert response.proximal_logprobs_t == logprobs


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--cov=areal.api.io_struct"])
