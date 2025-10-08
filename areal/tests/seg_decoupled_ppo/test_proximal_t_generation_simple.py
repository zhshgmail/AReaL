"""
Simple unit tests for proximal_logprobs_t generation logic.

These tests verify the core logic without needing full engine initialization.
"""

import pytest


def test_normal_generation_proximal_t_length():
    """Test that proximal_t has correct length in normal generation (no abort)."""
    # Simulate first iteration (normal generation)
    prompt_len = 5
    accumulated_output_tokens = []
    proximal_logprobs_t = []

    # Mock SGLang response
    output_logprobs = [1.0, 1.1, 1.2]  # 3 new tokens

    # Core logic from sglang_remote.py:agenerate()
    if len(accumulated_output_tokens) == 0:
        # First iteration: No previous output to update
        pass
    else:
        # Would update previous tokens here
        pass

    # Extend with new tokens
    proximal_logprobs_t.extend(output_logprobs)
    accumulated_output_tokens.extend([201, 202, 203])

    # Verify
    assert len(proximal_logprobs_t) == len(accumulated_output_tokens)
    assert proximal_logprobs_t == [1.0, 1.1, 1.2]


def test_abort_resume_proximal_t_update():
    """Test that abort-resume correctly updates proximal_t."""
    # Iteration 1: Generated 2 tokens at v5
    accumulated_output_tokens = [201, 202]
    proximal_logprobs_t = [1.0, 1.1]  # From first iteration
    prompt_len = 3

    # Iteration 2: Policy updated to v6, now recomputing
    # Mock SGLang response for iteration 2
    input_logprobs = [
        0.3,  # Last prompt token
        2.0,  # First output token under v6 (proximal_t!)
        2.1,  # Second output token under v6 (proximal_t!)
    ]
    output_logprobs = [1.2]  # New token

    # Core logic from sglang_remote.py:agenerate()
    if len(accumulated_output_tokens) == 0:
        pass
    else:
        # Abort-resume: Update proximal_t for previous tokens
        prev_output_len = len(accumulated_output_tokens)

        if len(input_logprobs) >= prev_output_len + 1:
            # Extract proximal_t for previous output tokens
            prev_output_proximal_t = input_logprobs[1:1+prev_output_len]

            # REPLACE existing values
            for i, logprob in enumerate(prev_output_proximal_t):
                if i < len(proximal_logprobs_t):
                    proximal_logprobs_t[i] = logprob

    # Extend with new token
    proximal_logprobs_t.extend(output_logprobs)
    accumulated_output_tokens.extend([203])

    # Verify
    assert len(proximal_logprobs_t) == len(accumulated_output_tokens)
    assert len(proximal_logprobs_t) == 3
    assert proximal_logprobs_t == [2.0, 2.1, 1.2]  # First 2 updated, last from generation


def test_multiple_abort_resume_proximal_t():
    """Test multiple abort-resume cycles maintain correct length."""
    # Iteration 1: v5, generated 2 tokens
    accumulated_output_tokens = [201, 202]
    proximal_logprobs_t = [1.0, 1.1]
    prompt_len = 2

    # Iteration 2: v6, update previous, generate 2 more
    input_logprobs_2 = [0.2, 2.0, 2.1]  # last prompt + 2 outputs under v6
    output_logprobs_2 = [1.2, 1.3]

    prev_output_len = len(accumulated_output_tokens)
    if len(input_logprobs_2) >= prev_output_len + 1:
        prev_output_proximal_t = input_logprobs_2[1:1+prev_output_len]
        for i, logprob in enumerate(prev_output_proximal_t):
            if i < len(proximal_logprobs_t):
                proximal_logprobs_t[i] = logprob

    proximal_logprobs_t.extend(output_logprobs_2)
    accumulated_output_tokens.extend([203, 204])

    # Now: [2.0, 2.1, 1.2, 1.3]
    assert len(proximal_logprobs_t) == 4

    # Iteration 3: v7, update all previous, generate 1 more
    input_logprobs_3 = [
        1.3,  # last output from iter 2
        3.0, 3.1, 3.2, 3.3  # All 4 outputs under v7
    ]
    output_logprobs_3 = [1.4]

    prev_output_len = len(accumulated_output_tokens)
    if len(input_logprobs_3) >= prev_output_len + 1:
        prev_output_proximal_t = input_logprobs_3[1:1+prev_output_len]
        for i, logprob in enumerate(prev_output_proximal_t):
            if i < len(proximal_logprobs_t):
                proximal_logprobs_t[i] = logprob

    proximal_logprobs_t.extend(output_logprobs_3)
    accumulated_output_tokens.extend([205])

    # Verify
    assert len(proximal_logprobs_t) == 5
    assert len(accumulated_output_tokens) == 5
    assert proximal_logprobs_t == [3.0, 3.1, 3.2, 3.3, 1.4]


def test_proximal_t_integration_with_rlvr():
    """Test that proximal_t can be used in rlvr.py without length mismatch."""
    # Simulate ModelResponse
    input_tokens = [101, 102, 103]  # prompt
    output_tokens = [201, 202]  # generated
    proximal_logprobs_t = [1.0, 1.1]  # same length as output
    input_len = len(input_tokens)

    # Simulate rlvr.py line 97
    seq = input_tokens + output_tokens
    proximal_t_full = [0.0] * input_len + proximal_logprobs_t

    # Verify no length mismatch
    assert len(seq) == len(proximal_t_full)
    assert len(seq) == 5
    assert proximal_t_full == [0.0, 0.0, 0.0, 1.0, 1.1]


def test_bug_scenario_would_fail():
    """
    This test demonstrates the BUG in the OLD code.

    OLD CODE (WRONG):
        proximal_logprobs_t.extend(input_logprobs[1:])  # Line 240
        ...
        proximal_logprobs_t.extend(output_logprobs)  # Line 254

    This would cause length mismatch!
    """
    # Iteration 1: Normal generation
    prompt_len = 5
    input_logprobs = [0.1, 0.2, 0.3, 0.4, 0.5]  # P=5
    output_logprobs = [1.0, 1.1, 1.2]  # N=3

    # OLD CODE (BUG):
    proximal_logprobs_t_buggy = []
    proximal_logprobs_t_buggy.extend(input_logprobs[1:])  # Adds 4 prompt logprobs
    accumulated_output_tokens = [201, 202, 203]
    # After loop:
    proximal_logprobs_t_buggy.extend(output_logprobs)  # Adds 3 output logprobs

    # BUG: Length mismatch!
    assert len(proximal_logprobs_t_buggy) == 7  # 4 + 3
    assert len(accumulated_output_tokens) == 3
    # This would FAIL: assert len(proximal_logprobs_t_buggy) == len(accumulated_output_tokens)

    # CORRECT CODE:
    proximal_logprobs_t_correct = []
    # First iteration: just extend with output_logprobs
    proximal_logprobs_t_correct.extend(output_logprobs)

    # Correct: Lengths match
    assert len(proximal_logprobs_t_correct) == len(accumulated_output_tokens)
    assert proximal_logprobs_t_correct == [1.0, 1.1, 1.2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
