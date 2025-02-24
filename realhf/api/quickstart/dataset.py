# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses


@dataclasses.dataclass
class PromptAnswerDatasetConfig:
    """Configuration for datasets used in Supervised Fine-Tuning (SFT).

    The raw data must be in a JSON or JSONL file format, where each entry is a dictionary
    with the keys `prompt` and `answer`. Both `prompt` and `answer` must be strings.

    :param train_path: Path to the training dataset.
    :type train_path: str
    :param valid_path: Path to the validation dataset.
    :type valid_path: str
    :param max_seqlen: Maximum sequence length (prompt + answer). Sequences longer than
        this will be truncated.
    :type max_seqlen: int
    :param train_bs_n_seqs: Number of sequences in each batch during training.
    :type train_bs_n_seqs: int
    :param valid_bs_n_seqs: Number of sequences in each batch during validation.
    :type valid_bs_n_seqs: int
    :param fill_to_max_length: Whether to fill sequences to the maximum length. If True,
        prompts will be left-filled with non-pad tokens. Only used for testing.
    :type fill_to_max_length: bool
    """

    train_path: str = ""
    valid_path: str = ""
    max_seqlen: int = 1024
    train_bs_n_seqs: int = 256
    valid_bs_n_seqs: int = 256
    fill_to_max_length: bool = False


@dataclasses.dataclass
class PairedComparisonDatasetConfig:
    """Configuration for datasets used in paired-comparison reward modeling,
    DPO, and SimPO.

    The raw data must be in a JSON or JSONL file format, where each entry is a dictionary
    with the keys `prompt`, `pos_answers`, and `neg_answers`. `prompt` is a string, while
    `pos_answers` and `neg_answers` are lists of strings. The lists must have the same length.

    The raw dataset may contain multiple answer pairs for each prompt. In each epoch, we will
    randomly sample `max_pairs_per_prompt` answer pairs for each prompt, so the maximum batch
    size (in terms of the number of sequences) per step is `train_bs_n_seqs` multiplied by
    `max_pairs_per_prompt`.

    :param train_path: Path to the training dataset.
    :type train_path: str
    :param valid_path: Path to the evaluation dataset.
    :type valid_path: str
    :param max_pairs_per_prompt: Maximum number of answer pairs per prompt.
    :type max_pairs_per_prompt: int
    :param max_seqlen: Maximum sequence length (prompt + answers). Sequences longer than
        this will be truncated.
    :type max_seqlen: int
    :param train_bs_n_seqs: Number of sequences in each batch during training.
    :type train_bs_n_seqs: int
    :param valid_bs_n_seqs: Number of sequences in each batch during validation.
    :type valid_bs_n_seqs: int
    """

    train_path: str = ""
    valid_path: str = ""
    max_pairs_per_prompt: int = 2
    max_seqlen: int = 1024
    train_bs_n_seqs: int = 256
    valid_bs_n_seqs: int = 256


@dataclasses.dataclass
class PromptOnlyDatasetConfig:
    """Configuration for datasets used in PPO RLHF.

    The raw data must be in a JSON or JSONL file format, where each entry is a dictionary
    with a single key called `prompt`, which is a string.

    :param path: Path to the dataset.
    :type path: str
    :param max_prompt_len: Maximum length of the prompt. Prompts longer than this will
        be truncated.
    :type max_prompt_len: int
    :param train_bs_n_seqs: Number of prompts in each batch.
    :type train_bs_n_seqs: int
    :param fill_to_max_length: Whether to fill prompts to the maximum length. If True,
        prompts will be left-filled with non-pad tokens. Only used for testing.
    :type fill_to_max_length: bool
    """

    path: str = ""
    max_prompt_len: int = 256
    train_bs_n_seqs: int = 256
    fill_to_max_length: bool = False
