from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class MicroBatchSpec:
    """Specification for splitting micro-batches during training."""

    n_mbs: int = field(
        default=1,
        metadata={
            "help": "Number of micro-batches (or minimum number if max_tokens_per_mb is set). Used when max_tokens_per_mb is None or as minimum count",
        },
    )
    max_tokens_per_mb: int = field(
        default=int(1e12),
        metadata={
            "help": "Maximum tokens per micro-batch. When set, n_mbs becomes the minimum number of micro-batches",
        },
    )

    @classmethod
    def new(cls, mb_spec: "MicroBatchSpec", **kwargs):
        """Create new spec with updated fields while maintaining Omegaconf compatibility."""
        fields = dict(
            n_mbs=mb_spec.n_mbs,
            max_tokens_per_mb=mb_spec.max_tokens_per_mb,
        )
        fields.update(kwargs)
        return cls(**fields)

@dataclass
class GenerationHyperparameters:
    """Controls text generation behavior for RL training."""

    n_samples: int = field(
        default=1, metadata={"help": "Number of sequences to generate per prompt."}
    )
    max_new_tokens: int = field(
        default=16384, metadata={"help": "Maximum number of tokens to generate."}
    )
    min_new_tokens: int = field(
        default=0, metadata={"help": "Minimum number of tokens to generate."}
    )
    greedy: bool = field(
        default=False,
        metadata={"help": "Whether to use greedy decoding (max probability)."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus sampling probability threshold (0.0, 1.0]."},
    )
    top_k: int = field(
        default=int(1e8),
        metadata={"help": "Number of highest probability tokens to consider."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature. Higher values increase diversity."},
    )
    stop_token_ids: List[int] = field(
        default_factory=list,
        metadata={"help": "Stop generation when encoutering these token ids."},
    )

    def new(self, **kwargs):
        args = asdict(self)
        args.update(kwargs)
        return GenerationHyperparameters(**args)

@dataclass
class InferenceEngineConfig:
    # Used by remote inference engines.
    server_addrs: List[str] = field(
        default_factory=list,
        metadata={"help": "List of server addresses for inference."}
    )
    schedule_policy: str = field(
        default="round_robin",
        metadata={"help": "Request scheduling policy", "choices": ["round_robin"]},
    )
    request_timeout: float = field(
        default=30.0,
        metadata={"help": "Timeout for HTTP requests."}
    )
    request_retries: int = field(
        default=3,
        metadata={"help": "Number of retries for failed requests."}
    )