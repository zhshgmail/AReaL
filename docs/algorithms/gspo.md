# Group Sequence Policy Optimization (GSPO)

Last updated: Oct 30, 2025

Author: [Bruce Li](https://github.com/HsiaoTsan)

![gspo figure](../figures/gspo.png)

Group Sequence Policy Optimization (GSPO), introduced by Zheng et al. (2025) in the context of training Qwen3 models, is a reinforcement learning algorithm that extends PPO by computing importance sampling ratios at the sequence level rather than the token level. Unlike standard PPO which computes a separate probability ratio for each token, GSPO calculates a single ratio per sequence as the geometric mean of per-token probability ratios, then applies this uniform ratio to all tokens within that sequence.

This sequence-level approach provides several key advantages: improved training stability especially for Mixture-of-Experts (MoE) models, reduced variance in policy updates when optimizing sequence-level rewards, and potential simplification of RL infrastructure design. GSPO has been successfully deployed in large-scale RL training for the Qwen3 model family.

## Algorithm Overview

The key distinction between GSPO and traditional PPO lies in the computation of importance sampling ratios:

**Standard PPO (token-level):**
- Computes per-token ratio: $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$
- Each token has its own importance weight
- Advantages are computed and applied per token

**GSPO (sequence-level):**
- Computes sequence-level geometric mean ratio: $r_i(\theta) = \exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}\right)$
- All tokens in a sequence share the same importance weight
- **Advantages are aggregated per sequence**: Each sequence contributes its total advantage (sum of per-token advantages) to the objective, ensuring gradient magnitude is independent of sequence length

## Key Differences from Related Algorithms

| Algorithm | Importance Ratio Level | Advantage Aggregation | Best Use Case |
|-----------|------------------------|----------------------|---------------|
| **PPO** | Token-level | Per-token | General RL tasks |
| **GRPO** | Token-level | Group-normalized per-token | Critic-free RL with sparse rewards |
| **GSPO** | Sequence-level (geometric mean) | Per-sequence total (averaged per token, summed over tokens) | Sequence-level rewards, MoE training |

## Implementation Details

### Advantage Aggregation

The GSPO paper objective is:

$$\mathcal{J}_\text{GSPO}(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^{G} \min(s_i(\theta) \hat{A}_i, \text{clip}(s_i(\theta)) \hat{A}_i)\right]$$

where $\hat{A}_i$ is the **total advantage for sequence $i$** (sum of per-token advantages).

**Critical Implementation Note**: To ensure gradient magnitude is independent of sequence length:

1. Compute sequence-level advantage: $\hat{A}_i = \sum_{t=1}^{|y_i|} A_{i,t}$
2. Compute average advantage per token: $\bar{A}_i = \frac{\hat{A}_i}{|y_i|}$
3. Broadcast $\bar{A}_i$ to all tokens in sequence $i$
4. When summing over tokens in the loss, each sequence contributes: $|y_i| \times \bar{A}_i = \hat{A}_i$

This ensures that:
- Each sequence contributes proportionally to its total advantage, not its length
- Longer sequences don't dominate the gradient
- Gradient magnitude remains stable across varying sequence lengths

**Common Pitfall**: Broadcasting $\hat{A}_i$ (sum) instead of $\bar{A}_i$ (average) causes gradients to scale by sequence length, leading to high gradient norms and training instability.

For more details:

- AReaL Detail: [Paper of AReaL](https://arxiv.org/abs/2505.24298)

- GSPO Detail: [Paper of GSPO](https://arxiv.org/abs/2507.18071)

- Qwen Team Blog: [GSPO Blog Post](https://qwenlm.github.io/blog/gspo/)

## Algorithm Core Parameters

GSPO shares most parameters with GRPO, with one key addition:

- `actor.importance_sampling_level`: Set to `"sequence"` to enable GSPO (default `"token"` for standard PPO/GRPO)

## Example Usage

We recommend changing parameters within the configuration file (e.g., `gsm8k_gspo.yaml`).

| Backend   | CMD                                                                                                                              |
| --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **local** | `python3 -m areal.launcher.local examples/experimental/gspo/gsm8k_gspo.py --config examples/experimental/gspo/gsm8k_gspo.yaml --<other_args_to_overwrite>` |
| **ray**   | `python3 -m areal.launcher.ray examples/experimental/gspo/gsm8k_gspo.py --config examples/experimental/gspo/gsm8k_gspo.yaml --<other_args_to_overwrite>`   |
| **slurm** | `python3 -m areal.launcher.slurm examples/experimental/gspo/gsm8k_gspo.py --config examples/experimental/gspo/gsm8k_gspo.yaml --<other_args_to_overwrite>` |

To enable GSPO, set `importance_sampling_level: sequence` in the actor configuration:

```yaml
actor:
  importance_sampling_level: sequence  # 'token' for standard PPO, 'sequence' for GSPO
  # ... other configurations
```

### Adding GSPO to Other Algorithms

GSPO can be easily applied to any PPO-based algorithm by adding the `importance_sampling_level` parameter. For example, to use GSPO with GRPO:

```bash
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py --config examples/math/gsm8k_grpo.yaml +actor.importance_sampling_level=sequence
```

Note: The plus sign (`+`) is required when the key `importance_sampling_level` is not in the YAML file.

## When to Use GSPO

GSPO is particularly beneficial in the following scenarios:

- **Sequence-level rewards**: When rewards are assigned based on entire sequences (e.g., task success/failure) rather than individual tokens
- **MoE model training**: GSPO has been shown to stabilize Mixture-of-Experts reinforcement learning training
- **High variance in token-level gradients**: The sequence-level geometric mean helps reduce variance
- **Long-horizon tasks**: When token-level credit assignment is difficult or unreliable

## Baselines

We still lack baselines, welcome to contribute!
