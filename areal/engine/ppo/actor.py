import functools
from typing import Dict, List, Optional
from collections import deque

import torch
from tensordict import TensorDict

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.utils.data import split_padded_tensor_dict_into_mb_list
from areal.utils.functional import (
    gather_logprobs,
    gather_logprobs_entropy,
    masked_normalization,
    ppo_actor_loss_fn,
)
from realhf.base import stats_tracker


class PPOActor:

    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.group_reward_norm = config.group_reward_norm
        self.group_adv_norm = config.group_adv_norm
        self.group_size = config.group_size

        self.kl_ctl = config.kl_ctl

        self.adv_norm = config.adv_norm
        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature

        # Tracking for cross-step reward mean change (for variance over time)
        self._prev_reward_mean: float | None = None
        self._reward_mean_hist = deque(maxlen=20)

    @torch.no_grad()
    def compute_logp(
        self,
        data: TensorDict,
        temperature: Optional[float] = None,
    ) -> torch.Tensor | None:

        def calc_logprobs(logits, input_data):
            labels = torch.roll(input_data["input_ids"], shifts=-1, dims=-1)
            logprobs = gather_logprobs(logits, labels, temperature or 1.0)
            return logprobs

        self.engine.eval()
        return self.engine.forward(
            input_=data,
            post_hook=calc_logprobs,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    def compute_advantages(self, data: TensorDict) -> None:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Compute rewards using the reward function in synchronous RLVR pipeline.
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.group_reward_norm:
            for i in range(bs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                r = reward_score[s]
                reward_score[s] = (r - r.mean()) / (r.std() + 1e-9)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            old_logp = data["logprobs"] = data["prox_logp"]
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp", torch.zeros_like(old_logp))
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * (old_logp - ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        for t in reversed(range(max_seqlen - 1)):
            nextvalues = values[:, t + 1]
            if t == max_seqlen - 2:
                nextvalues *= seq_no_eos_mask
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            lastgaelam = delta + self.discount * self.gae_lambda * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # Optionally perform advantage normalization.
        if self.adv_norm or self.group_adv_norm:
            if self.group_adv_norm:
                adv_list = []
                for i in range(0, bs // self.group_size):
                    s = slice(i * self.group_size, (i + 1) * self.group_size)
                    adv = advantages[s]
                    m = loss_mask[s]
                    adv_list.append(masked_normalization(adv, m, all_reduce=False))
                advantages = torch.cat(adv_list, 0)
            else:
                advantages = masked_normalization(advantages, loss_mask)

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp

    def ppo_update(self, data: TensorDict) -> List[Dict[str, float]]:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        all_stats = []
        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=torch.ones_like(loss_mask, dtype=torch.bool),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = []
        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            task_reward=reward_score.float(),
            task_reward_var=torch.var(reward_score.float()).expand_as(reward_score),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        # For variance via E[x^2] - (E[x])^2, log second moment as well
        stats_tracker.stat(
            task_reward_sq=reward_score.float() ** 2, denominator="n_seqs"
        )
        # Also log a direct per-batch variance (unbiased=False) as a scalar for quick inspection
        m = reward_score.float().mean().item()
        m2 = (reward_score.float() ** 2).mean().item()
        var_moment = max(0.0, m2 - m * m)
        stats_tracker.scalar(task_reward_var_moment=var_moment)
        # Cross-step change metrics: delta and z-score normalized by SE
        n = int(reward_score.numel()) or 1
        se = (var_moment / n) ** 0.5 if var_moment >= 0 else 0.0
        if self._prev_reward_mean is not None:
            delta = m - self._prev_reward_mean
            z = abs(delta) / (se if se > 1e-8 else 1.0)
            stats_tracker.scalar(
                task_reward_delta=delta,
                task_reward_delta_abs=abs(delta),
                task_reward_delta_z=z,
            )
        self._prev_reward_mean = m
        self._reward_mean_hist.append(m)
        if len(self._reward_mean_hist) >= 2:
            import torch as _torch
            roll_var = _torch.tensor(list(self._reward_mean_hist)).float().var(unbiased=False).item()
            stats_tracker.scalar(
                task_reward_mean_roll_var=roll_var,
                task_reward_mean_roll_std=(roll_var ** 0.5),
            )
        # Also log a direct per-batch variance (unbiased=False) as a scalar for quick inspection
        m = reward_score.float().mean().item()
        m2 = (reward_score.float() ** 2).mean().item()
        var_moment = max(0.0, m2 - m * m)
        stats_tracker.scalar(task_reward_var_moment=var_moment)
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.behav_imp_weight_cap is not None:
            scalars["behav_imp_weight_cap"] = self.config.behav_imp_weight_cap
        if self.config.behav_imp_weight_floor is not None:
            scalars["behav_imp_weight_floor"] = self.config.behav_imp_weight_floor
        stats_tracker.scalar(**scalars)

        global_stats = stats_tracker.export(reduce_group=self.engine.parallelism_group)
        for k in global_denominators:
            keys = list(global_stats.keys())
            for k2 in keys:
                if k2.endswith(k):
                    global_stats.pop(k2)
        ########## Logging code ends ##########

        for key in ["rewards", "tot_rewards", "kl_rewards", "versions"]:
            data.pop(key, None)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )
        for mb in mb_inputs.mbs:
            train_stat = self.engine.train_batch(
                mb,
                loss_fn=functools.partial(
                    grpo_loss_fn,
                    temperature=self.temperature,
                    eps_clip=self.config.eps_clip,
                    c_clip=self.config.c_clip,
                    behav_imp_weight_cap=self.config.behav_imp_weight_cap,
                    behav_imp_weight_floor=self.config.behav_imp_weight_floor,
                ),
                loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
            )
            stats_tracker.scalar(**train_stat)
            all_stats.append(
                stats_tracker.export(reduce_group=self.engine.parallelism_group)
            )
        all_stats[0].update(global_stats)
        return all_stats


class FSDPPPOActor(FSDPEngine):

    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> None:
        self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> List[Dict[str, float]]:
        return self.actor.ppo_update(*args, **kwargs)


def grpo_loss_fn(
    logits: torch.Tensor,
    input_data: Dict,
    temperature: float,
    eps_clip: float,
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
    behav_imp_weight_floor: float | None,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    input_ids = input_data["input_ids"]
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp = input_data["prox_logp"]

    # Get segment-wise proximal logprobs if available
    proximal_logprobs_t = input_data.get("proximal_logprobs_t", None)
    if proximal_logprobs_t is not None:
        # Apply same roll(-1) transformation as other logprobs for next-token prediction alignment
        proximal_logprobs_t = torch.roll(proximal_logprobs_t, shifts=-1, dims=-1)

    logprobs, entropy = gather_logprobs_entropy(
        logits, torch.roll(input_ids, shifts=-1, dims=-1), temperature
    )
    entropy = entropy.detach()
    loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=loss_mask,
        c_clip=c_clip,
        proximal_logprobs=prox_logp,
        proximal_logprobs_t=proximal_logprobs_t,
        behav_imp_weight_cap=behav_imp_weight_cap,
        behav_imp_weight_floor=behav_imp_weight_floor,
    )

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    clip_mask = stat["clip_mask"]
    clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
    clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
    stats_tracker.stat(
        clipped_new_logp=clipped_new_logp,
        clipped_old_logp=clipped_old_logp,
        denominator="clipped_tokens",
    )
    return loss
