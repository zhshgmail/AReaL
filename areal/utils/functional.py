from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


@torch.compile
def _gather_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels


@torch.compile
def _gather_logprobs_entropy(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels, entropy


def gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
):
    batch_size = logits.shape[0]

    if batch_size <= chunk_size:
        return _gather_logprobs(logits, labels, temperature)

    log_probs_labels_list = []

    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_logits = logits[i:end_idx]
        chunk_labels = labels[i:end_idx]

        chunk_log_probs = _gather_logprobs(chunk_logits, chunk_labels, temperature)

        log_probs_labels_list.append(chunk_log_probs)

    return torch.cat(log_probs_labels_list)


def gather_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
):
    batch_size = logits.shape[0]

    if batch_size <= chunk_size:
        return _gather_logprobs_entropy(logits, labels, temperature)

    log_probs_labels_list = []
    entropy_list = []

    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_logits = logits[i:end_idx]
        chunk_labels = labels[i:end_idx]

        chunk_log_probs, chunk_entropy = _gather_logprobs_entropy(
            chunk_logits, chunk_labels, temperature
        )

        log_probs_labels_list.append(chunk_log_probs)
        entropy_list.append(chunk_entropy)

    return torch.cat(log_probs_labels_list), torch.cat(entropy_list)


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim=None,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
    reduce_group=None,
):
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized() and all_reduce:
        dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(
            x_sum_sq,
            op=dist.ReduceOp.SUM,
            group=reduce_group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    proximal_logprobs_t: Optional[torch.Tensor] = None,
    c_clip: Optional[float] = None,
    behav_imp_weight_cap: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Params:
        proximal_logprobs: the proximal policy logprob closest to the target policy.
        proximal_logprobs_t: the segment-wise proximal policy logprobs.


    When decoupled loss is disabled:
    1. if recompute logp, both old_logprobs and proximal_logprobs are recomputed logp;
    2. if no recomputation, both old_logp and proximal_logprobs are produced by the inference backend.

    When decoupled loss is enabled, proximal_logprobs is the recomputed logp,
    old_logprobs is produced by the inference engine.
    """
    loss_mask_count = loss_mask.count_nonzero() or 1
    ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    pg_loss = torch.max(pg_loss1, pg_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)
    if proximal_logprobs_t is not None:
        behav_kl = proximal_logprobs_t - old_logprobs
        # Quick comparison print without changing logic
        behav_kl_decoupled = proximal_logprobs - old_logprobs
    else:
        behav_kl = proximal_logprobs - old_logprobs
    behav_imp_weight = behav_kl.exp()
    behav_imp_weight_decoupled = behav_kl_decoupled.exp()
    behav_mask = (
        (behav_imp_weight <= behav_imp_weight_cap).logical_and(loss_mask)
        if behav_imp_weight_cap is not None
        else loss_mask
    )
    behav_kl = torch.where(behav_mask, behav_kl, 0.0)
    behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)
    # # Print comparison after mask processing
    # if proximal_logprobs_t is not None:
    #     behav_kl_decoupled_masked = torch.where(behav_mask, behav_kl_decoupled, 0.0)
    #     behav_imp_weight_decoupled_masked = torch.where(behav_mask, behav_imp_weight_decoupled, 0.0)
    #     # 只在有效位上取值，避免 0 稀释
    #     seg_vals  = behav_imp_weight[behav_mask]
    #     seg_logw  = behav_kl[behav_mask]
    #     dec_vals  = behav_imp_weight_decoupled_masked[behav_mask]
    #     dec_logw  = behav_kl_decoupled_masked[behav_mask]

    #     # 保护：若无有效位则跳过
    #     if seg_vals.numel() > 0:
    #         # 对数域更稳健
    #         seg_mean_logw = seg_logw.mean().item()
    #         seg_std_logw  = seg_logw.std(unbiased=False).item()
    #         # ESS / N
    #         seg_ess_frac  = (seg_vals.sum() ** 2 / (seg_vals.pow(2).sum() * seg_vals.numel())).item()
    #         # 尾部占比与分位数
    #         seg_tail_hi   = (seg_vals > 2.0).float().mean().item()
    #         seg_tail_lo   = (seg_vals < 0.5).float().mean().item()
    #         q = torch.tensor([0.5, 0.9, 0.99], device=seg_vals.device)
    #         seg_p50, seg_p90, seg_p99 = torch.quantile(seg_vals, q).tolist()

    #         print(
    #             "[Segment w] "
    #             f"logw_mean={seg_mean_logw:.4f} "
    #             f"logw_std={seg_std_logw:.4f} "
    #             f"ESS%={seg_ess_frac*100:.1f} "
    #             f"P(w>2)={seg_tail_hi*100:.2f}% "
    #             f"P(w<0.5)={seg_tail_lo*100:.2f}% "
    #             f"p50/p90/p99={seg_p50:.3f}/{seg_p90:.3f}/{seg_p99:.3f}"
    #         )

    #     if dec_vals.numel() > 0:
    #         dec_mean_logw = dec_logw.mean().item()
    #         dec_std_logw  = dec_logw.std(unbiased=False).item()
    #         dec_ess_frac  = (dec_vals.sum() ** 2 / (dec_vals.pow(2).sum() * dec_vals.numel())).item()
    #         dec_tail_hi   = (dec_vals > 2.0).float().mean().item()
    #         dec_tail_lo   = (dec_vals < 0.5).float().mean().item()
    #         q = torch.tensor([0.5, 0.9, 0.99], device=dec_vals.device)
    #         dec_p50, dec_p90, dec_p99 = torch.quantile(dec_vals, q).tolist()

    #         print(
    #             "[Decoupled w] "
    #             f"logw_mean={dec_mean_logw:.4f} "
    #             f"logw_std={dec_std_logw:.4f} "
    #             f"ESS%={dec_ess_frac*100:.1f} "
    #             f"P(w>2)={dec_tail_hi*100:.2f}% "
    #             f"P(w<0.5)={dec_tail_lo*100:.2f}% "
    #             f"p50/p90/p99={dec_p50:.3f}/{dec_p90:.3f}/{dec_p99:.3f}"
    #         )
    pg_loss = pg_loss * behav_imp_weight
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
    )
    # Emit behavior-related stats when either proximal source is present
    if (proximal_logprobs is not None) or (proximal_logprobs_t is not None):
        stat["behave_imp_weight"] = behav_imp_weight
        stat["behave_approx_kl"] = behav_kl
        stat["behave_mask"] = behav_mask
    return pg_loss, stat
