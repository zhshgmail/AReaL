import time
from typing import Dict

import torch
import torch.distributed as dist
import torch.utils.data

from arealite.api.io_struct import SaveLoadMeta
from arealite.api.trainer_api import Trainer
from arealite.utils.functional import gather_logprobs
from arealite.utils.logging import record_timing
from realhf.api.core.data_api import tabulate_stats
from realhf.base import logging, stats_tracker


def compute_packed_sft_loss(
    logits: torch.Tensor,
    input_: Dict[str, torch.Tensor],
) -> torch.Tensor:
    packed_input_ids: torch.Tensor = input_["input_ids"]
    cu_seqlens: torch.Tensor = input_["cu_seqlens"]
    prompt_mask = input_["prompt_mask"].bool()
    logits = logits.float()

    logprobs = gather_logprobs(logits, torch.roll(packed_input_ids, shifts=-1, dims=-1))
    prompt_mask = torch.roll(prompt_mask, shifts=-1, dims=-1)
    logprobs = torch.where(prompt_mask, 0, logprobs)

    loss = -logprobs.sum() / prompt_mask.logical_not().count_nonzero()

    with torch.no_grad():
        seqlogp = torch.zeros(
            cu_seqlens.shape[0] - 1, device=logits.device, dtype=torch.float64
        )
        for i in range(cu_seqlens.shape[0] - 1):
            m = prompt_mask[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            logp = logprobs[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            assert cu_seqlens[i + 1] - i - 1 <= logprobs.shape[0], (
                cu_seqlens,
                logprobs.shape,
            )
            seqlogp[i] = torch.where(m, 0.0, logp.detach()).sum() / (
                m.numel() - m.count_nonzero()
            )

    ## Loggin stats
    stats_tracker.denominator(
        n_seqs=torch.ones(
            cu_seqlens.shape[0] - 1, dtype=torch.bool, device=logprobs.device
        ),
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=prompt_mask.logical_not(),
        prompt_tokens=prompt_mask,
    )
    stats_tracker.stat(ppl=(-seqlogp).exp().float(), denominator="n_seqs")
    stats_tracker.stat(loss=-logprobs.detach(), denominator="n_valid_tokens")
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    return loss


class SFTTrainer(Trainer):

    def train(self):
        total_epochs = self.config.exp_ctrl.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)

        self.log(f"total_epochs={total_epochs} step_per_epoch={steps_per_epoch}")
        global_step = 0
        start_time = time.monotonic()
        for epoch in range(total_epochs):
            for step, data in enumerate(self.train_dataloader):
                self.engine.train()
                timing_stats = {}
                with record_timing("timeperf/train_step", timing_stats):
                    with stats_tracker.scope("sft"):
                        stats = self.engine.train_batch(
                            input_=data,
                            loss_fn=compute_packed_sft_loss,
                            loss_weight_fn=lambda x: x["prompt_mask"]
                            .logical_not()
                            .count_nonzero(),
                            mb_spec=self.config.mb_spec,
                        )
                        self.engine.step_lr_scheduler()
                        stats_tracker.scalar(**stats)

                if self.save_ctl.check(
                    epochs=int(step == steps_per_epoch - 1), steps=1
                ):
                    self.log("Saving model ...")

                    with record_timing("timeperf/save", timing_stats):
                        save_path = self.get_save_checkpoint_path(
                            self.config, epoch, step, global_step
                        )
                        meta = SaveLoadMeta(
                            path=save_path,
                            weight_format="hf",
                            with_optim=False,
                            tokenizer=self.tokenizer,
                            base_model_path=self.config.tokenizer_path,
                        )
                        self.engine.save(meta)

                if self.eval_ctl.check(
                    epochs=int(step == steps_per_epoch - 1), steps=1
                ):
                    if dist.get_rank() == 0:
                        self.log("Running evaluation ...")
                    with record_timing("timeperf/eval", timing_stats):
                        self.evaluate()

                stats = stats_tracker.export()
                stats.update(timing_stats)
                self.log_wandb_tensorboard(global_step, stats)

                self.log(
                    f"Epoch {epoch+1}/{total_epochs} "
                    f"Step {step+1}/{steps_per_epoch} "
                    f"Train step {global_step + 1}/{total_epochs * steps_per_epoch} done."
                )
                self.log(
                    f"Detailed time stats: \n{tabulate_stats(timing_stats, floatfmt='.2f')}"
                )
                self.log(f"SFT training stats:\n{tabulate_stats(stats)}")
                global_step += 1

        self.log(
            f"Training completes! Total time elapsed {time.monotonic() - start_time:.2f}."
        )

        self.close_wandb_tensorboard()

    def evaluate(self):
        if self.valid_dataloader is None:
            return
        self.engine.eval()
        for data in self.valid_dataloader:
            with stats_tracker.scope("sft-eval"):
                # No need to log anything. Logging will be handled outside
                # via stats_tracker.export().
                self.engine.eval_batch(
                    input_=data,
                    loss_fn=compute_packed_sft_loss,
                    loss_weight_fn=lambda x: x["prompt_mask"]
                    .logical_not()
                    .count_nonzero(),
                    mb_spec=self.config.mb_spec,
                )
