import itertools
import os
import re
import sys

import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    StepInfo,
    WeightUpdateMeta,
)
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, seeding, stats_tracker
from realhf.impl.dataset.math_parser import process_results
from realhf.utils import load_hf_or_local_file

logger = logging.getLogger("boba_grpo")

MAX_PROMPT_LENGTH = 1024


def _prepare_prompt(prompt: str) -> tuple[str, bool]:
    local_thinking = False
    if "<think>" in prompt:
        local_thinking = True
        prompt = prompt.replace("<think>", "")
    prompt = re.sub(r"<[^>]*User[^>]*>", "", prompt)
    prompt = re.sub(r"<[^>]*Assistant[^>]*>", "", prompt)
    return prompt.strip(), local_thinking


def get_input_ids_fn(raw_prompt, tokenizer, enable_thinking):
    if not isinstance(raw_prompt, str):
        raise ValueError(f"Expected prompt to be a string, got {type(raw_prompt)}")
    prompt, detect_thinking = _prepare_prompt(raw_prompt)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking or detect_thinking,
    )


def data_extract_prompt_fn(sample):
    return sample["prompt"]


def get_boba_math_dataset(path, tokenizer, rank, world_size, max_prompt_len=MAX_PROMPT_LENGTH):
    resolved_path = load_hf_or_local_file(path)
    dataset = load_dataset("json", data_files=resolved_path, split="train")
    if max_prompt_len is not None:
        dataset = dataset.filter(
            lambda x: len(tokenizer.encode(x["prompt"])) <= max_prompt_len
        )
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def _normalize_solution_entry(entry):
    if entry is None:
        return None
    if isinstance(entry, dict):
        for key in ("value", "answer", "label", "solution"):
            if key in entry and entry[key]:
                return entry[key]
        return None
    return str(entry)


def boba_reward_fn(prompt, completions, prompt_ids, completion_ids, solutions=None, answer=None, **kwargs):
    candidates = []
    if solutions is None:
        solutions = []
    if isinstance(solutions, (str, dict)):
        solutions = [solutions]
    for entry in solutions:
        normalized = _normalize_solution_entry(entry)
        if not normalized:
            continue
        result = process_results(completions, normalized)
        if result and result[0]:
            return 1
    if answer:
        result = process_results(completions, answer)
        if result and result[0]:
            return 1
    return 0


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if config.train_dataset.batch_size % world_size != 0:
        raise ValueError(
            f"train batch size ({config.train_dataset.batch_size}) must be divisible by world size ({world_size})."
        )

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    train_dataset = get_boba_math_dataset(
        path=config.train_dataset.path,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
    )

    per_rank_batch_size = config.train_dataset.batch_size // world_size
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=per_rank_batch_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)

    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.initialize(None, ft_spec)

    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    weight_update_meta = [
        WeightUpdateMeta.from_fsdp_nccl(
            allocation_mode,
            actor,
        )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = RLVRWorkflow(
        reward_fn=boba_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        get_input_ids_fn=get_input_ids_fn,
        data_extract_prompt_fn=data_extract_prompt_fn,
    )

    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)
    recover_handler = RecoverHandler(config.recover, ft_spec)

    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    data_generator = itertools.cycle(train_dataloader)
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            else:
                batch = rollout.rollout_batch(next(data_generator), workflow=workflow)

        batch = batch.to(actor.device)
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        with stats_tracker.record_timing("update_weights"):
            rollout.pause()
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()
            rollout.resume()
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        stats_logger.commit(epoch, step, global_step, stats)

    stats_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
