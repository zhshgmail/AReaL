import os
import sys

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import GRPOConfig, load_expr_config
from arealite.api.io_struct import AllocationMode, FinetuneSpec, WeightUpdateMeta
from arealite.dataset.__init__ import get_custom_dataset
from arealite.engine.ppo.actor import FSDPPPOActor
from arealite.engine.sglang_remote import RemoteSGLangEngine
from arealite.utils.device import log_gpu_stats
from arealite.utils.evaluator import Evaluator
from arealite.utils.saver import Saver
from arealite.utils.stats_logger import StatsLogger
from arealite.workflow.rlvr import RLVRWorkflow
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, seeding, stats_tracker

logger = logging.getLogger("GSM8K grpo")


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from realhf.impl.dataset.math_parser import process_results

    return int(process_results(completions, answer)[0])


def main(args):
    """
    Main GRPO training function for GSM8K dataset.
    
    Important: This training script expects SGLang inference servers to be 
    ALREADY RUNNING before this script starts. The servers are launched by 
    the AReaL launcher (e.g., arealite.launcher.local) which:
    
    1. First starts SGLang servers based on allocation_mode config
    2. Waits for servers to be ready and healthy  
    3. Sets AREAL_LLM_SERVER_ADDRS environment variable
    4. Then starts this training script
    
    This script connects to the pre-existing servers via RemoteSGLangEngine.
    """
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        split="train",
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        path=config.valid_dataset.path,
        rank=rank,
        world_size=world_size,
        split="test",
        type=config.valid_dataset.type,
        tokenizer=tokenizer,
    )

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size // world_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.valid_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    # NOTE: SGLang servers must already be running at this point.
    # They are started by the launcher before this training script begins.
    logger.info("Connecting to pre-existing SGLang inference servers...")
    server_addrs = os.getenv("AREAL_LLM_SERVER_ADDRS", "").split(",")
    if not server_addrs or server_addrs == [""]:
        raise RuntimeError(
            "No SGLang server addresses found in AREAL_LLM_SERVER_ADDRS. "
            "SGLang servers must be started before this training script runs. "
            "Use the AReaL launcher (e.g., python -m arealite.launcher.local) to start servers first."
        )
    logger.info(f"Found SGLang servers at: {server_addrs}")
    
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)
    eval_rollout = RemoteSGLangEngine(config.rollout)
    eval_rollout.initialize(None, ft_spec)
    # NOTE: set a large version such that eval does not have any offpolicyness control
    eval_rollout.set_version(int(1e12))
    logger.info("Successfully connected to SGLang inference servers")

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.initialize(None, ft_spec)

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    weight_update_meta = [
        WeightUpdateMeta.from_fsdp_nccl(
            AllocationMode.from_str(config.allocation_mode), actor
        )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )

    # Run training.
    saver = Saver(config.saver, ft_spec, for_recover=False)
    logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    logger.info(f"total_epochs={total_epochs} step_per_epoch={steps_per_epoch}")
    data_generator = iter(train_dataloader)
    for global_step in range(max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            else:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(train_dataloader)
                    data = next(data_generator)
                batch = rollout.rollout_batch(data, workflow=workflow)

        batch = batch.to(actor.device)
        # Create barrier to synchronize all rollout processes.
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
            saver.save(actor, epoch, step, global_step)

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                rollout.pause()
                cnt = 0
                for data in valid_dataloader:
                    for item in data:
                        eval_rollout.submit(item, workflow)
                        cnt += 1
                batch = eval_rollout.wait(cnt, timeout=None)
                rewards = batch["rewards"].float().to(actor.device)
                with stats_tracker.scope("grpo-eval"):
                    stats_tracker.denominator(
                        n_seqs=torch.ones(
                            rewards.shape[0],
                            device=rewards.device,
                            dtype=torch.bool,
                        )
                    )
                    stats_tracker.stat(task_reward=rewards, denominator="n_seqs")
                rollout.resume()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        logger.commit(epoch, step, global_step, stats)

    logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
