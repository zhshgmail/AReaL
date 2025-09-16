import itertools
import os
import sys
from copy import deepcopy

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import AllocationMode, FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import seeding, stats_tracker
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow
from collections import defaultdict

def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from areal.reward.math_parser import process_results

    return int(process_results(completions, answer)[0])


def main(args):
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
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize(None, ft_spec)

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
    eval_workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    # Run training.
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
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        # Compute prox_logp using the proximal policy (one version newer than behavior policy) for each token
        # Assumptions:
        # - batch["policy_versions"]: shape [batch_size, seq_len], int, policy version for each token
        # - There is a function or mapping to get checkpoint path for a given version: get_checkpoint_path(version)
        # - FSDPPPOActor can be loaded from a checkpoint path
        # - batch["prox_logp"]: shape [batch_size, seq_len], to be filled

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                # 1. Prepare mapping from version to checkpoint path
                from areal.utils.saver import Saver
                # Helper to infer (epoch, step, global_step) from policy version (assuming version == global_step)
                def infer_epoch_step(version, steps_per_epoch):
                    epoch = version // steps_per_epoch
                    step = version % steps_per_epoch
                    return epoch, step

                def get_checkpoint_path(version):
                    # Infer epoch, step, global_step from policy version (assuming version == global_step)
                    epoch, step = infer_epoch_step(version, steps_per_epoch)
                    global_step = version
                    return Saver.get_model_save_path(
                        config.saver.experiment_name,
                        config.saver.trial_name,
                        config.saver.fileroot,
                        epoch,
                        step,
                        global_step,
                        name="default"
                    )

                # 2. Group (batch_idx, token_idx) by required proximal policy version
                policy_versions = batch["policy_versions"].cpu().numpy()  # shape [batch_size, seq_len]
                batch_size, seq_len = policy_versions.shape
                groups = defaultdict(list)  # {prox_version: [(batch_idx, token_idx), ...]}
                for batch_idx in range(batch_size):
                    for token_idx in range(seq_len):
                        behav_version = policy_versions[batch_idx, token_idx]
                        prox_version = behav_version + 1
                        groups[prox_version].append((batch_idx, token_idx))

                # 3. Cache temp_actor for each unique prox_version to avoid repeated construction/loading
                from areal.api.io_struct import SaveLoadMeta
                temp_actor_cache = {}
                prox_logp = torch.zeros_like(batch["logprobs"])
                # Determine the minimum version to keep in cache
                # max_head_offpolicyness is the max allowed difference between current policy and behavior policy
                # So, evict any cached version older than (current_policy_version - max_head_offpolicyness)
                # Assume current_policy_version = global_step (or set appropriately)
                max_head_offpolicyness = config.max_head_offpolicyness
                current_policy_version = global_step
                min_version_to_keep = current_policy_version - max_head_offpolicyness

                # Remove old cached actors
                old_versions = [v for v in temp_actor_cache if v < min_version_to_keep]
                for v in old_versions:
                    del temp_actor_cache[v]

                # Aggregate the tokens generated by the same behavior policy into the same group
                for prox_version, positions in groups.items():
                    if prox_version not in temp_actor_cache:
                        checkpoint_path = get_checkpoint_path(prox_version)
                        if not os.path.exists(checkpoint_path):
                            raise FileNotFoundError(f"Checkpoint for policy version {prox_version} not found: {checkpoint_path}")
                        temp_actor = FSDPPPOActor(config=config.actor)
                        save_load_meta = SaveLoadMeta(
                            path=checkpoint_path,
                            weight_format="hf",
                            tokenizer=tokenizer,
                            with_optim=True,
                            base_model_path=None,
                        )
                        temp_actor.load(save_load_meta)
                        temp_actor_cache[prox_version] = temp_actor
                    else:
                        temp_actor = temp_actor_cache[prox_version]
                    # Compute logp for the whole batch
                    logp_seq = temp_actor.compute_logp(batch)  # shape [batch_size, seq_len]
                    # Assign only the relevant positions for this prox_version
                    for batch_idx, token_idx in positions:
                        prox_logp[batch_idx, token_idx] = logp_seq[batch_idx, token_idx]
                batch["prox_logp"] = prox_logp
                log_gpu_stats("recompute logp (proximal policy)")

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

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                # Stats are logged in the workflow
                # and will be exported later
                cnt = 0
                for data in valid_dataloader:
                    for item in data:
                        eval_rollout.submit(item, eval_workflow)
                        cnt += 1
                eval_rollout.wait(cnt, timeout=None)

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

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

        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats[0].update(stats_tracker.export_all(reduce_group=actor.parallelism_group))
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
