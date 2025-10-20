import asyncio
import os
import random
import shutil
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import requests
import torch.distributed as dist
import uvloop
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import (
    ModelRequest,
    ModelResponse,
    ParamSpec,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow, WorkflowExecutor
from areal.api.workflow_factory import create_workflow_executor
from areal.core.staleness_manager import StalenessManager
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names
from areal.utils.http import arequest_with_retry, get_default_connector
from areal.utils.launcher import wait_llm_server_addrs

RID_CACHE_SIZE = 128


class RemoteSGLangEngine(InferenceEngine):

    def __init__(self, config: InferenceEngineConfig):
        self.config = config

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []
        self.addresses = []
        self.server_idx = 0

        self.distributed_weight_update_initialized = False
        self._version = 0

        self.lock = Lock()

        # Workflow executor will be initialized in initialize()
        self.workflow_executor: WorkflowExecutor

    def _wait_for_server(self, address):
        base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.config.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(1)
        raise RuntimeError("server launch failed")

    def check_health(self, base_url):
        # Check server endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=30)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            return False

    def initialize(
        self,
        engine_id: Optional[str] = None,
        addr: str | List[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[SGLang Remote Engine Rank {engine_id}]")

        if addr:
            self.addresses = addr if isinstance(addr, list) else [addr]
            self.logger.info(f"Get server addresses from the `addr` argument.")
        else:
            if (
                self.config.experiment_name is not None
                and self.config.trial_name is not None
            ):
                try:
                    self.addresses = wait_llm_server_addrs(
                        experiment_name=self.config.experiment_name,
                        trial_name=self.config.trial_name,
                        timeout=1,
                    )
                    self.logger.info(f"Get server addresses from name_resolve.")
                except (TimeoutError, RuntimeError):
                    # RuntimeError happens when name_resolve is not properly configured.
                    pass
        if not self.addresses and os.getenv("AREAL_LLM_SERVER_ADDRS"):
            # When addr is not provided, fallback to reading addrs from env var
            self.addresses = os.environ["AREAL_LLM_SERVER_ADDRS"].split(",")
            self.logger.info(f"Get server addresses from environment variable.")
        if not self.addresses:
            raise RuntimeError(
                "No configured SGLang servers. Please pass in SGLang server addresses by arguments "
                "for `RemoteSGLangEngine.initialize` or environment variable `AREAL_LLM_SERVER_ADDRS`."
            )

        self.logger.info("Waiting for server ready...")
        for addr_ in self.addresses:
            self._wait_for_server(addr_)
        self.server_idx = random.randint(0, len(self.addresses) - 1)
        self.logger.info("Servers are all ready!")
        self.executor = ProcessPoolExecutor(max_workers=1)
        self.lora_init = False

        # Create staleness manager (needed for factory)
        staleness_manager = StalenessManager(
            max_concurrent_rollouts=self.config.max_concurrent_rollouts or self.config.consumer_batch_size,
            consumer_batch_size=self.config.consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

        # Create workflow executor using factory
        self.workflow_executor = create_workflow_executor(
            inference_engine=self,
            staleness_manager=staleness_manager,
            config=self.config,
            logger=self.logger,
        )

    def destroy(self):
        self.workflow_executor.destroy()
        self.executor.shutdown()

    def set_version(self, version):
        with self.lock:
            self._version = version

    def get_version(self):
        with self.lock:
            return self._version

    def choose_server(self) -> str:
        if self.config.schedule_policy == "round_robin":
            server = self.addresses[self.server_idx]
            self.server_idx = (self.server_idx + 1) % len(self.addresses)
            return server
        raise NotImplementedError("Only round-robin scheduling is implemented.")

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Async version of generate using aiohttp."""
        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        stop = gconfig.stop

        if gconfig.n_samples != 1:
            raise ValueError(
                "RemoteSGLangEngine does not support n_samples > 1. "
                "Please call generate for multiple times with n_samples = 1."
            )

        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids), gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(
                f"max_new_tokens ({max_new_tokens}) is non-positive! "
                f"max_tokens={gconfig.max_tokens}, prompt_len={len(req.input_ids)}, "
                f"max_new_tokens={gconfig.max_new_tokens}."
            )

        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if stop:
            sample_params["stop"] = stop

        payload = {
            "input_ids": req.input_ids.copy(),
            "image_data": req.image_data,  # ImageObject or str
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
        }
        if self.lora_init:
            # Use the same lora name because we are unable to change
            # the lora_name of an inflight request during weight update.
            # If the lora_name mismatch, there'll be an error.
            payload["lora_path"] = f"lora_1"

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []  # Per-token policy version
        proximal_logprobs_t = [] if self.config.enable_segment_wise_ppo else None
        prompt_len = len(req.input_ids)  # Store original prompt length for proximal_t updates

        # A single "rid" shares the same sever to allow KV cache reuse
        if req.rid in self.rid_to_address:
            server_addr = self.rid_to_address[req.rid]
        else:
            server_addr = self.choose_server()
            if len(self.rid_queue) >= RID_CACHE_SIZE:
                # Remove the oldest entry if cache is full
                oldest_rid = self.rid_queue.pop(0)
                self.rid_to_address.pop(oldest_rid, None)
            self.rid_to_address[req.rid] = server_addr
            self.rid_queue.append(req.rid)

        # Create a new session because we don't know whether this method
        # is called in the workflow thread or the main thread.
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                sock_connect=self.config.request_timeout,
                connect=self.config.request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )

        # Deal with rollout interruption
        # "abort" is the stop reason for later v0.4.9.post2 after
        # we call the pause_generation endpoint
        stop_reason = None
        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):
            # Request is interrupted, wait for some time to avoid interfering
            # with update weights requests
            while self.workflow_executor.paused.is_set():
                await asyncio.sleep(0.5)

            # loop until the generation is complete
            result = await arequest_with_retry(
                session=session,
                addr=server_addr,
                endpoint="/generate",
                payload=payload,
                method="POST",
                max_retries=self.config.request_retries,
                timeout=self.config.request_timeout,
            )

            meta_info = result["meta_info"]
            # Check if generation is complete
            finish_reason = meta_info["finish_reason"]
            stop_reason = finish_reason["type"]
            if (
                stop_reason == "abort"
                and finish_reason.get("message") == "Abort before prefill"
            ):
                continue

            # Parse response
            output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
            output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

            # For segment-wise PPO: Handle proximal_logprobs_t correctly (only if enabled)
            if proximal_logprobs_t is not None:
                if len(accumulated_output_tokens) == 0:
                    # First iteration: No previous output to update
                    # proximal_t for new tokens will be their generation logprobs (appended below)
                    pass
                else:
                    # Abort-resume iteration: Update proximal_t for previously generated tokens
                    # input_logprobs contains logprobs under NEW policy for tokens after logprob_start_len
                    input_logprobs = [x[0] for x in meta_info["input_token_logprobs"]]

                    # logprob_start_len was set to len(input_ids) - 1 in previous iteration
                    # So input_logprobs[0] is for position (logprob_start_len)
                    # input_logprobs[1] is for position (logprob_start_len + 1), etc.

                    # We want logprobs for previous output tokens (positions prompt_len onwards)
                    # Previous iteration set logprob_start_len = prompt_len + prev_output_len - 1
                    # So input_logprobs structure:
                    # [0]: logprob for position (prompt_len + prev_output_len - 1) - last prompt token or first output
                    # [1:]: logprobs for subsequent positions

                    # Calculate how many previous output tokens we have
                    prev_output_len = len(accumulated_output_tokens)
                    if prev_output_len > 0:
                        # Extract proximal_t for previous output tokens (under new policy)
                        prev_output_proximal_t = input_logprobs[1:1+prev_output_len]

                        # REPLACE (not extend) the existing proximal_t values
                        for i, logprob in enumerate(prev_output_proximal_t):
                            if i < len(proximal_logprobs_t):
                                proximal_logprobs_t[i] = logprob

            # Accumulate new output tokens
            accumulated_output_tokens.extend(output_tokens)
            accumulated_output_logprobs.extend(output_logprobs)
            # Track the current policy version for each generated token
            accumulated_versions.extend([self.get_version()] * len(output_tokens))

            # For newly generated tokens, proximal_t is their generation logprobs (for now)
            # These will be updated in next abort-resume iteration if policy changes
            if proximal_logprobs_t is not None:
                proximal_logprobs_t.extend(output_logprobs)

            # Set logprob_start_len to current input length, then update input_ids
            payload["logprob_start_len"] = len(payload["input_ids"]) - 1
            payload["input_ids"] += output_tokens
            sample_params["max_new_tokens"] -= len(output_tokens)

        if stop_reason == "abort":
            # If stop_reason is "abort", the only reason we exit the loop is
            # len(accumulated_output_tokens) >= gconfig.max_new_tokens
            # so the actual reason is length
            stop_reason = "length"
        await session.close()
        latency = time.perf_counter() - start_time

        response = ModelResponse(
            input_tokens=req.input_ids,
            input_images=req.image_data,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            proximal_logprobs_t=proximal_logprobs_t if proximal_logprobs_t is not None else [],
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
            tokenizer=req.tokenizer,
            processor=req.processor,
        )
        return response

    def init_weights_update_group(self, meta: WeightUpdateMeta) -> Future[None]:
        # No need to init group for non-NCCL update
        assert meta.type == current_platform.communication_backend
        assert (
            not self.distributed_weight_update_initialized
        ), "Weight update group already initialized."

        fut = self.executor.submit(
            init_weights_update_group_remote,
            meta,
            self.addresses,
            self.config.request_timeout,
        )

        def callback(fut):
            self.logger.info(
                f"Initialized NCCL group for distributed weight update for {meta.nccl_group_name}."
            )
            self.distributed_weight_update_initialized = True

        fut.add_done_callback(callback)

        return fut

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: List[ParamSpec]
    ) -> Future[None]:
        assert meta.type == current_platform.communication_backend

        fut = self.executor.submit(
            update_weights_from_distributed,
            meta,
            param_specs,
            self.addresses,
            self.config.request_timeout,
        )

        return fut

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        assert meta.type == "disk"

        tik = time.perf_counter()

        # Use ProcessPool to bypass python GIL for running async coroutines
        if self.config.experiment_name is None or self.config.trial_name is None:
            raise RuntimeError(
                f"Experiment and trial names must be set for disk-based weight updates."
            )
        endpoints = ["update_weights_from_disk"]
        payloads = [dict(model_path=str(meta.path), abort_all_requests=True)]
        lora_name = "lora_1"
        if meta.use_lora:
            endpoints = []
            payloads = []
            if self.lora_init:
                endpoints.append("unload_lora_adapter")
                payloads.append(dict(lora_name=lora_name))
            else:
                self.lora_init = True
            endpoints.append("load_lora_adapter")
            payloads.append(dict(lora_name=lora_name, lora_path=str(meta.path)))

        fut = self.executor.submit(
            update_weights_from_disk,
            self.config.experiment_name,
            self.config.trial_name,
            self.get_version(),
            self.addresses,
            meta.path,
            self.config.request_retries,
            self.config.request_timeout,
            endpoints,
            payloads,
        )

        def callback(fut):
            respond_time = fut.result()
            self.logger.info(
                f"Loading weights from disk done in {(time.perf_counter() - tik):.2f}s. "
                f"Respond time: {respond_time:.2f}s."
            )
            shutil.rmtree(meta.path, ignore_errors=True)

        fut.add_done_callback(callback)

        return fut

    def submit(
        self,
        data: Dict[str, Any],
        workflow: Optional[RolloutWorkflow] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> None:
        return self.workflow_executor.submit(
            data,
            workflow=workflow,
            workflow_builder=workflow_builder,
            should_accept=should_accept,
        )

    def wait(self, count: int, timeout: float | None = None) -> Dict[str, Any]:
        return self.workflow_executor.wait(count, timeout=timeout)

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> Dict[str, Any]:
        return self.workflow_executor.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_builder=workflow_builder,
            should_accept=should_accept,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: Optional[RolloutWorkflow] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ):
        return self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_builder=workflow_builder,
            should_accept=should_accept,
        )

    def pause_generation(self):
        """Pause the generation of inference engine.

        Used during updating weights from distributed or disk.
        """
        for addr in self.addresses:
            res = requests.post(f"http://{addr}/pause_generation")
            res.raise_for_status()

        # The above http request may require some time to be scheduled and executed.
        # The following line waits until all requests are indeed dropped.
        time.sleep(self.config.pause_grace_period)

    def continue_generation(self):
        """Continue the generation of inference engine."""
        for addr in self.addresses:
            res = requests.post(f"http://{addr}/continue_generation")
            res.raise_for_status()

    def pause(self):
        """Pause request submission for async rollout. Used during evaluation to prevent data over generation."""
        return self.workflow_executor.pause()

    def resume(self):
        """Resume request submission for async rollout."""
        return self.workflow_executor.resume()

    def recompute_output_logprobs_sync(
        self,
        input_ids: List[int],
        start_index: int = 0,
    ) -> List[float]:
        """Recompute output logprobs for a given sequence (for segment-wise PPO).

        This is a synchronous method that makes a request to the SGLang server to
        recompute the log probabilities for the output tokens.

        Args:
            input_ids: Complete sequence of token IDs (prompt + generated tokens)
            start_index: Index to start computing logprobs from (default: 0)

        Returns:
            List of logprobs for tokens after start_index

        Note:
            This is used by ProximalRecomputer to recompute proximal_logprobs_t
            for v-1 samples before weight updates.
        """
        # Choose a server (round-robin)
        server_addr = self.choose_server()

        # Prepare payload for logprob recomputation
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,  # Greedy sampling for deterministic logprobs
                "max_new_tokens": 1,  # We only need logprobs, not generation
            },
            "return_logprob": True,
            "stream": False,
            "logprob_start_len": start_index + 1,  # SGLang parameter for logprob computation
        }

        try:
            response = requests.post(
                f"http://{server_addr}/generate",
                json=payload,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()
            result = response.json()

            # Extract logprobs from response
            meta_info = result.get("meta_info", {})
            input_token_logprobs = meta_info.get("input_token_logprobs", [])

            # Return logprobs for tokens after start_index
            # input_token_logprobs[i] is the logprob of input_ids[i+1] given input_ids[:i+1]
            return input_token_logprobs[start_index:]
        except Exception as e:
            self.logger.error(f"Failed to recompute logprobs: {e}")
            raise

    def update_weights(self, *args, **kwargs):
        """Update weights with proximal logprob recomputation (segment-wise PPO).

        This method wraps the normal weight update process and calls proximal
        recomputation BEFORE updating weights.
        """
        # Recompute proximal logprobs for v-1 samples before weight update
        self.workflow_executor.recompute_proximal_logprobs()

        # Then proceed with normal weight update (subclasses should implement actual update)
        # For RemoteSGLangEngine, weight updates happen via update_weights_from_disk() or
        # update_weights_from_distributed(), so this is just a hook point


def update_weights_from_disk(
    experiment_name,
    trial_name,
    model_version,
    addresses,
    path,
    request_retries,
    request_timeout,
    endpoints,
    payloads,
):
    async def _fn():
        update_name = names.update_weights_from_disk(
            experiment_name, trial_name, model_version
        )
        save_timestamp = float(name_resolve.wait(update_name, timeout=120))
        load_timestamp = datetime.now().timestamp()
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=request_timeout,
                sock_connect=request_timeout,
                connect=request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )
        for endpoint, payload in zip(endpoints, payloads):
            jobs = [
                arequest_with_retry(
                    addr=addr,
                    session=session,
                    endpoint=f"/{endpoint}",
                    payload=payload,
                    method="POST",
                    max_retries=request_retries,
                    timeout=request_timeout,
                )
                for addr in addresses
            ]
            await asyncio.gather(*jobs)
        await session.close()
        return load_timestamp - save_timestamp

    return uvloop.run(_fn())


def init_weights_update_group_remote(
    meta: WeightUpdateMeta,
    addresses: List[str],
    request_timeout,
):
    async def _fn():
        await asyncio.gather(
            *[
                ainit_weights_update_group(addr, i, meta, request_timeout)
                for i, addr in enumerate(addresses)
            ]
        )

    return uvloop.run(_fn())


def update_weights_from_distributed(
    meta: WeightUpdateMeta,
    param_specs: List[ParamSpec],
    addresses: List[str],
    request_timeout,
):
    async def _fn():
        await asyncio.gather(
            *[
                arequest_with_retry(
                    addr=addr,
                    endpoint="/update_weights_from_distributed",
                    payload={
                        "names": [pspec.name for pspec in param_specs],
                        "dtypes": [pspec.dtype for pspec in param_specs],
                        "shapes": [pspec.shape for pspec in param_specs],
                        "group_name": meta.nccl_group_name,
                        "abort_all_requests": True,
                    },
                    method="POST",
                    max_retries=1,
                    timeout=request_timeout,
                )
                for addr in addresses
            ]
        )

    return uvloop.run(_fn())


async def ainit_weights_update_group(
    addr: str,
    server_idx: int,
    meta: WeightUpdateMeta,
    request_timeout: float,
):
    assert meta.alloc_mode is not None
    if meta.alloc_mode.gen.pp_size != 1:
        raise NotImplementedError(
            "NCCL weight update with PP size > 1 is not implemented yet."
        )
    rank_offset = 1 + server_idx * meta.alloc_mode.gen.tp_size
    payload = {
        "master_address": meta.nccl_master_address,
        "master_port": str(meta.nccl_master_port),
        "rank_offset": rank_offset,
        "world_size": meta.alloc_mode.gen.world_size + 1,
        "backend": current_platform.communication_backend,
        "group_name": meta.nccl_group_name,
    }
    res = await arequest_with_retry(
        addr=addr,
        endpoint="/init_weights_update_group",
        payload=payload,
        method="POST",
        max_retries=1,
        timeout=request_timeout,
    )
    assert res["success"]
