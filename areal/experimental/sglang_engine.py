import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Callable, Dict, List, Optional

import sglang as sgl
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import (
    ModelRequest,
    ModelResponse,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow, WorkflowExecutor
from areal.api.workflow_factory import create_workflow_executor
from areal.core.staleness_manager import StalenessManager
from areal.utils import logging, name_resolve, names, pkg_version

logger = logging.getLogger(__name__)

if pkg_version.is_available("sglang"):
    if pkg_version.is_version_greater_or_equal("sglang", "0.4.4"):
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "output_ids"
    else:
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "token_ids"

ROLLOUT_POLL_WAIT_TIME = 0.4
RID_CACHE_SIZE = 128

"""
Local SGLang Inference Engine
SGLangEngine currently only supports single-controller. Cannot be used in SPMD
"""


class SGLangEngine(InferenceEngine):

    def __init__(
        self,
        config: InferenceEngineConfig,
        engine_args: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.engine_args = engine_args or {}

        self._version = 0

        # Workflow executor will be initialized in initialize()
        self.workflow_executor: WorkflowExecutor

    def initialize(
        self,
        engine_id: Optional[str] = None,
        train_data_parallel_size: int | None = None,
    ):
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[SGLang Local Engine Rank {engine_id}]")

        self.engine = sgl.Engine(**self.engine_args)

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

    def set_version(self, version):
        self._version = version

    def get_version(self):
        return self._version

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Async version of generate using local sglang engine."""
        if not hasattr(self, "engine") or self.engine is None:
            raise RuntimeError(
                "Local SGLang engine is not initialized, cannot generate."
            )

        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

        if gconfig.n_samples != 1:
            raise ValueError(
                "LocalSGLangEngine does not support n_samples > 1. "
                "Please call generate for multiple times with n_samples = 1."
            )
        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
        }

        completions = ""
        prompt = req.text if req.text else None
        input_ids = req.input_ids if req.input_ids else None

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []
        proximal_logprobs_t = [] if self.config.enable_segment_wise_ppo else None
        stop_reason = "length"
        while (
            stop_reason != "stop"
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):

            try:
                outputs = await self.engine.async_generate(
                    prompt=prompt,
                    input_ids=input_ids,
                    sampling_params=sample_params,
                    return_logprob=True,
                )

                completions += outputs["text"]
                if prompt is None:
                    prompt = outputs["text"]
                else:
                    prompt += outputs["text"]

                meta_info = outputs["meta_info"]
                output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
                output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

                finish_reason = meta_info.get("finish_reason", {})
                stop_reason = finish_reason.get("type", "length")

                accumulated_output_tokens.extend(output_tokens)
                accumulated_output_logprobs.extend(output_logprobs)
                accumulated_versions.extend([-1] * len(output_tokens))

                # For segment-wise PPO: Initialize proximal_t to generation logprobs
                # These will be recomputed later by ProximalRecomputer for v-1 tokens
                if proximal_logprobs_t is not None:
                    proximal_logprobs_t.extend(output_logprobs)

            except Exception as e:
                raise RuntimeError(f"Local SGLang engine generation failed: {e}")

        latency = time.perf_counter() - start_time

        return ModelResponse(
            completions=completions,
            input_tokens=req.input_ids if req.input_ids else [],
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            proximal_logprobs_t=proximal_logprobs_t if proximal_logprobs_t is not None else [],
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,
        )

    def update_weights(self, meta):
        executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(self._update_weights, meta)

    def _update_weights(self, meta: WeightUpdateMeta):
        if not hasattr(self, "engine") or self.engine is None:
            raise RuntimeError(
                "Local SGLang engine is not initialized, cannot update weights."
            )

        # Recompute proximal logprobs BEFORE weight update (segment-wise PPO)
        self.workflow_executor.recompute_proximal_logprobs()

        if meta.type == "disk":
            try:
                update_name = names.update_weights_from_disk(
                    self.config.experiment_name,
                    self.config.trial_name,
                    meta.model_version,
                )
                save_timestamp = int(name_resolve.wait(update_name, timeout=120))
                load_timestamp = time.time_ns()
                logger.info(
                    f"Begin update weights from {meta.path}, responded in {(load_timestamp - save_timestamp)/1e6:.2f} ms"
                )
                # Update weights from disk,
                self.engine.update_weights_from_disk(model_path=meta.path)

                logger.info(
                    f"Loading weights done in {(time.time_ns() - load_timestamp)/1e6:.2f} ms"
                )
                self.set_version(meta.model_version)
            except Exception as e:
                logger.error(f"Failed to update weights: {e}")
                raise
        else:
            raise NotImplementedError(f"Unsupported weight update type: {meta.type}")

    def recompute_output_logprobs_sync(
        self,
        input_ids: List[int],
        start_index: int = 0,
    ) -> List[float]:
        """Recompute output logprobs for a given sequence (for segment-wise PPO).

        This is a synchronous method that uses the local SGLang engine to
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
        if not hasattr(self, "engine") or self.engine is None:
            raise RuntimeError(
                "Local SGLang engine is not initialized, cannot recompute logprobs."
            )

        try:
            # Use SGLang's sync API for logprob computation
            # Generate with greedy decoding (temperature=0) to get deterministic logprobs
            outputs = self.engine.generate(
                input_ids=input_ids,
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": 1,  # We only need logprobs, not generation
                },
                return_logprob=True,
            )

            # Extract logprobs from output
            meta_info = outputs.get("meta_info", {})
            input_token_logprobs = meta_info.get("input_token_logprobs", [])

            # Return logprobs for tokens after start_index
            return input_token_logprobs[start_index:]
        except Exception as e:
            logger.error(f"Failed to recompute logprobs: {e}")
            raise

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

    def wait(self, count: int, timeout: float | None = None):
        return self.workflow_executor.wait(count, timeout=timeout)

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ):
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

    def pause(self):
        """Pause request submission for async rollout. Used during evaluation to prevent data over generation."""
        return self.workflow_executor.pause()

    def resume(self):
        """Resume request submission for async rollout."""
        return self.workflow_executor.resume()
