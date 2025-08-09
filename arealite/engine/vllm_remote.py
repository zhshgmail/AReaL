import asyncio
import dis
import os
import random
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import aiohttp
import requests
import torch.distributed as dist
import uvloop
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import InferenceEngineConfig
from arealite.api.engine_api import InferenceEngine
from arealite.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    LLMResponse,
    RolloutStat,
    WeightUpdateMeta,
)
from arealite.utils.data import concat_padded_tensors
from arealite.utils.http import arequest_with_retry, get_default_connector
from realhf.base import logging, name_resolve, names, pkg_version

if TYPE_CHECKING:
    from arealite.api.workflow_api import RolloutWorkflow
logger = logging.getLogger(__name__)

VLLM_TOKEN_OUTPUT_IDENTIFIER = "token_ids"

ROLLOUT_POLL_WAIT_TIME = 0.05
RID_CACHE_SIZE = 128


class RemotevLLMEngine(InferenceEngine):

    def __init__(self, config: InferenceEngineConfig):
        config.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []

        self.addresses = os.getenv("AREAL_LLM_SERVER_ADDRS").split(",")
        if not self.addresses:
            raise RuntimeError("No configured vLLM servers.")
        logger.info("Waiting for server ready...")
        for addr in self.addresses:
            self._wait_for_server(addr)
        logger.info("Servers are all ready!")

        self.server_idx = random.randint(0, len(self.addresses) - 1)

        qsize = config.queue_size or config.max_concurrent_rollouts * 16
        self.input_queue = Queue(maxsize=qsize)
        self.output_queue = Queue(maxsize=qsize)
        self.result_cache = []

        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()

        self.rollout_stat = RolloutStat()

        self._version = 0

    def _wait_for_server(self, address, sleep_time=1):
        base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.config.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(sleep_time)
        raise RuntimeError("server launch failed")

    def check_health(self, base_url):
        # Check server endpoint
        try:
            response = requests.get(
                f"{base_url}/metrics",
                timeout=30,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Check health failed: {e}")
            return False

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec = None):
        self.rollout_tasks: Dict[str, asyncio.Task] = {}

        self.executor = ProcessPoolExecutor(max_workers=1)
        self.rollout_thread = threading.Thread(target=self._rollout_thread)
        self.rollout_thread.start()

    def destroy(self):
        self.executor.shutdown()
        self.exiting.set()
        self.rollout_thread.join()

    def set_version(self, version):
        with self.lock:
            self._version = version

    def get_version(self):
        with self.lock:
            return self._version

    def _rollout_thread(self):
        """Thread that runs the rollout loop."""
        try:
            uvloop.run(self._rollout_thread_async())
        except Exception as e:
            traceback.print_exc()

    async def _rollout_thread_async(self):
        rollout_tasks = self.rollout_tasks
        rid = 0

        # NOTE: session is not thread-safe, but we only submit requests in the sub-thread.
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                sock_connect=self.config.request_timeout,
                connect=self.config.request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )

        try:
            while not self.exiting.is_set():
                # Check capacity
                capacity = self.get_capacity()
                # Create new rollout task
                while (
                    capacity > 0
                    and not self.paused.is_set()
                    and self.input_queue.qsize() > 0
                ):
                    data, workflow = self.input_queue.get_nowait()
                    logger.debug(f"Get data from puller: {data}")
                    task = asyncio.create_task(
                        workflow.arun_episode(self, data), name=str(rid)
                    )
                    with self.lock:
                        rollout_tasks[str(rid)] = task
                        self.rollout_stat.submitted += 1
                        self.rollout_stat.running += 1
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Submit rollout rid {rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
                            )
                    capacity -= 1
                    rid += 1
                # Wait for rollout completion
                with self.lock:
                    tasks = list(rollout_tasks.values())
                done = []
                if tasks:
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=ROLLOUT_POLL_WAIT_TIME,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                # Collect done results
                for task in done:
                    traj = await task
                    traj: TensorDict
                    task_rid = task.get_name()
                    with self.lock:
                        rollout_tasks.pop(task_rid)
                        self.rollout_stat.accepted += 1

                    try:
                        self.output_queue.put_nowait(traj)
                    except Full:
                        raise RuntimeError(
                            "Output queue full. Please increase queue_size."
                        )

                    with self.lock:
                        self.rollout_stat.running -= 1
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Finish rollout {task_rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
                            )
                await asyncio.sleep(1)
        except Exception:
            traceback.print_exc()
        finally:
            # Cancel remaining tasks
            with self.lock:
                for task in rollout_tasks.values():
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

    def choose_server(self) -> str:
        with self.lock:
            if self.config.schedule_policy == "round_robin":
                server = self.addresses[self.server_idx]
                self.server_idx = (self.server_idx + 1) % len(self.addresses)
                return server
        raise NotImplementedError("Only round-robin scheduling is implemented.")

    async def agenerate(self, req: LLMRequest, tokenizer) -> LLMResponse:
        """Async version of generate using aiohttp."""
        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

        if gconfig.n_samples != 1:
            raise ValueError(
                "RemotevLLMEngine does not support n_samples > 1. "
                "Please call generate for multiple times with n_samples = 1."
            )

        # Convert stop_token_ids to strings if provided
        stop_sequences = None
        if stop_token_ids:
            stop_sequences = [tokenizer.decode([token_id]) for token_id in stop_token_ids]

        # NOTE: rid should NOT be passed in payload  
        payload = {
            "prompt": req.input_ids,
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "logprobs": 0,
            "stream": False,
        }
        
        # Add stop parameter only if we have valid stop sequences
        if stop_sequences:
            payload["stop"] = stop_sequences

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []

        # Deal with rollout interruption
        stop_reason = "length"
        iteration_count = 0

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

        while (
            stop_reason != "stop"
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):
            iteration_count += 1
            logger.info(f"🔄 While loop iteration {iteration_count} START - stop_reason: '{stop_reason}', accumulated_tokens: {len(accumulated_output_tokens)}/{gconfig.max_new_tokens}")
            logger.info(f"🔍 DEBUG: About to call VLLM API with payload max_tokens={payload.get('max_tokens')}")
            
            # loop until the generation is complete
            result = await arequest_with_retry(
                session=self.session,
                addr=server_addr,
                endpoint="/v1/completions",
                payload=payload,
                method="POST",
                max_retries=self.config.request_retries,
                timeout=self.config.request_timeout,
            )

            logger.info(f"🔍 DEBUG: Got API response, parsing...")
            # Parse response
            meta_info = result["choices"][0]
            vllm_tokens = meta_info["logprobs"]["tokens"]
            output_tokens_before = meta_info['text']
            output_tokens = tokenizer.convert_tokens_to_ids(vllm_tokens)
            output_logprobs = meta_info["logprobs"]["token_logprobs"]

            # Update accumulated outputs
            accumulated_output_tokens.extend(output_tokens)
            accumulated_output_logprobs.extend(output_logprobs)
            # FIXME: Update with actual server versions
            accumulated_versions.extend([-1] * len(output_tokens))

            # Check if generation is complete
            stop_reason = meta_info["finish_reason"]
            logger.info(f"✅ Iteration {iteration_count} DONE - generated {len(output_tokens)} tokens, new_finish_reason: '{stop_reason}', total_tokens: {len(accumulated_output_tokens)}")
            logger.info(f"🔍 DEBUG: Checking continue condition: stop_reason='{stop_reason}' != 'stop' = {stop_reason != 'stop'}, tokens={len(accumulated_output_tokens)} < {gconfig.max_new_tokens} = {len(accumulated_output_tokens) < gconfig.max_new_tokens}")

            # Update payload for next iteration if needed
            if stop_reason != "stop" and len(accumulated_output_tokens) < gconfig.max_new_tokens:
                logger.info(f"🚀 CONTINUING generation - reason: stop_reason='{stop_reason}', tokens={len(accumulated_output_tokens)}<{gconfig.max_new_tokens}")
                # Update prompt with generated tokens for next request
                payload["prompt"] = req.input_ids + accumulated_output_tokens
                payload["max_tokens"] = gconfig.max_new_tokens - len(accumulated_output_tokens)
                # Keep the stop parameter unchanged for subsequent requests
                if stop_sequences:
                    payload["stop"] = stop_sequences
            else:
                logger.info(f"🛑 STOPPING generation - reason: stop_reason='{stop_reason}', tokens={len(accumulated_output_tokens)}/{gconfig.max_new_tokens}")

<<<<<<< HEAD
        logger.info(f"🔍 DEBUG: Exited while loop - final stop_reason='{stop_reason}', final_tokens={len(accumulated_output_tokens)}")

        # Log how many iterations the while loop performed (only if not single successful iteration)
        if not (iteration_count == 1 and stop_reason == "stop"):
            logger.info(f"🏁 FINAL RESULT: {iteration_count} iterations, finish_reason: '{stop_reason}', total_tokens: {len(accumulated_output_tokens)}")
=======
        # logger.info(f"🔍 DEBUG: Exited while loop - final stop_reason='{stop_reason}', final_tokens={len(accumulated_output_tokens)}")

        # Log how many iterations the while loop performed (only if not single successful iteration)
        # if not (iteration_count == 1 and stop_reason == "stop"):
            # logger.info(f"🏁 FINAL RESULT: {iteration_count} iterations, finish_reason: '{stop_reason}', total_tokens: {len(accumulated_output_tokens)}")
>>>>>>> a6ba4c9 (solving oom problem)
        
        latency = time.perf_counter() - start_time

        return LLMResponse(
            input_tokens=req.input_ids,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
        )

    def update_weights(self, meta: WeightUpdateMeta):
        if meta.type == "disk":
            rank = int(os.getenv("RANK"))
            if rank == 0 :
                # only restart vllm engine from rank 0
                logger.info('===================> rank 0 restart vllm engine.')
                dist.barrier()
                logger.info('===================> rank 0 finish vllm engine.')

                job_name_remained = str(os.getenv("JOB_NAME_REMAINED"))
                count_remained = int(os.getenv("COUNT_REMAINED"))
                gpu_remained = int(os.getenv("GPU_REMAINED"))
                get_all_pid = str(os.getenv("GET_ALL_PID"))
                get_all_pid = get_all_pid.split(',')
                cmd_remained = str(os.getenv('CMD_REMAINED'))
                cmd_remained_list = ['python3'+x for x in cmd_remained.split('python3')]
                cmd_remained_list_origin = [x.replace(',',' ') for x in cmd_remained_list[1:]]
                cmd_remained_list_origin_new_path = []

                for path in cmd_remained_list_origin:
                    arg_list = path.split(' ')
                    found = False
                    for index, each_args in enumerate(arg_list):
                        if found:
                            found = False
                            arg_list[index] = meta.path
                        if each_args=='--model':
                            found = True
                    cmd_remained_list_origin_new_path.append(' '.join(arg_list))

                # stop the existed vllm engine
                from ..launcher.local import terminate_process_and_children, LocalLauncher
                import time
                import gc
                import torch
                
                # First, try graceful shutdown
                logger.info('Attempting graceful shutdown of vLLM processes...')
                for pid in get_all_pid:
                    try:
                        terminate_process_and_children(int(pid), signal="SIGTERM")
                    except Exception as e:
                        logger.warning(f"Failed to terminate process {pid}: {e}")

                # Wait for processes to terminate and GPU memory to be released
                time.sleep(5)
                
                # Force cleanup any remaining processes
                logger.info('Force killing any remaining vLLM processes...')
                for pid in get_all_pid:
                    try:
                        terminate_process_and_children(int(pid), signal="SIGKILL")
                    except Exception as e:
                        logger.debug(f"Process {pid} already terminated: {e}")
<<<<<<< HEAD
=======
               # Extra sweep: kill any residual vLLM-related processes for current user
                try:
                    import psutil  # type: ignore
                    current_user = None
                    try:
                        current_user = psutil.Process(os.getpid()).username()
                    except Exception:
                        pass

                    # Only target definitive vLLM server/worker processes. Avoid generic 'api_server' matches.
                    vllm_markers = [
                        "vllm",
                        "vllm.entrypoints",
                        "vllm_worker",
                    ]
                    exclude_patterns = [
                        "torchrun",
                        "trainer",
                        "torch.distributed.run",
                        "torch.distributed.launch",
                        "torch.distributed.elastic",
                        "gsm8k_grpo.py",
                    ]
                    killed = []
                    skipped = []
                    for proc in psutil.process_iter(["pid", "name", "cmdline", "username"]):
                        try:
                            pid = proc.info.get("pid")
                            if not pid or pid == os.getpid():
                                continue
                            if current_user and proc.info.get("username") and proc.info.get("username") != current_user:
                                continue
                            name = (proc.info.get("name") or "").lower()
                            cmdline_list = proc.info.get("cmdline") or []
                            cmdline = " ".join(cmdline_list).lower()
                            if any(ex in name or ex in cmdline for ex in exclude_patterns):
                                # Skip processes that look like trainer/torchrun
                                skipped.append((pid, cmdline, "excluded"))
                                continue
                            # Only kill if it's very likely a vLLM server/worker:
                            #  - contains 'vllm' and (api_server/entrypoints/worker) or has a --model argument
                            looks_like_vllm = ("vllm" in name or "vllm" in cmdline)
                            has_entry = any(m in cmdline for m in ("api_server", "vllm.entrypoints", "vllm_worker"))
                            has_model_arg = "--model" in cmdline
                            if looks_like_vllm and (has_entry or has_model_arg):
                                try:
                                    proc.terminate()
                                except Exception:
                                    pass
                                try:
                                    proc.wait(timeout=3)
                                except Exception:
                                    pass
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                                killed.append((pid, cmdline))
                            else:
                                # Not confident it's a vLLM process; skip but record for debugging
                                if looks_like_vllm or has_entry:
                                    skipped.append((pid, cmdline, "not-confident-vllm"))
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            continue
                    if killed:
                        logger.info("Extra cleanup terminated vLLM-related processes:\n" +
                                    "\n".join([f"PID {p}: {c}" for p, c in killed]))
                    if skipped:
                        logger.debug("Extra cleanup skipped processes (by rule):\n" +
                                     "\n".join([f"PID {p} ({reason}): {c}" for p, c, reason in skipped]))
                except Exception as e:
                    logger.warning(f"Extra vLLM process sweep failed: {e}")

                # Ensure processes are fully gone and files unlocked
                time.sleep(3)


>>>>>>> a6ba4c9 (solving oom problem)
                # Synchronize only current process CUDA operations

              

                # torch.cuda.synchronize()
                # torch.cuda.empty_cache()
                # logger.info('Empty cache...')
                # gc.collect()
                # torch.cuda.empty_cache()
                # torch.cuda.reset_peak_memory_stats()
                # logger.info('Reser peak memory...')

                launcher = LocalLauncher(self.config.experiment_name, self.config.trial_name, self.config.fileroot)
                # restart


                # TODO replace model path to meta.path # FIXME

                launcher.reset_gpu_counter()
                launcher.submit_array(
                    job_name=job_name_remained,
                    cmd=cmd_remained_list_origin_new_path,
                    count=count_remained,
                    gpu=gpu_remained,
                    reset_gpu_counter=True,
                    new_env=True
                )
                logger.info('Wait for server ready...')
                for addr in self.addresses:
                    old_setup_time  = self.config.setup_timeout
                    self.config.setup_timeout = 3600
                    self._wait_for_server(addr, sleep_time=5)
                    self.config.setup_timeout = old_setup_time
                logger.info('Servers are all ready.')

                self.set_version(meta.model_version)
                dist.barrier()
                logger.info(
                    f"===================> rank {rank} LLM inference server relaunched."
                )
            else:
                logger.info(
                    f"===================> rank {rank} wait for restart vLLM, start barrier."
                )
                dist.barrier()

                logger.info(
                    f"===================> rank {rank} wait for restart vLLM, finish barrier."
                )

                dist.barrier()
                logger.info(
                    f"===================> rank {rank} LLM inference server relaunched."
                )

            # # Update weights from disk
            # # Use ProcessPool to bypass python GIL for running async coroutines
            # fut = self.executor.submit(
            #     update_weights_from_disk,
            #     self.config.experiment_name,
            #     self.config.trial_name,
            #     meta.model_version,
            #     self.addresses,
            #     meta.path,
            #     self.config.request_retries,
            #     self.config.request_timeout,
            # )
            #
            # def callback(fut):
            #     self.set_version(meta.model_version)
            #
            # fut.add_done_callback(callback)
            # return fut
        else:
            raise NotImplementedError(f"Unsupported weight update type: {meta.type}")

    def get_capacity(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        max_concurrent_rollouts = max(
            1, self.config.max_concurrent_rollouts // world_size
        )
        capacity = max_concurrent_rollouts - len(self.rollout_tasks)
        # Staleness control
        version = self.get_version()
        ofp = self.config.max_head_offpolicyness
        with self.lock:
            sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
        consumer_bs = max(1, self.config.consumer_batch_size // world_size)
        capacity = min(capacity, (ofp + version + 1) * consumer_bs - sample_cnt)
        return capacity

    def submit(self, data: Dict[str, Any], workflow: "RolloutWorkflow") -> None:
        try:
            self.input_queue.put_nowait((data, workflow))
        except Full:
            raise RuntimeError("Input queue full. Please increase queue_size.")

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        tik = time.perf_counter()
        accepted = len(self.result_cache)
        timeout = timeout or float(7 * 24 * 3600)
        while (
            accepted < count
            and not self.exiting.is_set()
            and time.perf_counter() - tik < timeout
        ):
            try:
                result = self.output_queue.get(timeout=ROLLOUT_POLL_WAIT_TIME)
                if should_accept is None or should_accept(result):
                    self.result_cache.append(result)
                    accepted += 1
                else:
                    with self.lock:
                        self.rollout_stat.accepted -= 1
            except Empty:
                pass
        if self.exiting.is_set():
            raise RuntimeError("Rollout engine is exiting, cannot wait for results.")
        if accepted < count:
            raise TimeoutError(
                f"Timed out waiting for {count} rollouts, " f"only received {accepted}."
            )
        results, self.result_cache = (
            self.result_cache[:count],
            self.result_cache[count:],
        )
        return concat_padded_tensors(results)

    def rollout_batch(
        self, data: List[Dict[str, Any]], workflow: "RolloutWorkflow"
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        for item in data:
            self.submit(item, workflow)
        return self.wait(count=len(data))

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: "RolloutWorkflow",
    ):
        if not hasattr(self, "data_generator"):
            self.data_generator = iter(dataloader)
        assert dataloader.batch_size is not None
        while True:
            # Submit at least two batches to allow maximum overlap
            if (
                self.get_capacity() + dataloader.batch_size > 0
                and self.input_queue.qsize() + dataloader.batch_size
                < self.input_queue.maxsize
            ):
                try:
                    data = next(self.data_generator)
                except StopIteration:
                    self.data_generator = iter(dataloader)
                    data = next(self.data_generator)
                for item in data:
                    self.submit(item, workflow=workflow)
            try:
                return self.wait(dataloader.batch_size, timeout=1)
            except TimeoutError:
                pass

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()


async def aupdate_weights_from_disk(
    session, addr, path: str, request_retries: int, request_timeout: float
):
    tik = time.time()
    res = await arequest_with_retry(
        addr=addr,
        session=session,
        endpoint="/update_weights_from_disk",
        payload=dict(model_path=str(path), allow_interrupt=True),
        method="POST",
        max_retries=request_retries,
        timeout=request_timeout,
    )
    assert res["success"]
    if "num_paused_requests" in res:
        logger.info(
            f"{res['num_paused_requests']} requests are interrupted "
            f"during updating weights for server {addr}"
        )


def update_weights_from_disk(
    experiment_name,
    trial_name,
    model_version,
    addresses,
    path,
    request_retries,
    request_timeout,
):
    async def _fn():
        # Wait for model checkpoints of meta.version
        update_name = names.update_weights_from_disk(
            experiment_name, trial_name, model_version
        )
        save_timestamp = float(name_resolve.wait(update_name, timeout=120))
        load_timestamp = datetime.now().timestamp()
        logger.info(
            f"Begin update weights from {path}, responded in {(load_timestamp - save_timestamp):.2f}s"
        )
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=request_timeout,
                sock_connect=request_timeout,
                connect=request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )
        jobs = [
            aupdate_weights_from_disk(
                session=session,
                addr=addr,
                path=path,
                request_retries=request_retries,
                request_timeout=request_timeout,
            )
            for addr in addresses
        ]
        await asyncio.gather(*jobs)
        await session.close()
        logger.info(
            f"Loading weights done in {(datetime.now().timestamp() - load_timestamp):.2f}s"
        )

    return uvloop.run(_fn())
