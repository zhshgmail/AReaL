from tensordict import TensorDict
from arealite.api.cli_args import GenerationHyperparameters
from arealite.api.workflow_api import RolloutWorkflow
from arealite.api.io_struct import LLMRequest
import uuid
import torch
from transformers import PreTrainedTokenizerFast


class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer

    async def arun_episode(self, engine, data):
        text = self.tokenizer.apply_chat_template(
            data["messages"], tokenize=False, add_generation_prompt=True
        )
        req = LLMRequest(
            rid=uuid.uuid4().hex,
            text=text,
            gconfig=self.gconfig,
        )
        resp = await engine.agenerate(req)

        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0] * resp.input_len + resp.output_logprobs
        prompt_mask = [1] * resp.input_len + [0] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        reward = self.reward_fn(
            prompt=req.text,
            completions=resp.completions,
            prompt_ids=resp.input_tokens,
            completion_ids=resp.output_tokens,
            **data,
        )
        res = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            # reward
            rewards=torch.tensor([reward]),
        )

        return TensorDict(res, batch_size=[1])
