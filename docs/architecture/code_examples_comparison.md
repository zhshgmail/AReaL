# 具体代码实例对比

本文档通过具体的代码片段对比，展示AReaLite和Core在实现上的主要差异。

## 1. 入口点代码对比

### AReaLite 入口点 (examples/arealite/gsm8k_grpo.py)

```python
def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # 直接创建数据集
    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        split="train",
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )

    # 直接创建引擎
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)
    
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)

    # 简单的训练循环
    for global_step in range(max_steps):
        # 数据收集
        if config.async_training:
            batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
        else:
            data = next(data_generator)
            batch = rollout.rollout_batch(data, workflow=workflow)

        batch = batch.to(actor.device)
        dist.barrier(device_ids=[actor.device.index])

        # PPO更新
        stats = actor.ppo_update(batch)
        
        # 权重更新
        rollout.pause()
        if dist.get_rank() == 0:
            future = rollout.update_weights(weight_update_meta)
        actor.upload_weights(weight_update_meta)
        rollout.resume()
```

### Core 入口点 (training/main_sync_ppo.py)

```python
@hydra.main(version_base=None, config_path="configs", config_name="sync-ppo")
def main(args):
    import realhf.base.logging as logging
    logger = logging.getLogger("quickstart", "colored")

    # 复杂的配置合并
    default_args = OmegaConf.structured(PPOMATHConfig)
    args = OmegaConf.merge(default_args, args)
    args: PPOMATHConfig = OmegaConf.to_object(args)

    # 设置实验名称
    exp_name = args.experiment_name
    if args.trial_name == MISSING:
        args.trial_name = trial_name = (
            f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )

    # 通过实验框架运行
    run_experiment(args)
```

## 2. 配置和组件定义对比

### AReaLite 配置 (简单的dataclass)

```python
@dataclasses.dataclass
class GRPOConfig:
    # 直接的配置字段
    tokenizer_path: str = "Qwen/Qwen3-1.7B"
    train_dataset: DatasetConfig = DatasetConfig()
    rollout: SGLangEngineConfig = SGLangEngineConfig()
    actor: PPOActorConfig = PPOActorConfig()
    async_training: bool = False
    total_train_epochs: int = 1
```

### Core 配置 (复杂的继承和属性方法)

```python
@dataclasses.dataclass
class PPOMATHConfig(CommonExperimentConfig, PPOMATHExperimentOptions):
    
    @property
    def ppo_kwargs(self):
        return dict(
            n_minibatches=self.ppo.ppo_n_minibatches,
            kl_ctl=self.ppo.kl_ctl,
            discount=self.ppo.discount,
            gae_lambda=self.ppo.gae_lambda,
            eps_clip=self.ppo.eps_clip,
            # ... 更多参数
        )

    def make_agents(self) -> List[ModelInterfaceAbstraction]:
        return [
            ModelInterfaceAbstraction(
                "actor-train",
                args=dict(
                    type_=self.actor.type,
                    path=self.actor.path,
                    # ... 复杂的参数配置
                ),
            ),
            ModelInterfaceAbstraction("ref", args=dict(...)),
        ]

    def make_dataset(self) -> DatasetAbstraction:
        return DatasetAbstraction(
            "prompt_answer_math",
            args=dict(
                dataset_path=self.dataset.path,
                max_length=self.dataset.max_length,
                # ... 更多配置
            ),
        )

    def make_dfg(self) -> DataFlowGraph:
        dfg = DataFlowGraph()
        # 复杂的数据流定义
        dfg.add_mfc_dataflow(
            self.prompt_mfc, self.gen_mfc,
            {"prompt", "prompt_mask"}
        )
        dfg.add_mfc_dataflow(
            self.gen_mfc, self.ppo_mfc,
            {"packed_seq", "response", "logprobs", "rewards"}
        )
        return dfg
```

## 3. 训练逻辑实现对比

### AReaLite PPO实现 (arealite/engine/ppo/actor.py)

```python
class PPOActor:
    def ppo_update(self, data: TensorDict) -> List[Dict[str, float]]:
        """直接的PPO更新实现"""
        self.engine.train()
        
        # 计算损失的内部函数
        def ppo_loss_fn(logits, input_data):
            logprobs = gather_logprobs(logits, input_data["labels"])
            
            # PPO clip损失
            ratio = torch.exp(logprobs - input_data["old_logprobs"])
            surr1 = ratio * input_data["advantages"]
            surr2 = torch.clamp(ratio, 1.0 - self.config.eps_clip, 
                               1.0 + self.config.eps_clip) * input_data["advantages"]
            policy_loss = -torch.min(surr1, surr2).mean()
            
            return policy_loss

        # 直接调用引擎更新
        stats = self.engine.train_batch(
            input_=data,
            loss_fn=ppo_loss_fn,
            loss_weight_fn=lambda x: 1.0,
        )
        
        return stats
```

### Core PPO实现 (通过MFC系统)

```python
# 在realhf/impl/model/nn/ppo.py中定义MFC
@register_mfc("ppo_update")
def ppo_update_mfc(
    batch: Sequence,
    n_minibatches: int = 4,
    kl_ctl: float = 0.1,
    eps_clip: float = 0.2,
    # ... 更多参数
):
    """通过MFC系统执行PPO更新"""
    model = mfc_model()
    
    # 复杂的minibatch处理
    for minibatch in split_minibatches(batch, n_minibatches):
        # 计算损失
        policy_loss, value_loss, kl_div = compute_ppo_loss(
            model, minibatch, eps_clip, kl_ctl
        )
        
        total_loss = policy_loss + value_loss
        total_loss.backward()
    
    # 参数更新
    optimizer = mfc_optimizer()
    optimizer.step()
    optimizer.zero_grad()
    
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "kl_divergence": kl_div.item(),
    }

# 在配置中调用MFC
class PPOMATHConfig:
    @property
    def ppo_mfc(self):
        return MFCDef(
            model_name="actor-train",
            n_seqs=self.rollout.n_seqs,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            model_function_name="ppo_update",
            func_kwargs=self.ppo_kwargs,
        )
```

## 4. 数据流处理对比

### AReaLite 数据处理 (直接的方法调用)

```python
class RLVRWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine: InferenceEngine, data):
        # 直接的tokenization
        input_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
        )

        # 直接的生成请求
        req = LLMRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
        )

        # 并发生成多个响应
        n_samples = self.gconfig.n_samples
        resps = await asyncio.gather(*[
            engine.agenerate(req) for _ in range(n_samples)
        ])

        # 直接计算奖励
        results = []
        for resp in resps:
            reward = self.reward_fn(
                prompt=prompt_str,
                completions=completions_str,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )
            results.append(TensorDict(res, batch_size=[1]))

        return concat_padded_tensors(results)
```

### Core 数据处理 (通过Agent和Environment抽象)

```python
# Agent抽象定义
@register_agent("math-single-step")
class MathSingleStepAgent:
    def __init__(self, gconfig, tokenizer_path, **kwargs):
        self.gconfig = gconfig
        self.tokenizer = load_hf_tokenizer(tokenizer_path)

    def step(self, env_output):
        # 复杂的状态处理
        packed_prompts = pack_prompts(env_output["prompt"])
        
        # 通过MFC系统生成
        generation_mfc = MFCDef(
            model_name="actor-gen",
            interface_type=ModelInterfaceType.GENERATE,
            model_function_name="generate",
            func_kwargs=dict(
                gconfig=self.gconfig,
                tokenizer=self.tokenizer,
            )
        )
        
        return {
            "packed_prompts": packed_prompts,
            "generation_mfc": generation_mfc,
        }

# Environment抽象定义
@register_env("math-code-single-step")
class MathCodeSingleStepEnv:
    def __init__(self, dataset_path, **kwargs):
        self.dataset = load_dataset(dataset_path)

    def reset(self, indices):
        # 复杂的环境重置逻辑
        data_batch = [self.dataset[i] for i in indices]
        return {
            "prompt": [item["prompt"] for item in data_batch],
            "answer": [item["answer"] for item in data_batch],
        }

    def step(self, agent_output):
        # 复杂的奖励计算逻辑
        rewards = []
        for completion, answer in zip(agent_output["completions"], self.answers):
            reward = compute_math_reward(completion, answer)
            rewards.append(reward)
        
        return {"rewards": rewards, "done": True}
```

## 5. 启动脚本对比

### AReaLite 启动 (简单的python调用)

```bash
# examples/arealite/configs/gsm8k_grpo.yaml
tokenizer_path: Qwen/Qwen3-1.7B
train_dataset:
  path: hf-dataset://inclusionAI/AReaL-RL-Data/data/train.jsonl
  batch_size: 32
rollout:
  server_address: http://localhost:30000
  timeout: 60
actor:
  lr: 1e-5
  eps_clip: 0.2

# 启动命令
python3 -m arealite.launcher.ray examples/arealite/gsm8k_grpo.py \
    --config examples/arealite/configs/gsm8k_grpo.yaml \
    experiment_name=my_experiment \
    trial_name=my_trial
```

### Core 启动 (复杂的hydra配置)

```yaml
# training/configs/sync-ppo.yaml
defaults:
  - _self_
  - cluster: slurm-gpu
  - actor: llama2-7b
  - ref: llama2-7b
  - dataset: math-ppo
  - ppo: default

experiment_name: ppo-math
trial_name: ???
mode: ray

# 数百行的详细配置...
allocation_mode: 
  actor_train: 
    placement_group:
      strategy: STRICT_PACK
      bundles:
        - GPU: 8
  actor_gen:
    placement_group:
      strategy: SPREAD
      bundles:
        - GPU: 2
```

```bash
# 启动命令
python3 training/main_sync_ppo.py \
    experiment_name=my_experiment \
    trial_name=my_trial \
    actor.path=Qwen/Qwen3-1.7B \
    dataset.path=path/to/dataset \
    cluster.n_nodes=4
```

## 6. 自定义开发对比

### AReaLite 自定义 (单文件修改)

```python
# 在同一个入口文件中自定义所有组件
def custom_reward_fn(prompt, completions, **kwargs):
    """自定义奖励函数"""
    # 业务逻辑
    if "正确答案" in completions:
        return 1.0
    else:
        return 0.0

class CustomWorkflow(RLVRWorkflow):
    """自定义工作流"""
    async def arun_episode(self, engine, data):
        # 自定义生成逻辑
        # 可以实现多轮对话、工具调用等
        return custom_results

class CustomPPOActor(FSDPPPOActor):
    """自定义PPO算法"""
    def compute_advantages(self, data):
        # 自定义优势函数计算
        # 例如实现GRPO算法
        pass

def main(args):
    # 直接使用自定义组件
    workflow = CustomWorkflow(reward_fn=custom_reward_fn)
    actor = CustomPPOActor(config=config.actor)
    # ... 其余逻辑保持不变
```

### Core 自定义 (多文件注册系统)

```python
# 1. 在 realhf/impl/agent/ 中定义新的Agent
@register_agent("custom-math-agent")
class CustomMathAgent:
    def step(self, env_output):
        # 自定义逻辑
        pass

# 2. 在 realhf/impl/env/ 中定义新的Environment  
@register_env("custom-math-env")
class CustomMathEnv:
    def reset(self, indices):
        # 自定义逻辑
        pass

# 3. 在 realhf/impl/model/nn/ 中定义新的MFC
@register_mfc("custom-ppo-update")
def custom_ppo_update_mfc(batch, **kwargs):
    # 自定义PPO逻辑
    pass

# 4. 在配置类中组合使用
class CustomPPOMATHConfig(PPOMATHConfig):
    @property
    def agent(self):
        return AgentAbstraction("custom-math-agent", args=dict(...))
    
    @property
    def env(self):
        return EnvServiceAbstraction("custom-math-env", args=dict(...))
    
    @property
    def ppo_mfc(self):
        return MFCDef(
            model_function_name="custom-ppo-update",
            func_kwargs=dict(...)
        )
```

## 总结

从具体代码实例可以看出：

1. **AReaLite**: 代码直观，逻辑清晰，容易调试和修改，适合快速原型开发
2. **Core**: 系统复杂，抽象层次多，但提供了更强的扩展性和资源管理能力

两套系统各有优势，选择哪个取决于具体的使用场景和团队能力。