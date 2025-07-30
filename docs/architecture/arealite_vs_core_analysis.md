# AReaLite vs Core: 软件架构分析

## 概述

AReaL项目包含两套不同的架构设计：

1. **AReaLite** (`/arealite/`, `/examples/arealite/`) - 轻量级、AI研究者友好的版本
2. **Core** (`/realhf/`, `/training/`) - 完整的分布式系统，支持大规模训练

本文档从软件架构角度深入分析这两套系统的主要区别，以同步PPO训练为例进行对比。

## 架构对比总览

| 维度 | AReaLite | Core |
|------|----------|------|
| **设计理念** | AI-centric，面向算法研究者 | System-centric，面向大规模生产 |
| **代码复杂度** | 简化，单文件可定制 | 复杂，多层抽象 |
| **学习曲线** | 平缓，PyTorch用户友好 | 陡峭，需要理解分布式系统概念 |
| **扩展性** | 适合中小规模实验 | 支持1000+ GPU大规模训练 |
| **灵活性** | 高度可定制，算法实验友好 | 结构化，适合生产部署 |

## 1. 入口点架构差异

### AReaLite 入口点 (`examples/arealite/gsm8k_grpo.py`)

```python
def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    
    # 直接创建组件，无需复杂的worker管理
    rollout = RemoteSGLangEngine(config.rollout)
    actor = FSDPPPOActor(config=config.actor)
    ref = FSDPPPOActor(config=config.ref) if config.ref else None
    
    # 简单的训练循环
    for global_step in range(max_steps):
        # 1. 数据收集
        batch = rollout.rollout_batch(data, workflow=workflow)
        
        # 2. 计算优势函数
        actor.compute_advantages(batch)
        
        # 3. PPO更新
        stats = actor.ppo_update(batch)
        
        # 4. 权重同步
        actor.upload_weights(weight_update_meta)
```

**特点：**
- SPMD (Single Program, Multiple Data) 模式
- 直接的PyTorch风格API
- 线性的训练流程，易于理解和调试
- 单文件包含完整训练逻辑

### Core 入口点 (`training/main_sync_ppo.py` → `realhf/experiments/`)

```python
@hydra.main(version_base=None, config_path="configs", config_name="sync-ppo")
def main(args):
    # 复杂的配置处理
    default_args = OmegaConf.structured(PPOMATHConfig)
    args = OmegaConf.merge(default_args, args)
    
    # 通过实验框架启动
    run_experiment(args)
```

```python
# 在 PPOMATHConfig 中定义复杂的工作流
class PPOMATHConfig(CommonExperimentConfig):
    def make_agents(self) -> List[ModelInterfaceAbstraction]:
        return [
            ModelInterfaceAbstraction(
                "actor-train", args=dict(...)
            ),
            ModelInterfaceAbstraction(
                "ref", args=dict(...)
            ),
        ]
    
    def make_dataset(self) -> DatasetAbstraction:
        return DatasetAbstraction(
            "prompt_answer_math", args=dict(...)
        )
```

**特点：**
- Worker-based分布式架构
- 多层抽象（Agent、Dataset、ModelInterface等）
- 声明式配置，运行时动态构建
- 复杂的依赖关系管理

## 2. 系统架构模式

### AReaLite: 组件化架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Entry Point   │───▶│  Engine Objects  │───▶│ Direct Methods  │
│                 │    │                  │    │                 │
│ gsm8k_grpo.py   │    │ RemoteSGLang     │    │ rollout_batch() │
│                 │    │ FSDPPPOActor    │    │ ppo_update()    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

**核心组件：**

1. **Engine API** (`arealite/api/engine_api.py`)
   ```python
   class TrainEngine(abc.ABC):
       def train_batch(self, input_, loss_fn, loss_weight_fn): ...
       def forward(self, input_, output_seqlens=None): ...
   
   class InferenceEngine(abc.ABC):
       async def agenerate(self, req: LLMRequest): ...
       def update_weights(self, meta: WeightUpdateMeta): ...
   ```

2. **Workflow** (`arealite/workflow/rlvr.py`)
   ```python
   class RLVRWorkflow(RolloutWorkflow):
       async def arun_episode(self, engine, data):
           # 简单的异步生成逻辑
           req = LLMRequest(input_ids=input_ids, gconfig=self.gconfig)
           resps = await asyncio.gather(*[
               engine.agenerate(req) for _ in range(n_samples)
           ])
           return process_responses(resps)
   ```

### Core: 分布式Worker架构

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│    Master    │───▶│     Workers     │───▶│  Model Functions │
│   Scheduler  │    │                 │    │                  │
│              │    │ RolloutWorker   │    │ generate_MFC     │
│ ExperimentRun│    │ ModelWorker     │    │ train_step_MFC   │
│              │    │ ControllerWorker│    │ compute_ref_MFC  │
└──────────────┘    └─────────────────┘    └──────────────────┘
```

**核心组件：**

1. **Master-Worker 系统** (`realhf/system/master_worker.py`)
   ```python
   class MasterWorker:
       def _schedule_model_function_calls(self, mfcs: List[MFCDef]):
           # 复杂的调度逻辑
           for mfc in mfcs:
               self._send_mfc_to_worker(mfc)
   ```

2. **模型函数调用 (MFC)** (`realhf/system/model_function_call.py`)
   ```python
   @register_mfc("ppo-update")
   def ppo_update_mfc(batch, kl_ctl, eps_clip, ...):
       # 分布式PPO更新逻辑
       return perform_ppo_update(batch, ...)
   ```

3. **数据流图 (DFG)** (`realhf/api/core/dfg.py`)
   ```python
   class DataFlowGraph:
       def add_mfc_dataflow(self, src_mfc, dst_mfc, spec):
           # 复杂的数据依赖管理
   ```

## 3. 同步训练实现对比

### AReaLite 同步训练

```python
# 直接在主循环中实现同步逻辑
for global_step in range(max_steps):
    # 同步数据收集
    if config.async_training:
        batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
    else:
        data = next(data_generator)
        batch = rollout.rollout_batch(data, workflow=workflow)
    
    # 同步屏障确保所有进程完成
    dist.barrier(device_ids=[actor.device.index])
    torch.cuda.synchronize()
    
    # 统一的PPO更新
    stats = actor.ppo_update(batch)
```

**特点：**
- 显式的同步控制
- 清晰的步骤划分
- 容易理解和修改

### Core 同步训练

```python
# 通过复杂的MFC调度实现同步
class PPOMATHConfig:
    def make_dfg(self) -> DataFlowGraph:
        dfg = DataFlowGraph()
        
        # 定义复杂的数据流依赖
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

**特点：**
- 声明式的依赖定义
- 系统自动处理同步
- 更好的资源利用，但复杂度高

## 4. 代码复杂度分析

### 代码行数对比

```bash
# AReaLite 核心文件
examples/arealite/gsm8k_grpo.py: ~233 lines
arealite/engine/ppo/actor.py: ~200-300 lines
arealite/workflow/rlvr.py: ~100-150 lines

# Core 核心文件  
training/main_sync_ppo.py: ~70 lines (但大量逻辑在配置类中)
realhf/experiments/common/ppo_math_exp.py: ~300+ lines
realhf/system/master_worker.py: ~500+ lines
realhf/system/model_worker.py: ~400+ lines
```

### 抽象层次对比

**AReaLite (3层抽象):**
```
用户脚本 → Engine对象 → 后端实现
gsm8k_grpo.py → FSDPPPOActor → PyTorch FSDP
```

**Core (5-6层抽象):**
```
用户配置 → 实验框架 → Master调度 → Worker执行 → MFC调用 → 后端实现
sync-ppo.yaml → PPOMATHConfig → MasterWorker → ModelWorker → ppo_update_MFC → PyTorch
```

## 5. 异步训练支持

### AReaLite 异步训练

```python
# 内置的异步支持
if config.async_training:
    # 异步准备批次，无需等待
    batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
else:
    # 同步模式
    batch = rollout.rollout_batch(data, workflow=workflow)

# 简单的异步控制参数
max_concurrent_rollouts=16
max_head_offpolicyness=4
```

### Core 异步训练

```python
# 独立的异步实验框架
class AsyncPPOMATHConfig(AsyncRLExperimentConfig, PPOMATHConfig):
    @property
    def agent(self) -> AgentAbstraction:
        return AgentAbstraction("math-single-step", args=dict(...))
    
    @property  
    def env(self) -> EnvServiceAbstraction:
        return EnvServiceAbstraction("math-code-single-step", args=dict(...))
```

## 6. 自定义和扩展性

### AReaLite: 单文件自定义

```python
# 在同一个文件中自定义所有组件
def custom_reward_fn(prompt, completions, **kwargs):
    # 自定义奖励函数
    return calculate_reward(completions)

class CustomWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        # 自定义rollout逻辑
        return custom_generation_logic()

def main(args):
    # 直接使用自定义组件
    workflow = CustomWorkflow(reward_fn=custom_reward_fn)
    # ... 训练逻辑
```

### Core: 模块化注册系统

```python
# 需要在多个文件中注册组件
@register_agent("custom-agent")
class CustomAgent:
    def step(self, ...): ...

@register_env("custom-env") 
class CustomEnv:
    def reset(self, ...): ...

@register_mfc("custom-mfc")
def custom_mfc(...):
    return ...

# 在配置中声明使用
class CustomConfig(PPOMATHConfig):
    @property
    def agent(self):
        return AgentAbstraction("custom-agent", args=dict(...))
```

## 7. 性能和扩展性

### AReaLite
- **适用场景**: 单节点到中等规模(8-64 GPU)实验
- **优势**: 启动快，调试容易，迭代效率高
- **限制**: 大规模扩展时可能遇到瓶颈

### Core  
- **适用场景**: 大规模生产训练(100-1000+ GPU)
- **优势**: 高效的资源利用，完善的容错机制
- **限制**: 复杂的设置和调试过程

## 8. 总结与建议

### 何时选择 AReaLite
- 算法研究和快速原型开发
- 需要频繁修改训练逻辑
- 团队成员主要是AI研究者而非系统工程师
- 资源规模在64 GPU以内

### 何时选择 Core
- 大规模生产训练
- 需要最高的训练效率
- 有专门的系统工程师团队
- 对容错和监控有高要求

### 架构演进方向

根据AReaLite设计文档，未来的发展方向是：

1. **AReaLite作为Core的API基础**: Core将重构为基于AReaLite的API设计
2. **统一的接口**: 用户可以从AReaLite无缝迁移到Core进行大规模训练
3. **渐进式复杂度**: 从简单的AReaLite开始，根据需要逐步使用更高级的Core功能

这种设计理念体现了"90%功能，10%复杂度"的哲学，让AI研究者能够专注于算法创新而不是系统细节。

## 附录：关键文件对比

### 入口点文件
- **AReaLite**: `examples/arealite/gsm8k_grpo.py` (完整的训练逻辑)
- **Core**: `examples/run_sync_ppo.sh` → `training/main_sync_ppo.py` (配置加载) → `realhf/experiments/common/ppo_math_exp.py` (实际逻辑)

### 架构文档
- **AReaLite**: `arealite/README.md` (设计原理和使用方法)
- **Core**: 分散在多个文档中，主要在 `realhf/` 目录

### 核心抽象
- **AReaLite**: `arealite/api/engine_api.py` (简洁的Engine接口)
- **Core**: `realhf/api/core/` (复杂的分布式抽象)