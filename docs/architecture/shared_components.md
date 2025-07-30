# AReaLite 与 Core 共享组件架构文档

## 概述

AReaLite和Core系统虽然在架构设计上有显著差异，但在某些底层实现和功能模块上存在共享。本文档详细分析两个系统之间的重叠部分，包括共享的代码模块、数据结构和工具函数。

## 共享组件总览

```mermaid
graph TB
    subgraph "共享组件架构"
        subgraph "1. 共享数据结构"
            DATA1[SequenceSample<br/>realhf/api/core/data_api.py]
            DATA2[MicroBatchSpec<br/>realhf/api/core/data_api.py]
            DATA3[ReaLModelConfig<br/>realhf/api/core/model_api.py]
            DATA4[GenerationHyperparameters<br/>realhf/api/cli_args.py]
        end
        
        subgraph "2. 共享模型组件"
            MODEL1[ReaLModel<br/>realhf/impl/model/nn/real_llm_api.py]
            MODEL2[Model Conversion<br/>realhf/impl/model/conversion/]
            MODEL3[HF Registry<br/>realhf/impl/model/conversion/hf_registry.py]
            MODEL4[Tokenizer Utils<br/>realhf/api/core/data_api.py]
        end
        
        subgraph "3. 共享工具函数"
            UTIL1[Stats Tracker<br/>realhf/base/stats_tracker.py]
            UTIL2[Logging<br/>realhf/base/logging.py]
            UTIL3[Constants<br/>realhf/base/constants.py]
            UTIL4[Name Resolve<br/>realhf/base/name_resolve.py]
        end
        
        subgraph "4. 共享算法实现"
            ALG1[PPO Functional<br/>realhf/impl/model/utils/ppo_functional.py]
            ALG2[Loss Functions<br/>realhf/impl/model/interface/*_interface.py]
            ALG3[Math Parser<br/>realhf/impl/dataset/math_parser.py]
            ALG4[Function Call<br/>functioncall/]
        end
        
        subgraph "5. AReaLite专有"
            LITE1[Engine API<br/>arealite/api/engine_api.py]
            LITE2[FSDP Utils<br/>arealite/utils/fsdp.py]
            LITE3[Launcher<br/>arealite/launcher/]
            LITE4[Workflow<br/>arealite/workflow/]
        end
        
        subgraph "6. Core专有"
            CORE1[System API<br/>realhf/api/core/system_api.py]
            CORE2[Master Worker<br/>realhf/system/master_worker.py]
            CORE3[DFG API<br/>realhf/api/core/dfg.py]
            CORE4[Model Worker<br/>realhf/system/model_worker.py]
        end
    end
    
    %% AReaLite使用共享组件
    LITE1 -.-> DATA1
    LITE1 -.-> DATA2
    LITE2 -.-> MODEL1
    LITE3 -.-> UTIL1
    LITE4 -.-> ALG1
    
    %% Core使用共享组件  
    CORE1 -.-> DATA1
    CORE1 -.-> DATA3
    CORE2 -.-> UTIL2
    CORE4 -.-> MODEL1
    
    %% 共享组件之间的依赖
    DATA1 --> UTIL3
    MODEL1 --> DATA1
    ALG1 --> DATA1
    
    style DATA1 fill:#e8f5e8
    style MODEL1 fill:#e8f5e8
    style UTIL1 fill:#e8f5e8
    style ALG1 fill:#e8f5e8
    style LITE1 fill:#e1f5fe
    style CORE1 fill:#fff3e0
```

## 详细共享组件分析

### 1. 共享数据结构

#### SequenceSample - 统一序列数据格式

```mermaid
classDiagram
    class SequenceSample {
        +data: Dict[str, torch.Tensor]
        +seqlens: Dict[str, List[List[int]]]
        +get_rl_samples()
        +get_sft_samples()
        +split_into_chunks()
        +concat_samples()
    }
    
    class AReaLiteUsage {
        +convert_to_tensordict()
        +batch_processing()
    }
    
    class CoreUsage {
        +mfc_data_transfer()
        +worker_communication()
    }
    
    SequenceSample --> AReaLiteUsage : 使用
    SequenceSample --> CoreUsage : 使用
```

**使用差异**：
- **AReaLite**: 主要用于Engine之间的数据传递，通常转换为TensorDict格式
- **Core**: 用于Worker之间的MFC调用，保持原始SequenceSample格式

#### MicroBatchSpec - 微批次规范

```python
# 共享的微批次规范定义
@dataclasses.dataclass
class MicroBatchSpec:
    mbs: int  # 微批次大小
    n_mbs: int  # 微批次数量
    
# AReaLite中的使用
def train_batch(self, input_: TensorDict, loss_fn, loss_weight_fn):
    mb_spec = MicroBatchSpec(mbs=self.config.micro_batch_size, n_mbs=...)
    return self._train_batch_impl(input_, mb_spec, loss_fn, loss_weight_fn)

# Core中的使用  
def process_mfc_request(self, request):
    mb_spec = request.mb_spec
    return self.interface.run_interface(request.data, mb_spec)
```

### 2. 共享模型组件

#### ReaLModel - 统一模型实现

```mermaid
sequenceDiagram
    participant A as AReaLite
    participant R as ReaLModel
    participant C as Core
    participant H as HF Registry
    
    Note over A,H: 共享模型组件使用流程
    
    A->>H: 加载HF模型配置
    H->>R: 创建ReaL模型实例
    R-->>A: 返回模型对象
    
    C->>H: 请求模型转换
    H->>R: 转换状态字典
    R-->>C: 返回转换后模型
    
    par AReaLite使用
        A->>R: FSDP包装
        R->>R: 前向/反向传播
        R-->>A: 训练结果
    and Core使用
        C->>R: Pipeline并行
        R->>R: 分布式计算
        R-->>C: 计算结果
    end
```

**使用差异分析**：

| 维度 | AReaLite | Core |
|------|----------|------|
| **包装方式** | FSDP2包装 | Pipeline + Tensor并行 |
| **初始化** | 直接在Engine中创建 | 通过ModelWorker管理 |
| **权重同步** | 手动上传到SGLang | 自动通过Worker系统 |
| **内存管理** | FSDP CPU卸载 | Pipeline分块加载 |

#### 模型转换工具

```mermaid
graph LR
    subgraph "HF Registry共享机制"
        HF[HuggingFace<br/>Model] --> CONV[Conversion<br/>Utils]
        CONV --> REAL[ReaLModel<br/>Config]
        
        REAL --> LITE[AReaLite<br/>FSDPEngine]
        REAL --> CORE[Core<br/>ModelWorker]
        
        CONV --> SAVE[State Dict<br/>Conversion]
        SAVE --> LITE
        SAVE --> CORE
    end
    
    style CONV fill:#e8f5e8
    style REAL fill:#e8f5e8
    style SAVE fill:#e8f5e8
```

### 3. 共享工具函数

#### Stats Tracker - 统计追踪系统

```python
# 共享的统计追踪接口
from realhf.base import stats_tracker

# AReaLite中的使用
def grpo_update(self, batch):
    with stats_tracker.time("grpo_update"):
        loss = self.compute_loss(batch)
        stats_tracker.register_scalar("grpo_loss", loss.item())
        return {"loss": loss}

# Core中的使用  
def train_step_interface(self, data, mb_spec):
    with stats_tracker.time("train_step"):
        result = self.model.forward(data)
        stats_tracker.register_scalar("train_loss", result.loss)
        return result
```

#### Logging系统

```mermaid
classDiagram
    class SharedLogging {
        +getLogger(name, category)
        +setup_logging()
        +ColoredFormatter
        +FileHandler
    }
    
    class AReaLiteLogger {
        +engine logs
        +workflow logs  
        +launcher logs
    }
    
    class CoreLogger {
        +worker logs
        +system logs
        +benchmark logs
    }
    
    SharedLogging --> AReaLiteLogger
    SharedLogging --> CoreLogger
```

### 4. 共享算法实现

#### GRPO算法核心函数

```python
# 共享的算法计算函数 (文件命名仍为ppo_functional但AReaLite实现GRPO)
from realhf.impl.model.utils import ppo_functional

# AReaLite中的使用 (实际是GRPO算法)
class FSDPPPOActor:
    def compute_advantages(self, batch):
        return ppo_functional.compute_gae(
            values=batch["values"],
            rewards=batch["rewards"], 
            gamma=self.config.gamma,
            lam=self.config.lam
        )
    
    def grpo_update(self, batch):
        # 注意：AReaLite实际使用GRPO算法，虽然调用相同的函数
        return ppo_functional.compute_ppo_loss(
            logprobs=batch["logprobs"],
            old_logprobs=batch["old_logprobs"],
            advantages=batch["advantages"],
            clip_ratio=self.config.clip_ratio
        )

# Core中的使用 (标准PPO算法)
class PPOInterface:
    def train_step_interface(self, data, mb_spec):
        advantages = ppo_functional.compute_gae(...)
        loss = ppo_functional.compute_ppo_loss(...)
        return SequenceSample(data={"loss": loss})
```

#### 数学验证函数

```mermaid
graph TB
    subgraph "共享的Function Call系统"
        FC1[Math Verify<br/>functioncall/math/verify.py]
        FC2[Code Verify<br/>functioncall/code/verify.py]
        FC3[Parser Utils<br/>realhf/impl/dataset/math_parser.py]
        
        subgraph "AReaLite使用"
            LITE_REWARD[Reward Function<br/>examples/arealite/gsm8k_grpo.py]
        end
        
        subgraph "Core使用"  
            CORE_REWARD[Reward Interface<br/>realhf/impl/model/interface/math_rw_interface.py]
        end
        
        FC1 --> LITE_REWARD
        FC2 --> LITE_REWARD
        FC3 --> LITE_REWARD
        
        FC1 --> CORE_REWARD
        FC2 --> CORE_REWARD
        FC3 --> CORE_REWARD
    end
    
    style FC1 fill:#e8f5e8
    style FC2 fill:#e8f5e8
    style FC3 fill:#e8f5e8
```

### 5. 架构差异对比

#### 数据流对比

```mermaid
graph TB
    subgraph "AReaLite数据流"
        subgraph "简化数据路径"
            A1[Raw Data] --> A2[TensorDict]
            A2 --> A3[Engine Processing]
            A3 --> A4[Direct Output]
        end
    end
    
    subgraph "Core数据流"  
        subgraph "复杂数据路径"
            C1[Raw Data] --> C2[SequenceSample]
            C2 --> C3[MFC Transport]
            C3 --> C4[Worker Processing]
            C4 --> C5[Data Redistribution] 
            C5 --> C6[Final Output]
        end
    end
    
    subgraph "共享组件使用"
        SHARED[ReaLModel<br/>Stats Tracker<br/>RL算法函数<br/>Math Verify]
        
        A3 -.-> SHARED
        C4 -.-> SHARED
    end
    
    style SHARED fill:#e8f5e8
```

#### 代码重用程度分析

```mermaid
pie title 代码重用程度分析
    "AReaLite专有代码" : 40
    "Core专有代码" : 35  
    "共享代码" : 25
```

**共享度分析**：
- **高度共享** (90%+): 基础模型实现、数学函数、工具类
- **部分共享** (50-90%): 数据结构定义、算法接口  
- **独立实现** (0-50%): 系统架构、启动器、工作流

### 6. 共享组件的设计模式

#### 适配器模式

```mermaid
classDiagram
    class SharedComponent {
        <<abstract>>
        +core_functionality()
    }
    
    class AReaLiteAdapter {
        +adapt_for_engines()
        +convert_to_tensordict()
    }
    
    class CoreAdapter {  
        +adapt_for_workers()
        +convert_to_sequence_sample()
    }
    
    SharedComponent <|-- AReaLiteAdapter
    SharedComponent <|-- CoreAdapter
    
    AReaLiteAdapter --> AReaLiteEngine
    CoreAdapter --> CoreWorker
```

#### 策略模式

```python
# 共享接口，不同实现策略
class ModelLoadStrategy:
    def load_model(self, config): pass

class AReaLiteFSDPStrategy(ModelLoadStrategy):
    def load_model(self, config):
        model = create_real_model(config)
        return apply_fsdp2(model)

class CorePipelineStrategy(ModelLoadStrategy):  
    def load_model(self, config):
        model = create_real_model(config)
        return setup_pipeline_parallel(model)
```

## 共享组件演进建议

### 1. 增强共享度
- 提取更多通用算法到shared模块
- 统一数据格式转换接口
- 共享更多工具函数

### 2. 减少耦合
- 通过接口而非具体实现依赖
- 使用依赖注入模式
- 增加配置驱动的组件选择

### 3. 向前兼容
- 保持共享接口的稳定性
- 提供版本兼容层
- 渐进式重构策略

这种共享设计体现了代码复用的价值，同时允许两个系统在架构层面保持各自的设计哲学和优化目标。
