# AReaLite 组件架构

本文档提供AReaLite核心组件及其交互的详细架构视图，旨在为开发者提供对系统内部结构的清晰理解。

## 组件交互架构

```mermaid
graph TB
    %% 应用层
    subgraph "应用层"
        APP[训练脚本<br/>gsm8k_grpo.py]
        CFG[配置<br/>GRPOConfig]
    end
    
    %% API抽象层
    subgraph "API抽象层" 
        ENGINE_API[引擎API<br/>InferenceEngine/TrainEngine]
        WORKFLOW_API[工作流API<br/>RolloutWorkflow]
        CLI_API[CLI参数<br/>配置类]
        IO_API[IO结构<br/>数据类型]
    end
    
    %% 算法实现层
    subgraph "算法实现层"
        GRPO[GRPO Actor<br/>FSDPPPOActor]
        SFT[SFT引擎<br/>FSDPLMEngine]  
        RLVR[RLVR工作流<br/>RLVRWorkflow]
        VISION[视觉RLVR<br/>VisionRLVRWorkflow]
    end
    
    %% 引擎后端层
    subgraph "引擎后端层"
        FSDP_ENG[FSDP引擎<br/>分布式训练]
        BASE_ENG[基础HF引擎<br/>模型管理]
        REMOTE_ENG[远程SGLang引擎<br/>推理编排]
    end
    
    %% 基础设施层
    subgraph "基础设施层"
        SGLANG_CLUSTER[SGLang服务器集群]
        FSDP_BACKEND[FSDP2后端]
        DATA_PIPELINE[数据管道]
    end
    
    %% 外部依赖  
    subgraph "外部依赖"
        PYTORCH[PyTorch + FSDP]
        SGLANG_LIB[SGLang库]
        HF[HuggingFace]
        TORCHDATA[TorchData]
    end
    
    %% Application to API
    APP --> ENGINE_API
    APP --> WORKFLOW_API
    CFG --> CLI_API
    
    %% API to Algorithm
    ENGINE_API --> GRPO
    ENGINE_API --> SFT
    WORKFLOW_API --> RLVR
    WORKFLOW_API --> VISION
    CLI_API --> GRPO
    CLI_API --> SFT
    
    %% Algorithm to Engine Backend
    GRPO --> FSDP_ENG
    SFT --> FSDP_ENG
    RLVR --> REMOTE_ENG
    VISION --> REMOTE_ENG
    
    %% Engine Backend to Infrastructure
    FSDP_ENG --> BASE_ENG
    BASE_ENG --> FSDP_BACKEND
    REMOTE_ENG --> SGLANG_CLUSTER
    GRPO --> DATA_PIPELINE
    SFT --> DATA_PIPELINE
    
    %% Infrastructure to External
    FSDP_BACKEND --> PYTORCH
    SGLANG_CLUSTER --> SGLANG_LIB
    BASE_ENG --> HF
    DATA_PIPELINE --> TORCHDATA
    
    %% Cross-layer Dependencies (dotted lines)
    FSDP_ENG -.-> IO_API
    RLVR -.-> IO_API
    
    %% Styling
    classDef app fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef api fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef algo fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef engine fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef infra fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef external fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    
    class APP,CFG app
    class ENGINE_API,WORKFLOW_API,CLI_API,IO_API api
    class GRPO,SFT,RLVR,VISION algo
    class FSDP_ENG,BASE_ENG,REMOTE_ENG engine
    class SGLANG_CLUSTER,FSDP_BACKEND,DATA_PIPELINE infra
    class PYTORCH,SGLANG_LIB,HF,TORCHDATA external
```

## 核心组件详情

### 1. FSDPPPOActor (GRPO实现)

```mermaid
classDiagram
    class FSDPPPOActor {
        +config: PPOActorConfig
        +engine: TrainEngine
        +reward_bias: float
        +reward_scaling: float
        +group_size: int
        +kl_ctl: float
        
        +compute_logp(data: TensorDict) torch.Tensor
        +compute_advantages(data: TensorDict) void
        +grpo_update(data: TensorDict) List[Dict]
        +upload_weights(meta: WeightUpdateMeta) void
        +set_version(version: int) void
        
        -_compute_rewards() torch.Tensor
        -_compute_gae() torch.Tensor
        -_apply_group_normalization() void
    }
    
    class GRPOConfig {
        +eps_clip: float
        +c_clip: Optional[float]
        +group_size: int
        +group_reward_norm: bool  
        +group_adv_norm: bool
        +reward_bias: float
        +reward_scaling: float
        +kl_ctl: float
        +adv_norm: bool
        +discount: float
        +gae_lambda: float
    }
    
    class TrainEngine {
        <<interface>>
        +train_batch(input_, loss_fn, loss_weight_fn)*
        +forward(input_, output_seqlens)*
        +upload_weights(meta)*
        +save_checkpoint()*
        +load_checkpoint()*
    }
    
    FSDPPPOActor --> GRPOConfig : 使用
    FSDPPPOActor --> TrainEngine : 实现
    
    note for FSDPPPOActor : "实现GRPO算法\n基于分组的优化"
```

### 2. RemoteSGLangEngine (推理编排)

```mermaid
classDiagram
    class RemoteSGLangEngine {
        +server_urls: List[str]
        +session: aiohttp.ClientSession
        +version: int
        
        +agenerate(req: LLMRequest) LLMResponse
        +rollout_batch(data, workflow) TensorDict
        +update_weights(meta: WeightUpdateMeta) Future
        +set_version(version: int) void
        +pause() void
        +resume() void
        
        -_load_balance_request() str
        -_aggregate_responses() List[LLMResponse]
        -_handle_server_error() void
    }
    
    class LLMRequest {
        +text: str
        +generation_config: dict
        +request_id: str
        +stream: bool
    }
    
    class LLMResponse {
        +text: str
        +logprobs: List[float]
        +finish_reason: str
        +usage: dict
    }
    
    class WeightUpdateMeta {
        +version: int
        +model_name: str
        +weights_path: str
        +metadata: dict
    }
    
    RemoteSGLangEngine --> LLMRequest : processes
    RemoteSGLangEngine --> LLMResponse : returns
    RemoteSGLangEngine --> WeightUpdateMeta : handles
    
    note for RemoteSGLangEngine : "Orchestrates multiple\nSGLang server instances"
```

### 3. RLVRWorkflow (Rollout-Learning-Verify-Reward)

```mermaid
classDiagram
    class RLVRWorkflow {
        +reward_fn: Callable
        +n_generations: int
        +temperature: float
        +max_tokens: int
        
        +arun_episode(engine, data) TensorDict
        
        -_generate_responses(engine, prompts) List[str]
        -_compute_rewards(responses, references) torch.Tensor
        -_format_output() TensorDict
    }
    
    class VisionRLVRWorkflow {
        +processor: AutoProcessor
        +image_token: str
        
        +arun_episode(engine, data) TensorDict
        
        -_process_images(data) torch.Tensor
        -_format_vision_prompt() str
    }
    
    class MathRLVRWorkflow {
        +math_verifier: MathVerifier
        +use_tool: bool
        
        +arun_episode(engine, data) TensorDict
        
        -_verify_math_solution() bool
        -_compute_math_reward() float
    }
    
    RLVRWorkflow <|-- VisionRLVRWorkflow
    RLVRWorkflow <|-- MathRLVRWorkflow
    
    note for RLVRWorkflow : "Base workflow for RLHF\nwith customizable rewards"
```

## 数据流架构

```mermaid
flowchart TD
    %% 数据源
    subgraph "数据源"
        HF_DATASET[HuggingFace数据集<br/>GSM8K/CLEVR等]
        USER_DATA[自定义数据集<br/>JSON/CSV等]
    end
    
    %% 数据处理
    subgraph "数据处理管道"
        LOADER[StatefulDataLoader<br/>批处理和分发]
        TOKENIZER[分词器<br/>文本 → Token ID]
        COLLATOR[数据整理器<br/>填充和掩码]
    end
    
    %% Rollout阶段
    subgraph "Rollout阶段"
        PROMPTS[提示批次<br/>输入序列]
        INFERENCE[SGLang推理<br/>文本生成]
        RESPONSES[生成回复<br/>每个提示多个回复]
        REWARDS[奖励计算<br/>任务特定评分]
    end
    
    %% 训练阶段
    subgraph "训练阶段"
        LOGP[对数概率<br/>计算]
        ADVANTAGES[优势估计<br/>GAE算法]
        GRPO_LOSS[GRPO损失<br/>基于分组的截断]
        GRADIENTS[梯度计算<br/>反向传播]
        UPDATES[参数更新<br/>FSDP同步]
    end
    
    %% 权重同步
    subgraph "权重同步"
        PAUSE[暂停推理<br/>服务器协调]
        TRANSFER[权重传输<br/>Actor → 服务器]
        RESUME[恢复推理<br/>热切换完成]
    end
    
    %% 数据流连接
    HF_DATASET --> LOADER
    USER_DATA --> LOADER
    
    LOADER --> TOKENIZER
    TOKENIZER --> COLLATOR
    COLLATOR --> PROMPTS
    
    PROMPTS --> INFERENCE
    INFERENCE --> RESPONSES  
    RESPONSES --> REWARDS
    
    REWARDS --> LOGP
    LOGP --> ADVANTAGES
    ADVANTAGES --> GRPO_LOSS
    GRPO_LOSS --> GRADIENTS
    GRADIENTS --> UPDATES
    
    UPDATES --> PAUSE
    PAUSE --> TRANSFER
    TRANSFER --> RESUME
    
    %% Cycle Back
    RESUME -.->|Next Iteration| PROMPTS
    
    %% Styling
    classDef dataSource fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px  
    classDef rollout fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef training fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef sync fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class HF_DATASET,USER_DATA dataSource
    class LOADER,TOKENIZER,COLLATOR processing
    class PROMPTS,INFERENCE,RESPONSES,REWARDS rollout
    class LOGP,ADVANTAGES,GRPO_LOSS,GRADIENTS,UPDATES training
    class PAUSE,TRANSFER,RESUME sync
```

## Memory and Communication Patterns

### FSDP2 Sharding Strategy

```mermaid
graph TB
    subgraph "GPU 0"
        P0[Parameters Shard 0<br/>Layers 0-7] 
        G0[Gradients Shard 0]
        O0[Optimizer State 0]
    end
    
    subgraph "GPU 1" 
        P1[Parameters Shard 1<br/>Layers 8-15]
        G1[Gradients Shard 1] 
        O1[Optimizer State 1]
    end
    
    subgraph "GPU 2"
        P2[Parameters Shard 2<br/>Layers 16-23]
        G2[Gradients Shard 2]
        O2[Optimizer State 2] 
    end
    
    subgraph "GPU 3"
        P3[Parameters Shard 3<br/>Layers 24-31]
        G3[Gradients Shard 3]
        O3[Optimizer State 3]
    end
    
    %% All-Gather for Forward Pass
    P0 <-.->|All-Gather| P1
    P1 <-.->|All-Gather| P2  
    P2 <-.->|All-Gather| P3
    P3 <-.->|All-Gather| P0
    
    %% Reduce-Scatter for Backward Pass
    G0 <-.->|Reduce-Scatter| G1
    G1 <-.->|Reduce-Scatter| G2
    G2 <-.->|Reduce-Scatter| G3  
    G3 <-.->|Reduce-Scatter| G0
    
    classDef gpu fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef param fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef grad fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef opt fill:#fce4ec,stroke:#ad1457,stroke-width:2px
    
    class P0,P1,P2,P3 param
    class G0,G1,G2,G3 grad  
    class O0,O1,O2,O3 opt
```

### SGLang Server Load Balancing

```mermaid
graph LR
    subgraph "RemoteSGLangEngine"
        LB[Load Balancer<br/>Round Robin]
        QUEUE[Request Queue<br/>Async Processing]
    end
    
    subgraph "SGLang Server Pool"
        S1[Server 1<br/>GPU 0-1<br/>TP=2]
        S2[Server 2<br/>GPU 2-3<br/>TP=2] 
        S3[Server 3<br/>GPU 4-5<br/>TP=2]
        S4[Server N<br/>GPU N-M<br/>TP=2]
    end
    
    QUEUE --> LB
    LB --> S1
    LB --> S2  
    LB --> S3
    LB --> S4
    
    S1 -.->|Health Check| LB
    S2 -.->|Health Check| LB
    S3 -.->|Health Check| LB  
    S4 -.->|Health Check| LB
    
    classDef engine fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef server fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class LB,QUEUE engine
    class S1,S2,S3,S4 server
```

## Configuration Architecture

```mermaid
classDiagram
    class GRPOConfig {
        +actor: PPOActorConfig
        +ref: PPOActorConfig
        +rollout: RolloutConfig  
        +train_dataset: DatasetConfig
        +eval_dataset: DatasetConfig
        +sglang: SGLangConfig
        +allocation_mode: str
        +async_training: bool
    }
    
    class PPOActorConfig {
        +path: str
        +eps_clip: float
        +c_clip: Optional[float]
        +group_size: int
        +group_reward_norm: bool
        +kl_ctl: float
        +lr: float
        +weight_decay: float
        +max_grad_norm: float
    }
    
    class RolloutConfig {
        +n_generations: int
        +temperature: float
        +max_tokens: int
        +batch_size: int
        +reward_fn: str
    }
    
    class SGLangConfig {
        +tp_size: int
        +attention_backend: str
        +trust_remote_code: bool
        +max_total_tokens: int
    }
    
    class DatasetConfig {
        +name: str
        +split: str
        +batch_size: int
        +shuffle: bool
        +num_workers: int
    }
    
    GRPOConfig --> PPOActorConfig : contains
    GRPOConfig --> RolloutConfig : contains  
    GRPOConfig --> SGLangConfig : contains
    GRPOConfig --> DatasetConfig : contains
    
    note for GRPOConfig : "Unified configuration\nfor entire training pipeline"
```

This component architecture provides developers with a clear understanding of how AReaLite's various parts interact, enabling easier debugging, extension, and customization of the system.
