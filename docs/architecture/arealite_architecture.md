# AReaLite 架构文档

## 高层功能层次视图

### 系统层次结构

```mermaid
graph TB
    subgraph "AReaLite 系统层次架构"
        subgraph "1. 应用层 (Entry Points)"
            APP1[gsm8k_grpo.py]
            APP2[gsm8k_sft.py] 
            APP3[clevr_count_70k_grpo.py]
            APP4[clevr_count_70k_sft.py]
        end
        
        subgraph "2. API 抽象层"
            API1[Engine API<br/>engine_api.py]
            API2[Workflow API<br/>workflow_api.py]
            API3[CLI Args<br/>cli_args.py]
            API4[IO Struct<br/>io_struct.py]
        end
        
        subgraph "3. 算法实现层"
            ALG1[GRPO Actor<br/>engine/ppo/actor.py]
            ALG2[SFT Engine<br/>engine/sft/lm_engine.py]
            ALG3[RLVR Workflow<br/>workflow/rlvr.py]
            ALG4[Vision RLVR<br/>workflow/vision_rlvr.py]
        end
        
        subgraph "4. 引擎后端层"
            ENG1[FSDP Engine<br/>engine/fsdp_engine.py]
            ENG2[Base HF Engine<br/>engine/base_hf_engine.py]
            ENG3[SGLang Remote<br/>engine/sglang_remote.py]
        end
        
        subgraph "5. 工具支撑层"
            UTIL1[数据处理<br/>utils/data.py]
            UTIL2[分布式工具<br/>utils/distributed.py]
            UTIL3[FSDP工具<br/>utils/fsdp.py]
            UTIL4[保存加载<br/>utils/save_load.py]
        end
        
        subgraph "6. 启动器层"
            LAUNCH1[Local Launcher<br/>launcher/local.py]
            LAUNCH2[Ray Launcher<br/>launcher/ray.py]
            LAUNCH3[Slurm Launcher<br/>launcher/slurm.py]
        end
        
        subgraph "7. 基础依赖层"
            DEP1[PyTorch FSDP]
            DEP2[SGLang]
            DEP3[HuggingFace]
            DEP4[TensorDict]
        end
    end
    
    %% 正向依赖关系
    APP1 --> API1
    APP1 --> API2
    APP1 --> API3
    
    API1 --> ALG1
    API1 --> ALG2
    API2 --> ALG3
    API2 --> ALG4
    
    ALG1 --> ENG1
    ALG2 --> ENG1
    ALG3 --> ENG3
    
    ENG1 --> ENG2
    ENG3 --> UTIL1
    ENG1 --> UTIL2
    ENG1 --> UTIL3
    
    ENG2 --> DEP1
    ENG2 --> DEP3
    ENG3 --> DEP2
    
    APP1 --> LAUNCH1
    APP1 --> LAUNCH2
    APP1 --> LAUNCH3
    
    LAUNCH1 --> UTIL1
    
    %% 反向依赖 (特殊情况)
    ENG1 -.-> API4
    ALG3 -.-> API4
    
    style APP1 fill:#e1f5fe
    style API1 fill:#f3e5f5
    style ALG1 fill:#fff3e0
    style ENG1 fill:#e8f5e8
    style UTIL1 fill:#fce4ec
    style LAUNCH1 fill:#f1f8e9
    style DEP1 fill:#f9fbe7
```

### 层次职责说明

1. **应用层 (Entry Points)**: 用户直接使用的训练脚本，组合下层组件实现完整的训练流程
2. **API 抽象层**: 定义统一的接口规范，解耦各层之间的依赖关系
3. **算法实现层**: 实现具体的RL算法(GRPO)和工作流(RLVR)
4. **引擎后端层**: 提供训练和推理的底层实现，对接第三方库
5. **工具支撑层**: 提供通用的工具函数和实用程序
6. **启动器层**: 管理分布式启动和资源分配
7. **基础依赖层**: 外部依赖库和框架

### 反向依赖说明

AReaLite中存在以下反向依赖情况（用虚线表示）：
- **IO Struct反向依赖**: `fsdp_engine.py`和`rlvr.py`需要引用`io_struct.py`中定义的数据结构，这是为了类型安全和数据传递的需要
- 这种反向依赖是设计上的合理选择，因为数据结构定义需要在整个系统中保持一致

## 典型场景动态流程图

### GRPO训练场景序列图

```mermaid
sequenceDiagram
    participant U as 用户脚本<br/>gsm8k_grpo.py
    participant L as 启动器<br/>Launcher
    participant R as 推理引擎<br/>RemoteSGLangEngine
    participant A as 训练引擎<br/>FSDPPPOActor
    participant W as 工作流<br/>RLVRWorkflow
    participant S as SGLang服务器
    participant F as FSDP后端
    
    Note over U,F: AReaLite GRPO训练流程
    
    U->>L: 启动分布式训练
    L->>S: 启动SGLang推理服务器
    L->>U: 设置环境变量(服务器地址)
    
    U->>R: 初始化推理引擎
    R->>S: 连接推理服务器
    
    U->>A: 初始化GRPO训练引擎
    A->>F: 加载模型到FSDP
    
    U->>W: 创建RLVR工作流
    
    loop 训练循环
        U->>R: rollout_batch(data, workflow)
        R->>W: 执行工作流
        W->>S: 生成多个回复
        S-->>W: 返回生成结果
        W->>W: 计算奖励
        W-->>R: 返回批次数据
        R-->>U: 返回训练批次
        
        U->>A: grpo_update(batch)
        A->>F: 前向传播计算损失
        F->>F: 反向传播更新权重
        F-->>A: 返回训练统计
        A-->>U: 返回更新结果
        
        U->>A: upload_weights()
        A->>S: 更新推理服务器权重
    end
```

### SFT训练场景序列图

```mermaid
sequenceDiagram
    participant U as 用户脚本<br/>gsm8k_sft.py
    participant E as SFT引擎<br/>FSDPLMEngine
    participant F as FSDP后端
    participant D as 数据加载器<br/>StatefulDataLoader
    
    Note over U,D: AReaLite SFT训练流程
    
    U->>D: 创建数据加载器
    U->>E: 初始化SFT引擎
    E->>F: 加载模型到FSDP
    
    loop 训练循环
        U->>D: 获取下一批数据
        D-->>U: 返回数据批次
        
        U->>E: train_lm(data)
        E->>F: 计算语言模型损失
        F->>F: 执行反向传播
        F-->>E: 返回损失和统计
        E-->>U: 返回训练结果
        
        U->>E: 保存检查点(可选)
        U->>E: 评估模型(可选)
    end
```

## 层次详细展开 

### 1. 应用层 (Entry Points) 详细架构

#### 静态结构图

```mermaid
classDiagram
    class GSM8KGRPO {
        +main(args)
        +load_expr_config()
        +create_engines()
        +training_loop()
        -init_distributed()
        -setup_workflow()
    }
    
    class GSM8KSFT {
        +main(args)
        +load_expr_config()
        +create_engine()
        +training_loop() 
        -setup_dataloader()
    }
    
    class CLEVRCountGRPO {
        +main_grpo()
        +setup_vision_workflow()
        +handle_vision_data()
    }
    
    class CLEVRCountSFT {
        +main_sft()
        +setup_vision_data()
        +vision_training_loop()
    }
    
    GSM8KGRPO ..> RemoteSGLangEngine : 使用
    GSM8KGRPO ..> FSDPPPOActor : 使用
    GSM8KGRPO ..> RLVRWorkflow : 使用
    
    GSM8KSFT ..> FSDPLMEngine : 使用
    
    CLEVRCountGRPO ..> VisionRLVRWorkflow : 使用
```

#### 应用层动态交互图

```mermaid
sequenceDiagram
    participant C as 配置加载
    participant E as 引擎创建
    participant T as 训练循环
    participant S as 状态管理
    
    Note over C,S: 应用层内部流程
    
    C->>C: 解析命令行参数
    C->>C: 加载YAML配置
    C->>E: 传递配置信息
    
    E->>E: 初始化分布式环境
    E->>E: 创建推理引擎
    E->>E: 创建训练引擎
    E-->>T: 返回引擎实例
    
    loop 主训练循环
        T->>E: 执行rollout
        E-->>T: 返回数据
        T->>E: 执行训练更新
        E-->>T: 返回统计信息
        T->>S: 更新状态和指标
        T->>S: 保存检查点
    end
```

### 2. API 抽象层详细架构

#### 静态结构图

```mermaid
classDiagram
    class InferenceEngine {
        <<abstract>>
        +agenerate(req: LLMRequest)* 
        +rollout_batch(data, workflow)*
        +set_version(version)*
        +update_weights(meta)*
    }
    
    class TrainEngine {
        <<abstract>>
        +train_batch(input_, loss_fn, loss_weight_fn)*
        +eval_batch(input_, loss_fn, loss_weight_fn)*
        +forward(input_, output_seqlens)*
        +upload_weights(meta)*
    }
    
    class RolloutWorkflow {
        <<abstract>>
        +arun_episode(engine, data)*
    }
    
    class LLMRequest {
        +text: str
        +generation_config: dict
        +request_id: str
    }
    
    class LLMResponse {
        +text: str
        +logprobs: List
        +finish_reason: str
    }
    
    class WeightUpdateMeta {
        +version: int
        +model_name: str
        +weights_path: str
    }
    
    InferenceEngine --> LLMRequest : 接收
    InferenceEngine --> LLMResponse : 返回
    InferenceEngine --> WeightUpdateMeta : 使用
    TrainEngine --> WeightUpdateMeta : 生成
    RolloutWorkflow --> InferenceEngine : 调用
```

#### API层动态交互图

```mermaid
sequenceDiagram
    participant A as 应用代码
    participant I as InferenceEngine
    participant T as TrainEngine  
    participant W as RolloutWorkflow
    
    Note over A,W: API层接口调用流程
    
    A->>I: rollout_batch(data, workflow)
    I->>W: arun_episode(engine, data)
    W->>I: agenerate(req)
    I-->>W: LLMResponse
    W-->>I: TensorDict结果
    I-->>A: 批次数据
    
    A->>T: train_batch(input_, loss_fn)
    T->>T: 前向传播
    T->>T: 计算损失
    T->>T: 反向传播
    T-->>A: 训练统计
    
    A->>T: upload_weights(meta)
    T->>I: update_weights(meta)
    I-->>T: 确认更新
    T-->>A: 完成
```

### 3. 算法实现层详细架构

#### 静态结构图

```mermaid
classDiagram
    class FSDPPPOActor {
        +compute_logp(batch)
        +compute_advantages(batch)
        +grpo_update(batch)
        +upload_weights(meta)
        -_compute_policy_loss()
        -_compute_value_loss()
    }
    
    class FSDPLMEngine {
        +train_lm(data)
        +evaluate_lm(data)
        -lm_engine: LMEngine
    }
    
    class LMEngine {
        +train_lm(data)
        +evaluate_lm(data)
        -engine: TrainEngine
    }
    
    class RLVRWorkflow {
        +arun_episode(engine, data)
        +reward_fn: Callable
        -_generate_responses()
        -_compute_rewards()
    }
    
    class VisionRLVRWorkflow {
        +arun_episode(engine, data)
        +processor: AutoProcessor
        -_process_images()
    }
    
    FSDPPPOActor --|> TrainEngine
    FSDPLMEngine --> LMEngine
    LMEngine --> TrainEngine
    RLVRWorkflow --|> RolloutWorkflow
    VisionRLVRWorkflow --|> RLVRWorkflow
```

#### 算法层动态交互图

```mermaid
sequenceDiagram
    participant P as GRPO算法
    participant V as 价值函数
    participant R as 奖励计算
    participant L as 损失函数
    
    Note over P,L: GRPO算法内部流程
    
    P->>P: 接收rollout数据
    P->>V: 计算状态价值
    V-->>P: 返回价值估计
    
    P->>R: 计算优势函数
    R->>R: GAE计算
    R-->>P: 返回优势值
    
    loop GRPO更新循环
        P->>L: 计算策略损失
        L->>L: clip目标函数
        L-->>P: 策略损失
        
        P->>L: 计算价值损失
        L->>L: MSE损失
        L-->>P: 价值损失
        
        P->>P: 反向传播
        P->>P: 参数更新
    end
```

### 4. 引擎后端层详细架构

#### 静态结构图

```mermaid
classDiagram
    class FSDPEngine {
        +initialize(config, ft_spec)
        +train_batch(input_, loss_fn, loss_weight_fn)
        +forward(input_, output_seqlens)
        +save_checkpoint()
        +load_checkpoint()
        -model: ReaLModel
        -optimizer: Optimizer
        -lr_scheduler: LRScheduler
    }
    
    class BaseHFEngine {
        +_create_model()
        +_create_optimizer()
        +_create_lr_scheduler()
        +_prepare_batch()
        -config: TrainEngineConfig
    }
    
    class RemoteSGLangEngine {
        +agenerate(req)
        +rollout_batch(data, workflow)
        +set_version(version)
        +update_weights(meta)
        -server_urls: List[str]
        -session: aiohttp.ClientSession
    }
    
    FSDPEngine --|> BaseHFEngine
    FSDPEngine --|> TrainEngine
    RemoteSGLangEngine --|> InferenceEngine
```

#### 引擎层动态交互图

```mermaid
sequenceDiagram
    participant F as FSDPEngine
    participant M as Model
    participant O as Optimizer
    participant S as SGLangServer
    
    Note over F,S: 引擎后端执行流程
    
    F->>M: 前向传播
    M->>M: 计算logits
    M-->>F: 返回输出
    
    F->>F: 计算损失
    F->>M: 反向传播  
    M->>M: 计算梯度
    M-->>F: 梯度完成
    
    F->>O: 参数更新
    O->>O: 应用梯度
    O-->>F: 更新完成
    
    F->>S: 同步权重
    S->>S: 加载新权重
    S-->>F: 同步完成
```

### 5. 工具支撑层详细架构

#### 静态结构图

```mermaid
classDiagram
    class DataUtils {
        +pad_sequences_to_tensors()
        +concat_padded_tensors() 
        +pack_tensor_dict()
        +split_padded_tensor_dict()
    }
    
    class DistributedUtils {
        +init_custom_process_group()
        +all_gather_with_padding()
        +reduce_scatter_tensor()
    }
    
    class FSDPUtils {
        +apply_fsdp2()
        +create_fsdp_device_mesh() 
        +fsdp2_clip_grad_norm_()
        +CPUOffloadPolicy
        +MixedPrecisionPolicy
    }
    
    class SaveLoadUtils {
        +get_state_dict_from_repo()
        +save_checkpoint()
        +load_checkpoint()
        +convert_state_dict()
    }
```

### 6. 启动器层详细架构

#### 静态结构图

```mermaid
classDiagram
    class LocalLauncher {
        +launch(script, config, args)
        +start_sglang_servers()
        +run_training_script()
        -_setup_environment()
    }
    
    class RayLauncher {
        +launch(script, config, args)
        +setup_ray_cluster()
        +distribute_tasks()
        -_ray_remote_wrapper()
    }
    
    class SlurmLauncher {
        +launch(script, config, args) 
        +submit_slurm_jobs()
        +coordinate_nodes()
        -_generate_slurm_script()
    }
    
    class BaseLauncher {
        <<abstract>>
        +parse_allocation_mode()*
        +validate_config()*
        +setup_servers()*
    }
    
    LocalLauncher --|> BaseLauncher
    RayLauncher --|> BaseLauncher  
    SlurmLauncher --|> BaseLauncher
```

#### 启动器动态交互图

```mermaid
sequenceDiagram
    participant U as 用户命令
    participant L as Launcher
    participant S as SGLang服务器
    participant T as 训练进程
    
    Note over U,T: 分布式启动流程
    
    U->>L: 启动命令
    L->>L: 解析配置
    L->>L: 验证资源分配
    
    L->>S: 启动推理服务器集群
    S->>S: 加载模型
    S-->>L: 服务器就绪
    
    L->>L: 设置环境变量
    L->>T: 启动训练进程
    T->>S: 连接推理服务器
    T->>T: 开始训练循环
```

## 算法实现说明

### GRPO vs PPO

**重要声明**: AReaLite实际实现的是**GRPO (Group Relative Policy Optimization)**算法，而不是标准的PPO算法，这是一个关键的技术细节：

#### GRPO算法特点
- **Group-based设计**: 支持group_size配置，对多个样本进行分组优化
- **GRPO损失函数**: 使用专门的`grpo_loss_fn`函数 (位于`arealite/engine/ppo/actor.py`第286行)
- **配置差异**: 包含group_reward_norm、group_adv_norm等GRPO特有参数

#### 代码实现证据
```python
# 文件：arealite/engine/ppo/actor.py
def grpo_loss_fn(
    logits: torch.Tensor,
    input_data: Dict,
    temperature: float,
    eps_clip: float,
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
):
    """GRPO算法的实际损失函数实现"""
    
# 文件：arealite/api/cli_args.py  
class GRPOConfig:
    """GRPO专用配置类"""
```

#### 为什么存在命名混淆
1. **历史原因**: 代码中的类名`PPOActor`和文件路径`engine/ppo/`保持了早期PPO实现的命名
2. **兼容性**: 保持了与Core系统中PPO接口的兼容性
3. **实际算法**: 尽管命名为PPO相关，但AReaLite实际执行的是GRPO算法逻辑

#### 示例文件确认
- `examples/arealite/gsm8k_grpo.py` - GSM8K的GRPO训练示例
- `examples/arealite/clevr_count_70k_grpo.py` - CLEVR的GRPO训练示例

这种设计确保了AReaLite在保持简单易用的同时，提供了比标准PPO更适合群组优化的GRPO算法实现。
