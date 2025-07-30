# Core系统架构概览

本文档提供AReaL Core系统的全面架构概览，采用视觉导向的方法展示其复杂的分布式架构设计。

## 系统架构概览

```mermaid
graph TB
    subgraph "Core系统全景架构"
        subgraph "1. 应用入口层"
            APP1[main_sync_ppo.py<br/>同步PPO训练]
            APP2[main_async_ppo.py<br/>异步PPO训练]
            APP3[main_sft.py<br/>监督微调]
            SHELL[Shell脚本<br/>环境配置]
        end
        
        subgraph "2. 实验配置层"
            EXP_SYNC[PPO Math实验<br/>同步配置]
            EXP_ASYNC[异步RL实验<br/>异步配置]
            EXP_BASE[通用实验基类<br/>共享配置]
            DFG[数据流图<br/>MFC定义]
        end
        
        subgraph "3. 系统协调层"
            MASTER[Master Worker<br/>全局协调器]
            CONTROLLER[Controller<br/>任务调度]
            DATA_MGR[Data Manager<br/>数据管理]
            FUNC_EXEC[Function Executor<br/>函数执行器]
        end
        
        subgraph "4. 分布式Worker层"
            MODEL_W[Model Workers<br/>训练节点集群]
            ROLLOUT_W[Rollout Workers<br/>数据收集节点]
            GEN_MGR[Generation Manager<br/>生成服务管理]
            GEN_SERVERS[Generation Servers<br/>推理服务集群]
        end
        
        subgraph "5. 模型执行层"
            PPO_IMPL[PPO实现<br/>策略优化]
            SFT_IMPL[SFT实现<br/>监督学习]
            REWARD_IMPL[奖励模型<br/>价值评估]
            PIPELINE[Pipeline引擎<br/>并行计算]
        end
        
        subgraph "6. 通信基础设施"
            RPC[RPC通信<br/>进程间调用]
            ZMQ[ZeroMQ<br/>消息队列]
            TCP[TCP网络<br/>数据传输]
            MFC[MFC系统<br/>模型函数调用]
        end
        
        subgraph "7. 外部依赖"
            PYTORCH[PyTorch<br/>深度学习框架]
            RAY[Ray/Slurm<br/>集群管理]
            NCCL[NCCL<br/>通信后端]
            HF[HuggingFace<br/>模型库]
        end
    end
    
    %% 连接关系
    APP1 --> EXP_SYNC
    APP2 --> EXP_ASYNC
    APP3 --> EXP_BASE
    SHELL --> APP1
    SHELL --> APP2
    
    EXP_SYNC --> DFG
    EXP_ASYNC --> DFG
    EXP_BASE --> DFG
    
    DFG --> MASTER
    MASTER --> CONTROLLER
    MASTER --> DATA_MGR
    MASTER --> FUNC_EXEC
    
    CONTROLLER --> MODEL_W
    CONTROLLER --> ROLLOUT_W
    CONTROLLER --> GEN_MGR
    GEN_MGR --> GEN_SERVERS
    
    MODEL_W --> PPO_IMPL
    MODEL_W --> SFT_IMPL
    ROLLOUT_W --> REWARD_IMPL
    PPO_IMPL --> PIPELINE
    SFT_IMPL --> PIPELINE
    
    MASTER --> RPC
    MODEL_W --> ZMQ
    ROLLOUT_W --> TCP
    DATA_MGR --> MFC
    
    PIPELINE --> PYTORCH
    CONTROLLER --> RAY
    ZMQ --> NCCL
    PPO_IMPL --> HF
    
    %% 样式定义
    classDef app fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef exp fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef coord fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef worker fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef impl fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef comm fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    classDef deps fill:#fff8e1,stroke:#fbc02d,stroke-width:2px
    
    class APP1,APP2,APP3,SHELL app
    class EXP_SYNC,EXP_ASYNC,EXP_BASE,DFG exp
    class MASTER,CONTROLLER,DATA_MGR,FUNC_EXEC coord
    class MODEL_W,ROLLOUT_W,GEN_MGR,GEN_SERVERS worker
    class PPO_IMPL,SFT_IMPL,REWARD_IMPL,PIPELINE impl
    class RPC,ZMQ,TCP,MFC comm
    class PYTORCH,RAY,NCCL,HF deps
```

## 核心特征

### 1. 分布式架构设计
- **Master-Worker模式**: Master节点协调多个Worker节点
- **多类型Worker**: Model Worker、Rollout Worker、Generation Server
- **动态负载均衡**: 根据计算负载动态分配任务

### 2. 数据流图(DFG)驱动
- **MFC系统**: 模型函数调用抽象
- **依赖管理**: 自动处理计算依赖关系
- **异步执行**: 支持异步和同步两种执行模式

### 3. 高度可扩展性
- **多GPU支持**: 原生支持100+ GPU的大规模训练
- **灵活拓扑**: 支持多种集群拓扑结构
- **弹性伸缩**: 动态添加/移除计算节点

## 同步PPO训练流程

```mermaid
sequenceDiagram
    participant U as 用户脚本
    participant M as Master Worker
    participant MW as Model Workers
    participant RW as Rollout Workers
    participant GS as Generation Servers
    participant D as 数据流图
    
    Note over U,D: Core同步PPO完整训练流程
    
    U->>M: 启动实验配置
    M->>M: 构建数据流图
    M->>MW: 启动Model Workers
    M->>RW: 启动Rollout Workers
    M->>GS: 启动Generation Servers
    
    Note over M,D: 训练循环开始
    
    M->>D: 执行数据流图
    
    %% Rollout阶段
    D->>RW: 执行generate_MFC
    RW->>GS: 请求文本生成
    GS->>GS: 并行生成多个回复
    GS-->>RW: 返回生成结果
    RW->>RW: 计算奖励分数
    RW-->>D: 返回rollout数据
    
    %% 训练阶段
    D->>MW: 执行train_step_MFC
    MW->>MW: 计算PPO损失
    MW->>MW: 执行梯度更新
    MW->>MW: 同步模型参数
    MW-->>D: 返回训练统计
    
    %% 参考模型计算
    D->>MW: 执行compute_ref_MFC
    MW->>MW: 计算参考logprobs
    MW-->>D: 返回参考数据
    
    D-->>M: 完成训练步骤
    
    %% 权重同步
    M->>GS: 同步最新权重
    GS->>GS: 更新生成模型
    
    Note over M,D: 循环执行直到收敛
```

## 异步PPO训练流程

```mermaid
sequenceDiagram
    participant M as Master Worker
    participant MW as Model Workers
    participant RW as Rollout Workers
    participant GM as Generation Manager
    participant GS as Generation Servers
    participant B as Buffer System
    
    Note over M,B: Core异步PPO训练流程
    
    M->>MW: 启动训练节点集群
    M->>RW: 启动Rollout节点集群
    M->>GM: 启动生成管理器
    GM->>GS: 管理生成服务器池
    M->>B: 初始化缓冲区系统
    
    par 异步Rollout流程
        loop 持续数据收集
            RW->>GM: 请求可用生成服务器
            GM-->>RW: 分配服务器实例
            RW->>GS: 提交生成请求
            GS->>GS: 异步文本生成
            GS-->>RW: 返回生成结果
            RW->>RW: 计算奖励和优势
            RW->>B: 存储训练数据
        end
    and 异步训练流程
        loop 持续模型更新
            MW->>B: 从缓冲区获取批次
            B-->>MW: 返回训练数据
            MW->>MW: 计算PPO损失
            MW->>MW: 执行参数更新
            MW->>MW: 应用梯度
            MW->>GM: 上传新权重
            GM->>GS: 推送权重到生成服务器
            GS->>GS: 热更新模型权重
        end
    and 协调控制流程
        loop 系统监控
            M->>B: 监控缓冲区状态
            M->>MW: 监控训练进度
            M->>RW: 监控数据收集率
            M->>GM: 监控服务器健康状态
            M->>M: 调整系统参数
        end
    end
```

## MFC系统详解

### MFC定义结构

```mermaid
classDiagram
    class MFCDef {
        +model_name: str
        +interface_type: ModelInterfaceType
        +interface_kwargs: dict
        +n_mbs: int
        +input_spec: dict
        +output_spec: dict
        
        +create_interface()
        +validate_inputs()
        +format_outputs()
    }
    
    class ModelInterfaceType {
        <<enumeration>>
        INFERENCE
        TRAIN_STEP
        GENERATE
        COMPUTE_REF_LOGPROBS
    }
    
    class DataFlowGraph {
        +nodes: List[MFCDef]
        +edges: List[Edge]
        +execution_order: List[str]
        
        +add_mfc_dataflow()
        +compute_dependencies()
        +execute_graph()
    }
    
    class FunctionExecutor {
        +mfc_registry: Dict
        +buffer_manager: BufferManager
        
        +execute_mfc()
        +coordinate_dataflow()
        +handle_dependencies()
    }
    
    MFCDef --> ModelInterfaceType
    DataFlowGraph --> MFCDef
    FunctionExecutor --> DataFlowGraph
```

### MFC执行流程

```mermaid
flowchart TD
    subgraph "MFC执行管道"
        START([开始MFC执行])
        PARSE[解析MFC定义]
        VALIDATE[验证输入数据]
        ROUTE[路由到目标Worker]
        EXECUTE[执行模型接口]
        COLLECT[收集输出结果]
        BUFFER[写入输出缓冲区]
        END([MFC执行完成])
        
        START --> PARSE
        PARSE --> VALIDATE
        VALIDATE --> ROUTE
        ROUTE --> EXECUTE
        EXECUTE --> COLLECT
        COLLECT --> BUFFER
        BUFFER --> END
    end
    
    subgraph "并行执行示例"
        MFC1[generate_MFC]
        MFC2[train_step_MFC]
        MFC3[compute_ref_MFC]
        
        MFC1 --> MFC2
        MFC1 --> MFC3
    end
    
    classDef mfc fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef flow fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class PARSE,VALIDATE,ROUTE,EXECUTE,COLLECT,BUFFER flow
    class MFC1,MFC2,MFC3 mfc
```

## Worker节点详细架构

### Model Worker架构

```mermaid
graph TB
    subgraph "Model Worker内部架构"
        IFACE[接口层<br/>PPO/SFT/Reward接口]
        MODEL[模型层<br/>ReaLModel实例]
        PARALLEL[并行层<br/>Pipeline/Tensor并行]
        COMM[通信层<br/>梯度同步]
        
        IFACE --> MODEL
        MODEL --> PARALLEL
        PARALLEL --> COMM
    end
    
    subgraph "支持的操作"
        TRAIN[train_step<br/>训练步骤]
        INFER[inference<br/>推理计算]
        GEN[generate<br/>文本生成]
        REF[compute_ref<br/>参考logprobs]
        
        IFACE --> TRAIN
        IFACE --> INFER
        IFACE --> GEN
        IFACE --> REF
    end
    
    classDef worker fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef op fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class IFACE,MODEL,PARALLEL,COMM worker
    class TRAIN,INFER,GEN,REF op
```

### Rollout Worker架构

```mermaid
graph TB
    subgraph "Rollout Worker内部架构"
        ENV[环境接口<br/>任务环境]
        AGENT[智能体<br/>策略执行器]
        REWARD[奖励计算<br/>评分函数]
        BUFFER[数据缓冲<br/>轨迹存储]
        
        ENV --> AGENT
        AGENT --> REWARD
        REWARD --> BUFFER
    end
    
    subgraph "数据收集流程"
        PROMPT[获取提示]
        GENERATE[请求生成]
        SCORE[计算奖励]
        FORMAT[格式化数据]
        
        PROMPT --> GENERATE
        GENERATE --> SCORE
        SCORE --> FORMAT
    end
    
    classDef worker fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef flow fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class ENV,AGENT,REWARD,BUFFER worker
    class PROMPT,GENERATE,SCORE,FORMAT flow
```

## 通信架构深度解析

### 多层通信栈

```mermaid
graph TB
    subgraph "Core通信架构栈"
        subgraph "应用层协议"
            MFC_PROTO[MFC协议<br/>模型函数调用]
            DFG_PROTO[DFG协议<br/>数据流控制]
        end
        
        subgraph "中间件层"
            RPC_LAYER[RPC层<br/>远程过程调用]
            MSG_QUEUE[消息队列<br/>ZeroMQ]
        end
        
        subgraph "传输层"
            TCP_CONN[TCP连接<br/>可靠传输]
            IPC_CONN[进程间通信<br/>共享内存]
        end
        
        subgraph "物理层"
            ETHERNET[以太网<br/>网络连接]
            INFINIBAND[InfiniBand<br/>高速互联]
        end
    end
    
    MFC_PROTO --> RPC_LAYER
    DFG_PROTO --> MSG_QUEUE
    RPC_LAYER --> TCP_CONN
    MSG_QUEUE --> IPC_CONN
    TCP_CONN --> ETHERNET
    IPC_CONN --> INFINIBAND
    
    classDef proto fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef middle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef trans fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef phys fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class MFC_PROTO,DFG_PROTO proto
    class RPC_LAYER,MSG_QUEUE middle
    class TCP_CONN,IPC_CONN trans
    class ETHERNET,INFINIBAND phys
```

### 消息流示例

```mermaid
sequenceDiagram
    participant M as Master
    participant MW as Model Worker
    participant RW as Rollout Worker
    participant GS as Gen Server
    
    Note over M,GS: 复杂消息协调示例
    
    %% 启动阶段
    M->>MW: RPC: 启动模型服务
    M->>RW: RPC: 启动Rollout服务
    M->>GS: TCP: 启动生成服务器
    
    %% 数据流阶段
    M->>RW: MFC: generate_request
    RW->>GS: HTTP: 生成请求
    GS-->>RW: HTTP: 生成响应
    RW-->>M: MFC: rollout数据
    
    M->>MW: MFC: train_step_request
    MW->>MW: IPC: 模型前向/反向
    MW-->>M: MFC: 训练统计
    
    %% 同步阶段
    M->>GS: TCP: 权重更新
    GS->>GS: 本地: 模型热更新
    
    Note over M,GS: 一轮训练完成
```

## 性能优化特性

### 1. 计算优化
- **Pipeline并行**: 模型层间流水线
- **Tensor并行**: 张量维度并行
- **Data并行**: 数据批次并行
- **混合精度**: FP16/BF16支持

### 2. 通信优化
- **梯度压缩**: 减少通信开销
- **异步通信**: 计算通信重叠
- **拓扑感知**: 网络拓扑优化
- **带宽聚合**: 多路径传输

### 3. 内存优化
- **激活重计算**: 内存换时间
- **梯度累积**: 减少内存峰值
- **模型分片**: 大模型分布存储
- **动态分配**: 按需内存管理

```mermaid
graph LR
    subgraph "性能优化矩阵"
        subgraph "计算维度"
            COMP1[Pipeline并行]
            COMP2[Tensor并行]
            COMP3[Data并行]
        end
        
        subgraph "通信维度"
            COMM1[异步通信]
            COMM2[梯度压缩]
            COMM3[拓扑优化]
        end
        
        subgraph "内存维度"
            MEM1[激活重计算]
            MEM2[模型分片]
            MEM3[动态分配]
        end
        
        COMP1 <--> COMM1
        COMP2 <--> COMM2
        COMP3 <--> COMM3
        COMM1 <--> MEM1
        COMM2 <--> MEM2
        COMM3 <--> MEM3
    end
    
    classDef comp fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef comm fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef mem fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class COMP1,COMP2,COMP3 comp
    class COMM1,COMM2,COMM3 comm
    class MEM1,MEM2,MEM3 mem
```

## Core系统优势

### 1. 企业级可靠性
- **容错机制**: 节点故障自动恢复
- **监控告警**: 全面系统监控
- **版本管理**: 模型版本控制
- **审计追踪**: 完整操作日志

### 2. 极致性能
- **大规模扩展**: 支持1000+ GPU
- **高效通信**: 优化的通信栈
- **智能调度**: 动态负载均衡
- **资源优化**: 最大化硬件利用率

### 3. 算法灵活性
- **多算法支持**: PPO、SFT、奖励建模
- **自定义接口**: 可扩展的算法框架
- **实验管理**: 复杂实验配置
- **A/B测试**: 多版本并行测试

这个Core架构文档展示了一个成熟的工业级分布式RLHF系统的复杂性和强大功能，为大规模语言模型训练提供了坚实的基础设施支持。
