# AReaL架构图表总览

本文档包含AReaL系统的完整架构图表，展示AReaLite和Core两个系统的设计对比。

## 整体架构对比

### 高层架构对比图

```mermaid
graph TB
    subgraph "AReaLite Architecture (AI-Centric)"
        A1[Entry Point<br/>gsm8k_grpo.py] --> A2[Engine Objects]
        A2 --> A3[RemoteSGLangEngine]
        A2 --> A4[FSDPPPOActor]
        A3 --> A5["rollout_batch()"]
        A4 --> A6["ppo_update()"]
        A5 --> A7[PyTorch/SGLang]
        A6 --> A7
        
        style A1 fill:#e1f5fe
        style A2 fill:#f3e5f5
        style A7 fill:#e8f5e8
    end
    
    subgraph "Core Architecture (System-Centric)"
        B1[Config Files<br/>sync-ppo.yaml] --> B2[Experiment Framework<br/>PPOMATHConfig]
        B2 --> B3[Master Scheduler<br/>MasterWorker]
        B3 --> B4[Workers]
        B4 --> B5[RolloutWorker]
        B4 --> B6[ModelWorker]
        B4 --> B7[ControllerWorker]
        B5 --> B8[MFC Calls]
        B6 --> B8
        B7 --> B8
        B8 --> B9[generate_MFC]
        B8 --> B10[train_step_MFC]
        B8 --> B11[compute_ref_MFC]
        B9 --> B12[PyTorch/SGLang]
        B10 --> B12
        B11 --> B12
        
        style B1 fill:#fff3e0
        style B2 fill:#fce4ec
        style B3 fill:#f1f8e9
        style B4 fill:#e3f2fd
        style B8 fill:#f9fbe7
        style B12 fill:#e8f5e8
    end
```

### 详细层次架构对比

```mermaid
graph TB
    subgraph "AReaLite 详细架构"
        subgraph "应用层"
            AL1[gsm8k_grpo.py<br/>gsm8k_sft.py<br/>clevr_count_*.py]
        end
        
        subgraph "API层"
            AL2[Engine API<br/>Workflow API<br/>CLI Args]
        end
        
        subgraph "算法层"
            AL3[PPO Actor<br/>SFT Engine<br/>RLVR Workflow]
        end
        
        subgraph "引擎层"
            AL4[FSDP Engine<br/>SGLang Remote<br/>Base HF Engine]
        end
        
        subgraph "工具层"
            AL5[Data Utils<br/>FSDP Utils<br/>Distributed Utils]
        end
        
        subgraph "启动层"
            AL6[Local/Ray/Slurm<br/>Launcher]
        end
        
        AL1 --> AL2
        AL2 --> AL3
        AL3 --> AL4
        AL4 --> AL5
        AL1 --> AL6
    end
    
    subgraph "Core 详细架构"
        subgraph "应用层"
            CL1[main_*_ppo.py<br/>run_*.sh]
        end
        
        subgraph "实验层"
            CL2[PPO Math Exp<br/>Async RL Exp<br/>Common Config]
        end
        
        subgraph "API层"
            CL3[System API<br/>Model API<br/>Data API<br/>DFG API]
        end
        
        subgraph "系统层"
            CL4[Master Worker<br/>Model Worker<br/>Rollout Worker]
        end
        
        subgraph "实现层"
            CL5[Model Interface<br/>Model Backend<br/>Neural Networks]
        end
        
        subgraph "基础层"
            CL6[Constants<br/>Logging<br/>Topology<br/>Network]
        end
        
        CL1 --> CL2
        CL2 --> CL3
        CL3 --> CL4
        CL4 --> CL5
        CL5 --> CL6
    end
    
    style AL1 fill:#e1f5fe
    style AL2 fill:#f3e5f5
    style AL3 fill:#fff3e0
    style AL4 fill:#e8f5e8
    style AL5 fill:#fce4ec
    style AL6 fill:#f1f8e9
    
    style CL1 fill:#e1f5fe
    style CL2 fill:#f3e5f5  
    style CL3 fill:#fff3e0
    style CL4 fill:#e8f5e8
    style CL5 fill:#fce4ec
    style CL6 fill:#f1f8e9
```

## 代码复杂度对比

```mermaid
graph LR
    subgraph "AReaLite: 3 层抽象"
        AL1[用户脚本] --> AL2[引擎对象] --> AL3[后端实现]
    end
    
    subgraph "Core: 6 层抽象"
        CL1[用户配置] --> CL2[实验框架] --> CL3[主调度器]
        CL3 --> CL4[工作节点执行] --> CL5[MFC调用] --> CL6[后端实现]
    end
    
    style AL1 fill:#e1f5fe
    style AL2 fill:#f3e5f5
    style AL3 fill:#e8f5e8
    
    style CL1 fill:#fff3e0
    style CL2 fill:#fce4ec
    style CL3 fill:#f1f8e9
    style CL4 fill:#e3f2fd
    style CL5 fill:#f9fbe7
    style CL6 fill:#e8f5e8
```

## 共享组件关系图

```mermaid
graph TB
    subgraph "共享组件"
        SHARED1[ReaLModel<br/>模型实现]
        SHARED2[Stats Tracker<br/>统计追踪]
        SHARED3[PPO Functions<br/>算法函数]
        SHARED4[Math Verify<br/>数学验证]
        SHARED5[HF Registry<br/>模型转换]
        
        style SHARED1 fill:#e8f5e8
        style SHARED2 fill:#e8f5e8
        style SHARED3 fill:#e8f5e8
        style SHARED4 fill:#e8f5e8
        style SHARED5 fill:#e8f5e8
    end
    
    subgraph "AReaLite专有"
        LITE1[FSDP Engine]
        LITE2[Workflow API]
        LITE3[Launcher System]
        
        style LITE1 fill:#e1f5fe
        style LITE2 fill:#e1f5fe
        style LITE3 fill:#e1f5fe
    end
    
    subgraph "Core专有"
        CORE1[Master Worker]
        CORE2[DFG System]
        CORE3[Worker Network]
        
        style CORE1 fill:#fff3e0
        style CORE2 fill:#fff3e0
        style CORE3 fill:#fff3e0
    end
    
    %% 依赖关系
    LITE1 -.-> SHARED1
    LITE1 -.-> SHARED2
    LITE2 -.-> SHARED3
    LITE3 -.-> SHARED4
    
    CORE1 -.-> SHARED1
    CORE1 -.-> SHARED2
    CORE2 -.-> SHARED3
    CORE3 -.-> SHARED5
```

## 数据流对比图

```mermaid
graph TB
    subgraph "AReaLite 数据流"
        subgraph "简化流程"
            AD1[原始数据] --> AD2[TensorDict]
            AD2 --> AD3[Engine处理]
            AD3 --> AD4[直接输出]
        end
    end
    
    subgraph "Core 数据流"
        subgraph "复杂流程"
            CD1[原始数据] --> CD2[SequenceSample]
            CD2 --> CD3[MFC传输]
            CD3 --> CD4[Worker处理]
            CD4 --> CD5[数据重分布]
            CD5 --> CD6[最终输出]
        end
    end
    
    style AD1 fill:#e1f5fe
    style AD2 fill:#f3e5f5
    style AD3 fill:#e8f5e8
    style AD4 fill:#fce4ec
    
    style CD1 fill:#fff3e0
    style CD2 fill:#fce4ec
    style CD3 fill:#f1f8e9
    style CD4 fill:#e3f2fd
    style CD5 fill:#f9fbe7
    style CD6 fill:#e8f5e8
```

## 扩展性对比图

```mermaid
graph TB
    subgraph "AReaLite 扩展性"
        subgraph "单节点到中等规模"
            AS1[1-8 GPU<br/>优秀体验] 
            AS2[8-64 GPU<br/>良好性能]
            AS3[64+ GPU<br/>可能瓶颈]
        end
    end
    
    subgraph "Core 扩展性"
        subgraph "中等到大规模"
            CS1[1-64 GPU<br/>复杂设置]
            CS2[64-256 GPU<br/>优秀性能] 
            CS3[256+ GPU<br/>工业级别]
        end
    end
    
    style AS1 fill:#4caf50
    style AS2 fill:#8bc34a
    style AS3 fill:#ffc107
    
    style CS1 fill:#ffc107
    style CS2 fill:#4caf50
    style CS3 fill:#2196f3
```

## 训练流程对比

### AReaLite训练流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant AL as AReaLite
    participant E as Engine
    participant PT as PyTorch/SGLang
    
    Note over U,PT: AReaLite训练流程
    U->>AL: 启动训练脚本
    AL->>E: 创建Engine对象
    loop 训练循环
        AL->>E: rollout_batch()
        E->>PT: 直接调用
        PT-->>E: 返回结果
        AL->>E: ppo_update()
        E->>PT: 直接调用
        PT-->>E: 返回结果
    end
```

### Core训练流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant C as 配置
    participant M as Master
    participant W as Workers
    participant MFC as MFC系统
    participant PT as PyTorch/SGLang
    
    Note over U,PT: Core训练流程
    U->>C: 提供配置文件
    C->>M: 创建实验框架
    M->>W: 调度Worker
    loop 训练循环
        M->>W: 分配任务
        W->>MFC: 调用MFC
        MFC->>PT: 执行计算
        PT-->>MFC: 返回结果
        MFC-->>W: 返回结果
        W-->>M: 汇报状态
    end
```

## 系统启动对比

### AReaLite启动流程

```mermaid
graph TB
    START[启动命令] --> PARSE[解析参数]
    PARSE --> LAUNCH[选择启动器]
    LAUNCH --> SERVER[启动SGLang服务器]
    SERVER --> ENV[设置环境变量]
    ENV --> SCRIPT[运行训练脚本]
    SCRIPT --> INIT[初始化Engine]
    INIT --> TRAIN[开始训练]
    
    style START fill:#e1f5fe
    style SCRIPT fill:#f3e5f5
    style TRAIN fill:#e8f5e8
```

### Core启动流程

```mermaid
graph TB
    CONFIG[配置文件] --> EXP[实验定义]
    EXP --> MASTER[启动Master]
    MASTER --> SCHEDULE[调度Workers]
    SCHEDULE --> MODEL[Model Workers]
    SCHEDULE --> ROLLOUT[Rollout Workers]
    SCHEDULE --> GEN[Generation Servers]
    MODEL --> DFG[执行数据流图]
    ROLLOUT --> DFG
    GEN --> DFG
    DFG --> RESULT[训练结果]
    
    style CONFIG fill:#fff3e0
    style MASTER fill:#f1f8e9
    style DFG fill:#f9fbe7
    style RESULT fill:#e8f5e8
```

## 架构演进趋势

```mermaid
graph TB
    subgraph "当前状态"
        CURRENT1[AReaLite<br/>轻量级框架]
        CURRENT2[Core<br/>完整系统]
    end
    
    subgraph "未来演进"
        FUTURE1[统一API层]
        FUTURE2[可插拔后端]
        FUTURE3[渐进式复杂度]
    end
    
    subgraph "目标架构"
        TARGET[90%功能<br/>10%复杂度]
    end
    
    CURRENT1 --> FUTURE1
    CURRENT2 --> FUTURE1
    FUTURE1 --> FUTURE2
    FUTURE2 --> FUTURE3
    FUTURE3 --> TARGET
    
    style CURRENT1 fill:#e1f5fe
    style CURRENT2 fill:#fff3e0
    style TARGET fill:#4caf50
```
