# AReaLite vs Core 架构对比

本文档提供AReaLite和Core架构的并排比较，重点展示它们的设计理念、权衡取舍和使用场景。

## 高层架构对比

```mermaid
graph TB
    subgraph "AReaLite架构 (1-64 GPU)"
        subgraph "AL_APP[应用层]"
            AL_SCRIPT[训练脚本<br/>gsm8k_grpo.py]
            AL_CONFIG[YAML配置<br/>简单设置]
        end
        
        subgraph "AL_EXEC[执行层]"
            AL_LAUNCHER[启动器<br/>local/ray/slurm]
            AL_SGLANG[SGLang服务器<br/>推理集群]
        end
        
        subgraph "AL_ALGO[算法层]"  
            AL_GRPO[GRPO Actor<br/>分组优化]
            AL_RLVR[RLVR工作流<br/>奖励计算]
        end
        
        subgraph "AL_ENGINE[引擎层]"
            AL_FSDP[FSDP引擎<br/>训练后端]
            AL_REMOTE[远程引擎<br/>推理代理]
        end
        
        subgraph "AL_INFRA[基础设施]"
            AL_FSDP2[FSDP2<br/>参数分片]
            AL_SGLANG_LIB[SGLang<br/>高性能推理]
        end
        
        AL_SCRIPT --> AL_LAUNCHER
        AL_CONFIG --> AL_LAUNCHER
        AL_LAUNCHER --> AL_SGLANG
        AL_SCRIPT --> AL_GRPO
        AL_SCRIPT --> AL_RLVR
        AL_GRPO --> AL_FSDP
        AL_RLVR --> AL_REMOTE
        AL_REMOTE --> AL_SGLANG
        AL_FSDP --> AL_FSDP2
        AL_SGLANG --> AL_SGLANG_LIB
    end
    
    subgraph "Core架构 (64+ GPU)"
        subgraph "C_APP[应用层]"
            C_SCRIPT[训练脚本<br/>main_async_ppo.py] 
            C_CONFIG[复杂配置<br/>实验设置]
        end
        
        subgraph "C_SYSTEM[系统层]"
            C_MASTER[主控Worker<br/>协调管理]
            C_MODEL[模型Workers<br/>训练进程]
            C_ROLLOUT[Rollout Workers<br/>生成进程]
            C_GSERVER[GServer管理器<br/>服务器管理]
        end
        
        subgraph "C_ALGO[算法层]"
            C_PPO[PPO接口<br/>标准PPO]
            C_MFC[MFC系统<br/>消息传递]
        end
        
        subgraph "C_ENGINE[引擎层]"
            C_TRAINER[训练引擎<br/>分布式训练]
            C_INFERENCE[推理引擎<br/>生成后端]
        end
        
        subgraph "C_INFRA[基础设施]"
            C_DIST[自定义分布式<br/>Worker系统]
            C_COMM[TCP通信<br/>Worker间消息]
        end
        
        C_SCRIPT --> C_MASTER
        C_CONFIG --> C_MASTER
        C_MASTER --> C_MODEL
        C_MASTER --> C_ROLLOUT
        C_MASTER --> C_GSERVER
        C_MODEL --> C_PPO
        C_ROLLOUT --> C_MFC
        C_PPO --> C_TRAINER
        C_MFC --> C_INFERENCE
        C_TRAINER --> C_DIST
        C_INFERENCE --> C_COMM
    end
    
    %% Styling
    classDef arealite fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    
    class AL_SCRIPT,AL_CONFIG,AL_LAUNCHER,AL_SGLANG,AL_GRPO,AL_RLVR,AL_FSDP,AL_REMOTE,AL_FSDP2,AL_SGLANG_LIB arealite
    class C_SCRIPT,C_CONFIG,C_MASTER,C_MODEL,C_ROLLOUT,C_GSERVER,C_PPO,C_MFC,C_TRAINER,C_INFERENCE,C_DIST,C_COMM core
```

## 关键差异总结

| 方面 | AReaLite | Core |
|--------|----------|------|
| **目标规模** | 1-64 GPU | 64+ GPU |
| **设计理念** | AI为中心，简单 | 系统为中心，全面 |
| **算法** | GRPO (分组优化) | PPO (标准) |
| **架构层次** | 7层 | 6层 |
| **入口点** | 单个训练脚本 | 多种worker类型 |
| **配置** | 基于YAML | 复杂实验配置 |
| **推理后端** | SGLang服务器 | 自定义推理workers |
| **通信** | FSDP2 + HTTP | 自定义TCP + MFC |
| **部署** | 启动器管理 | Worker管理 |
| **复杂度** | 低到中等 | 高 |

## 详细组件对比

### 1. 算法实现

#### AReaLite: GRPO算法
```mermaid
graph LR
    subgraph "GRPO特性"
        GRP_SIZE[分组大小<br/>可配置批处理]
        GRP_NORM[分组归一化<br/>奖励和优势]
        GRP_LOSS[分组损失函数<br/>相对优化]
        GRP_STAB[增强稳定性<br/>多回复场景]
    end
    
    GRP_SIZE --> GRP_NORM
    GRP_NORM --> GRP_LOSS  
    GRP_LOSS --> GRP_STAB
    
    classDef grpo fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    class GRP_SIZE,GRP_NORM,GRP_LOSS,GRP_STAB grpo
```

#### Core: 标准PPO
```mermaid
graph LR
    subgraph "PPO特性"
        PPO_CLIP[截断目标<br/>标准实现]
        PPO_ADV[独立优势<br/>每样本计算]
        PPO_LOSS[标准损失<br/>策略 + 价值]
        PPO_STABLE[经验证稳定性<br/>成熟算法]
    end
    
    PPO_CLIP --> PPO_ADV
    PPO_ADV --> PPO_LOSS
    PPO_LOSS --> PPO_STABLE
    
    classDef ppo fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    class PPO_CLIP,PPO_ADV,PPO_LOSS,PPO_STABLE ppo
```

### 2. 基础设施对比

#### AReaLite基础设施
```mermaid
graph TB
    subgraph "AReaLite基础设施栈"
        AL_USER[用户脚本层<br/>简单入口点]
        AL_LAUNCH[启动器层<br/>资源管理]
        AL_SGLANG[SGLang层<br/>推理服务器]
        AL_ENGINE[引擎层<br/>FSDP + 远程]
        AL_BACKEND[后端层<br/>FSDP2 + SGLang]
    end
    
    AL_USER --> AL_LAUNCH
    AL_LAUNCH --> AL_SGLANG
    AL_LAUNCH --> AL_ENGINE
    AL_ENGINE --> AL_BACKEND
    
    classDef simple fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    class AL_USER,AL_LAUNCH,AL_SGLANG,AL_ENGINE,AL_BACKEND simple
```

#### Core基础设施
```mermaid
graph TB
    subgraph "Core基础设施栈"
        C_USER[用户脚本层<br/>复杂设置]
        C_COORD[协调层<br/>主从架构]
        C_WORKERS[Worker层<br/>多种类型]
        C_ENGINE[引擎层<br/>自定义分布式]
        C_COMM[通信层<br/>TCP + MFC]
        C_BACKEND[后端层<br/>自定义系统]
    end
    
    C_USER --> C_COORD
    C_COORD --> C_WORKERS  
    C_WORKERS --> C_ENGINE
    C_ENGINE --> C_COMM
    C_COMM --> C_BACKEND
    
    classDef complex fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    class C_USER,C_COORD,C_WORKERS,C_ENGINE,C_COMM,C_BACKEND complex
```

### 3. 通信模式

#### AReaLite通信
```mermaid
sequenceDiagram
    participant U as 用户脚本
    participant L as 启动器
    participant S as SGLang服务器
    participant F as FSDP引擎
    
    Note over U,F: 简化通信
    
    U->>L: 启动请求
    L->>S: 启动服务器
    U->>S: HTTP请求 (推理)
    U->>F: 直接调用 (训练)
    F->>F: FSDP2 all-reduce
    U->>S: 权重更新 (HTTP)
```

#### Core通信  
```mermaid
sequenceDiagram
    participant M as 主控Worker
    participant MW as 模型Worker
    participant RW as Rollout Worker
    participant GS as GServer管理器
    participant IS as 推理服务器
    
    Note over M,IS: 复杂Worker通信
    
    M->>MW: TCP命令
    M->>RW: TCP命令  
    M->>GS: TCP命令
    RW->>IS: 生成请求
    MW->>RW: TCP数据流
    GS->>IS: 管理命令
    MW->>MW: 自定义all-reduce
```

## 性能特征

### AReaLite性能特征
```mermaid
graph LR
    subgraph "AReaLite优势"
        AS1[快速设置<br/>几分钟启动]
        AS2[高推理吞吐量<br/>SGLang优化]
        AS3[内存高效<br/>FSDP2分片]
        AS4[简单调试<br/>单进程模型]
    end
    
    subgraph "AReaLite限制"
        AL1[规模上限<br/>~64 GPU限制]
        AL2[定制化有限<br/>固定架构]
        AL3[SGLang依赖<br/>推理后端锁定]
    end
    
    classDef strength fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef limitation fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class AS1,AS2,AS3,AS4 strength
    class AL1,AL2,AL3 limitation
```

### Core性能特征
```mermaid
graph LR
    subgraph "Core优势"
        CS1[大规模<br/>1000+ GPU]
        CS2[完全定制化<br/>每个组件]
        CS3[经过实战测试<br/>生产就绪]
        CS4[算法灵活性<br/>多种RL方法]
    end
    
    subgraph "Core限制"  
        CL1[复杂设置<br/>小时级配置]
        CL2[更高开销<br/>Worker通信]
        CL3[调试困难<br/>多进程复杂性]
        CL4[资源密集<br/>管理开销]
    end
    
    classDef strength fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef limitation fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class CS1,CS2,CS3,CS4 strength
    class CL1,CL2,CL3,CL4 limitation
```

## 使用场景建议

### 何时选择AReaLite

```mermaid
graph TD
    subgraph "AReaLite适用场景"
        AL_UC1[研究原型开发<br/>快速迭代]
        AL_UC2[中小规模<br/>1-64 GPU]
        AL_UC3[标准RLHF任务<br/>数学、编程、推理]
        AL_UC4[快速实验<br/>概念验证]
        AL_UC5[教育用途<br/>学习RLHF]
    end
    
    AL_UC1 --> AL_UC2
    AL_UC2 --> AL_UC3
    AL_UC3 --> AL_UC4
    AL_UC4 --> AL_UC5
    
    classDef arealite_use fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    class AL_UC1,AL_UC2,AL_UC3,AL_UC4,AL_UC5 arealite_use
```

### 何时选择Core

```mermaid
graph TD
    subgraph "Core适用场景"
        C_UC1[生产部署<br/>大规模训练]
        C_UC2[大规模<br/>100+ GPU]
        C_UC3[自定义算法<br/>新颖RL方法]
        C_UC4[工业应用<br/>高可靠性]
        C_UC5[高级定制<br/>专业需求]
    end
    
    C_UC1 --> C_UC2
    C_UC2 --> C_UC3
    C_UC3 --> C_UC4
    C_UC4 --> C_UC5
    
    classDef core_use fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    class C_UC1,C_UC2,C_UC3,C_UC4,C_UC5 core_use
```

## 迁移路径

```mermaid
flowchart LR
    subgraph "开发历程"
        START([开始项目])
        PROTOTYPE[AReaLite<br/>原型开发]
        EVALUATE{规模和复杂度<br/>需求评估?}
        PRODUCTION_LITE[AReaLite<br/>生产部署]
        MIGRATE[迁移至<br/>Core]
        PRODUCTION_CORE[Core<br/>生产部署]
    end
    
    START --> PROTOTYPE
    PROTOTYPE --> EVALUATE
    EVALUATE -->|小规模<br/>标准需求| PRODUCTION_LITE
    EVALUATE -->|大规模<br/>自定义需求| MIGRATE
    MIGRATE --> PRODUCTION_CORE
    
    classDef start fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef arealite fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef decision fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    
    class START start
    class PROTOTYPE,PRODUCTION_LITE arealite
    class MIGRATE,PRODUCTION_CORE core
    class EVALUATE decision
```

## 代码共享分析

两个系统在通用工具中共享大约25%的代码库：

### 共享组件
- **ReaLModel**: 基础模型实现
- **数学工具**: 奖励函数、验证逻辑  
- **统计跟踪**: 日志和监控系统
- **数据处理**: 分词和预处理工具

### 系统专用组件
- **AReaLite**: GRPO算法、FSDP引擎、SGLang集成
- **Core**: PPO接口、worker系统、MFC通信

这个架构对比帮助用户根据具体需求选择合适的系统，并理解简单性和可扩展性之间的权衡取舍。
