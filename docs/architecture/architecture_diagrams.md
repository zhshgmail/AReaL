```mermaid
graph TB
    subgraph "AReaLite Architecture (AI-Centric)"
        A1[Entry Point<br/>gsm8k_grpo.py] --> A2[Engine Objects]
        A2 --> A3[RemoteSGLangEngine]
        A2 --> A4[FSDPPPOActor]
        A3 --> A5[rollout_batch()]
        A4 --> A6[ppo_update()]
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

## 代码复杂度对比

```mermaid
graph LR
    subgraph "AReaLite: 3 Layer Abstraction"
        AL1[User Script] --> AL2[Engine Objects] --> AL3[Backend Implementation]
    end
    
    subgraph "Core: 6 Layer Abstraction"
        CL1[User Config] --> CL2[Experiment Framework] --> CL3[Master Scheduler]
        CL3 --> CL4[Worker Execution] --> CL5[MFC Calls] --> CL6[Backend Implementation]
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

## 训练流程对比

```mermaid
sequenceDiagram
    participant U as User
    participant AL as AReaLite
    participant E as Engine
    participant PT as PyTorch
    
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

```mermaid
sequenceDiagram
    participant U as User
    participant C as Config
    participant M as Master
    participant W as Workers
    participant MFC as MFC System
    participant PT as PyTorch
    
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