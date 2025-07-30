# AReaLite 执行流程和数据管道

本文档提供了数据在GRPO训练不同阶段如何流经AReaLite的详细视图，从初始化到权重更新。

## 完整训练循环流程

```mermaid
flowchart TD
    %% 初始化阶段
    subgraph "初始化阶段"
        START([开始训练])
        PARSE_CONFIG[解析配置<br/>GRPOConfig]
        INIT_DIST[初始化分布式<br/>torch.distributed]
        START_SERVERS[启动SGLang服务器<br/>启动器进程]
        INIT_ENGINES[初始化引擎<br/>Actor + Rollout]
        LOAD_DATA[加载数据集<br/>HuggingFace + DataLoader]
    end
    
    %% 主训练循环
    subgraph "主训练循环"
        LOOP_START([训练步骤开始])
        GET_BATCH[获取数据批次<br/>来自DataLoader]
        
        %% Rollout阶段
        subgraph "Rollout阶段"
            ROLLOUT_START[开始Rollout]
            SEND_PROMPTS[发送提示至<br/>SGLang服务器]
            GENERATE[生成回复<br/>每个提示多个回复] 
            COMPUTE_REWARDS[计算奖励<br/>任务特定函数]
            COLLECT_RESULTS[收集Rollout结果<br/>到TensorDict]
        end
        
        %% 训练阶段
        subgraph "训练阶段"
            COMPUTE_LOGP[计算对数概率<br/>Actor模型]
            COMPUTE_REF[计算参考LogP<br/>参考模型]
            COMPUTE_ADV[计算优势<br/>GAE算法]
            GRPO_UPDATE[GRPO更新<br/>策略 + 价值损失]
            UPDATE_LR[更新学习率<br/>调度器步骤]
        end
        
        %% 同步阶段
        subgraph "同步阶段"  
            PAUSE_ROLLOUT[暂停Rollout服务器]
            UPLOAD_WEIGHTS[上传新权重<br/>Actor → 服务器]
            WAIT_SYNC[等待同步<br/>所有服务器已更新]
            RESUME_ROLLOUT[恢复Rollout服务器]
            UPDATE_VERSION[更新模型版本<br/>一致性检查]
        end
        
        LOOP_END{Continue Training?}
        SAVE_CHECKPOINT[Save Checkpoint<br/>Optional]
    end
    
    FINISH([Training Complete])
    
    %% Flow Connections
    START --> PARSE_CONFIG
    PARSE_CONFIG --> INIT_DIST  
    INIT_DIST --> START_SERVERS
    START_SERVERS --> INIT_ENGINES
    INIT_ENGINES --> LOAD_DATA
    LOAD_DATA --> LOOP_START
    
    LOOP_START --> GET_BATCH
    GET_BATCH --> ROLLOUT_START
    
    ROLLOUT_START --> SEND_PROMPTS
    SEND_PROMPTS --> GENERATE
    GENERATE --> COMPUTE_REWARDS
    COMPUTE_REWARDS --> COLLECT_RESULTS
    
    COLLECT_RESULTS --> COMPUTE_LOGP
    COMPUTE_LOGP --> COMPUTE_REF
    COMPUTE_REF --> COMPUTE_ADV
    COMPUTE_ADV --> GRPO_UPDATE
    GRPO_UPDATE --> UPDATE_LR
    
    UPDATE_LR --> PAUSE_ROLLOUT
    PAUSE_ROLLOUT --> UPLOAD_WEIGHTS
    UPLOAD_WEIGHTS --> WAIT_SYNC
    WAIT_SYNC --> RESUME_ROLLOUT
    RESUME_ROLLOUT --> UPDATE_VERSION
    
    UPDATE_VERSION --> LOOP_END
    LOOP_END -->|Yes| SAVE_CHECKPOINT
    SAVE_CHECKPOINT --> LOOP_START
    LOOP_END -->|No| FINISH
    
    %% Styling
    classDef init fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef rollout fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef training fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef sync fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef decision fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    classDef endpoint fill:#e0f2f1,stroke:#00695c,stroke-width:3px
    
    class PARSE_CONFIG,INIT_DIST,START_SERVERS,INIT_ENGINES,LOAD_DATA init
    class ROLLOUT_START,SEND_PROMPTS,GENERATE,COMPUTE_REWARDS,COLLECT_RESULTS rollout
    class COMPUTE_LOGP,COMPUTE_REF,COMPUTE_ADV,GRPO_UPDATE,UPDATE_LR training
    class PAUSE_ROLLOUT,UPLOAD_WEIGHTS,WAIT_SYNC,RESUME_ROLLOUT,UPDATE_VERSION sync
    class LOOP_END decision  
    class START,FINISH endpoint
```

## 详细Rollout阶段流程

```mermaid
sequenceDiagram
    participant DS as DataLoader
    participant RE as RemoteSGLangEngine  
    participant WF as RLVRWorkflow
    participant S1 as SGLang服务器1
    participant S2 as SGLang服务器2
    participant SN as SGLang服务器N
    participant RF as 奖励函数
    
    Note over DS,RF: Rollout阶段 - 生成训练数据
    
    %% 从数据加载器获取批次
    DS->>RE: next_batch(prompts)
    
    %% 初始化工作流
    RE->>WF: arun_episode(prompts)
    
    %% 跨服务器并行生成
    par 服务器1生成
        WF->>S1: generate_request(prompt_1)
        S1-->>WF: responses_1
    and 服务器2生成  
        WF->>S2: generate_request(prompt_2)
        S2-->>WF: responses_2
    and 服务器N生成
        WF->>SN: generate_request(prompt_n)
        SN-->>WF: responses_n
    end
    
    %% 聚合响应
    WF->>WF: aggregate_responses()
    
    %% 为每个响应计算奖励
    loop 对每个提示-响应对
        WF->>RF: compute_reward(prompt, response, reference)
        RF-->>WF: reward_score
    end
    
    %% 格式化输出
    WF->>WF: format_tensordict()
    WF-->>RE: rollout_results
    
    %% 返回主训练循环
    RE-->>DS: training_batch
    
    Note over DS,RF: Rollout完成 - 准备训练
```

## GRPO Training Phase Details

```mermaid
flowchart TD
    %% Input Data
    subgraph "Input Data"
        BATCH[Training Batch<br/>Prompts + Responses + Rewards]
        INPUT_IDS[Input Token IDs<br/>Tokenized Sequences]
        ATTENTION_MASK[Attention Masks<br/>Valid Token Positions]
        REWARD_SCORES[Reward Scores<br/>Task Performance]
    end
    
    %% Log Probability Computation
    subgraph "Log Probability Phase"
        ACTOR_FORWARD[Actor Forward Pass<br/>Compute Logits]
        ACTOR_LOGP[Actor Log Probabilities<br/>Current Policy]
        REF_FORWARD[Reference Forward Pass<br/>Frozen Model]
        REF_LOGP[Reference Log Probabilities<br/>Initial Policy]
        PROX_LOGP[Proximal Log Probabilities<br/>Recomputed if needed]
    end
    
    %% Advantage Computation
    subgraph "Advantage Computation"
        KL_REWARDS[KL Regularized Rewards<br/>reward - β * KL(π||π_ref)]
        VALUE_EST[Value Estimation<br/>Critic Network]
        GAE_COMPUTE[GAE Computation<br/>Generalized Advantage Estimation]
        ADV_NORM[Advantage Normalization<br/>Group-wise if configured]
    end
    
    %% GRPO Loss Computation
    subgraph "GRPO Loss Computation"
        RATIO_COMPUTE[Importance Ratio<br/>exp(log_π - log_π_old)]
        CLIP_RATIO[Clipped Ratio<br/>clip(ratio, 1-ε, 1+ε)]
        POLICY_LOSS[Policy Loss<br/>-min(ratio*A, clipped_ratio*A)]
        VALUE_LOSS[Value Loss<br/>MSE(V, returns)]
        TOTAL_LOSS[Total Loss<br/>policy_loss + value_coeff * value_loss]
    end
    
    %% Parameter Update
    subgraph "Parameter Update"
        BACKWARD[Backward Pass<br/>Compute Gradients]
        GRAD_CLIP[Gradient Clipping<br/>Prevent Instability]
        OPTIMIZER_STEP[Optimizer Step<br/>Apply Parameter Updates]
        FSDP_SYNC[FSDP Synchronization<br/>All-Reduce Gradients]
    end
    
    %% Data Flow
    BATCH --> INPUT_IDS
    BATCH --> ATTENTION_MASK  
    BATCH --> REWARD_SCORES
    
    INPUT_IDS --> ACTOR_FORWARD
    INPUT_IDS --> REF_FORWARD
    ACTOR_FORWARD --> ACTOR_LOGP
    REF_FORWARD --> REF_LOGP
    ACTOR_LOGP --> PROX_LOGP
    
    REWARD_SCORES --> KL_REWARDS
    ACTOR_LOGP --> KL_REWARDS
    REF_LOGP --> KL_REWARDS
    
    KL_REWARDS --> VALUE_EST
    VALUE_EST --> GAE_COMPUTE
    GAE_COMPUTE --> ADV_NORM
    
    ADV_NORM --> RATIO_COMPUTE
    ACTOR_LOGP --> RATIO_COMPUTE
    PROX_LOGP --> RATIO_COMPUTE
    RATIO_COMPUTE --> CLIP_RATIO
    
    CLIP_RATIO --> POLICY_LOSS
    ADV_NORM --> POLICY_LOSS
    VALUE_EST --> VALUE_LOSS
    POLICY_LOSS --> TOTAL_LOSS
    VALUE_LOSS --> TOTAL_LOSS
    
    TOTAL_LOSS --> BACKWARD
    BACKWARD --> GRAD_CLIP
    GRAD_CLIP --> OPTIMIZER_STEP
    OPTIMIZER_STEP --> FSDP_SYNC
    
    %% Styling
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef logp fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef advantage fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef loss fill:#fce4ec,stroke:#ad1457,stroke-width:2px
    classDef update fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    
    class BATCH,INPUT_IDS,ATTENTION_MASK,REWARD_SCORES input
    class ACTOR_FORWARD,ACTOR_LOGP,REF_FORWARD,REF_LOGP,PROX_LOGP logp
    class KL_REWARDS,VALUE_EST,GAE_COMPUTE,ADV_NORM advantage
    class RATIO_COMPUTE,CLIP_RATIO,POLICY_LOSS,VALUE_LOSS,TOTAL_LOSS loss
    class BACKWARD,GRAD_CLIP,OPTIMIZER_STEP,FSDP_SYNC update
```

## Weight Synchronization Flow

```mermaid
sequenceDiagram
    participant T as Training Process
    participant A as Actor Engine
    participant RE as RemoteSGLangEngine
    participant S1 as SGLang Server 1
    participant S2 as SGLang Server 2  
    participant SN as SGLang Server N
    participant FS as Filesystem/Storage
    
    Note over T,FS: Weight Synchronization Phase
    
    %% Pause inference
    T->>RE: pause()
    RE->>S1: pause_generation()
    RE->>S2: pause_generation()
    RE->>SN: pause_generation()
    
    par Wait for pause confirmation
        S1-->>RE: paused
    and
        S2-->>RE: paused  
    and
        SN-->>RE: paused
    end
    
    RE-->>T: all_servers_paused
    
    %% Prepare weight update
    T->>A: upload_weights(meta)
    A->>A: serialize_state_dict()
    A->>FS: save_weights(path)
    
    %% Notify servers of weight update
    Note right of T: Only rank 0 sends update notifications
    alt rank == 0
        T->>RE: update_weights(meta) 
        par Send update requests
            RE->>S1: update_weights_request(meta)
        and
            RE->>S2: update_weights_request(meta)
        and  
            RE->>SN: update_weights_request(meta)
        end
    end
    
    %% Servers load new weights
    par Load weights on servers
        S1->>FS: load_weights(path)
        FS-->>S1: weight_data
        S1->>S1: hot_swap_model()
    and
        S2->>FS: load_weights(path)  
        FS-->>S2: weight_data
        S2->>S2: hot_swap_model()
    and
        SN->>FS: load_weights(path)
        FS-->>SN: weight_data
        SN->>SN: hot_swap_model()
    end
    
    %% Confirm weight updates
    par Confirm updates
        S1-->>RE: update_complete
    and
        S2-->>RE: update_complete
    and
        SN-->>RE: update_complete  
    end
    
    alt rank == 0
        RE-->>T: all_updates_complete
    end
    
    %% Synchronize across training processes
    Note over T,FS: Distributed barrier synchronization
    T->>T: dist.barrier()
    T->>T: torch.cuda.synchronize()
    
    %% Resume inference
    T->>RE: resume()
    RE->>S1: resume_generation()
    RE->>S2: resume_generation()
    RE->>SN: resume_generation()
    
    par Resume confirmations
        S1-->>RE: resumed
    and
        S2-->>RE: resumed
    and
        SN-->>RE: resumed
    end
    
    %% Update version tracking
    T->>A: set_version(new_version)
    T->>RE: set_version(new_version)
    
    Note over T,FS: Synchronization Complete - Ready for Next Step
```

## Memory Usage Patterns

```mermaid
graph TB
    subgraph "Memory Usage During Training"
        subgraph "Forward Pass"
            F_ACT[Activations<br/>~2GB per layer]
            F_GRAD[Gradient Buffers<br/>~1GB per layer]  
            F_OPT[Optimizer States<br/>~2GB per layer]
        end
        
        subgraph "Backward Pass"
            B_GRAD[Accumulated Gradients<br/>Peak Usage]
            B_ACT[Cached Activations<br/>For Checkpointing]
            B_TEMP[Temporary Buffers<br/>FSDP Communication]
        end
        
        subgraph "FSDP Sharding"
            SHARD_P[Parameter Shards<br/>1/N of total]
            SHARD_G[Gradient Shards<br/>1/N of total]
            SHARD_O[Optimizer Shards<br/>1/N of total]
        end
        
        subgraph "Peak Memory Events"
            ALL_GATHER[All-Gather Peak<br/>Full Parameter Access]
            REDUCE_SCATTER[Reduce-Scatter Peak<br/>Gradient Aggregation]
            OPTIM_PEAK[Optimizer Peak<br/>Parameter Updates]
        end
    end
    
    %% Memory relationships
    F_ACT --> B_ACT
    F_GRAD --> B_GRAD
    B_GRAD --> REDUCE_SCATTER
    
    SHARD_P --> ALL_GATHER
    ALL_GATHER --> F_ACT
    
    SHARD_G --> REDUCE_SCATTER
    REDUCE_SCATTER --> SHARD_O
    SHARD_O --> OPTIM_PEAK
    
    classDef forward fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef backward fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px  
    classDef shard fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef peak fill:#ffebee,stroke:#c62828,stroke-width:3px
    
    class F_ACT,F_GRAD,F_OPT forward
    class B_GRAD,B_ACT,B_TEMP backward
    class SHARD_P,SHARD_G,SHARD_O shard
    class ALL_GATHER,REDUCE_SCATTER,OPTIM_PEAK peak
```

## Performance Bottleneck Analysis

```mermaid
graph LR
    subgraph "Potential Bottlenecks"
        subgraph "Inference Bottlenecks"
            I1[Server Capacity<br/>Limited Concurrent Requests]
            I2[Load Balancing<br/>Uneven Request Distribution]  
            I3[Generation Speed<br/>Model Size vs GPU Memory]
            I4[Network Latency<br/>Client-Server Communication]
        end
        
        subgraph "Training Bottlenecks"
            T1[FSDP Communication<br/>All-Gather/Reduce-Scatter]
            T2[Memory Bandwidth<br/>Large Model Loading]
            T3[Gradient Computation<br/>Backward Pass Complexity]
            T4[Optimizer Updates<br/>Parameter Synchronization]
        end
        
        subgraph "Synchronization Bottlenecks"
            S1[Weight Upload<br/>File I/O Performance]
            S2[Hot Swap Delay<br/>Model Reloading Time]
            S3[Distributed Barrier<br/>Process Coordination]
            S4[Version Consistency<br/>State Synchronization]
        end
        
        subgraph "Data Bottlenecks"
            D1[DataLoader Speed<br/>Preprocessing Overhead]
            D2[Tokenization<br/>CPU-bound Operations]
            D3[Reward Computation<br/>Custom Function Complexity]
            D4[Batch Aggregation<br/>Memory Copy Operations]
        end
    end
    
    %% Mitigation Strategies (shown as solutions)
    I1 -.->|Scale Servers| SCALE1[Horizontal Scaling]
    I2 -.->|Improve LB| SCALE2[Smart Load Balancing]
    T1 -.->|Optimize Comm| SCALE3[Communication Optimization]
    S1 -.->|Async Upload| SCALE4[Asynchronous Operations]
    D1 -.->|Parallel Load| SCALE5[Multi-process Loading]
    
    classDef bottleneck fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef solution fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class I1,I2,I3,I4,T1,T2,T3,T4,S1,S2,S3,S4,D1,D2,D3,D4 bottleneck
    class SCALE1,SCALE2,SCALE3,SCALE4,SCALE5 solution
```

This execution flow documentation provides developers and researchers with detailed insights into how AReaLite orchestrates complex distributed GRPO training, enabling better optimization and debugging of the system.
