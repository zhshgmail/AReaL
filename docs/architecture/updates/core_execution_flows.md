# Core执行流程和系统管道

本文档提供AReaL Core系统执行流程的详细视图，展示其复杂的分布式训练管道和系统协调机制。

## 完整分布式训练流程

```mermaid
flowchart TD
    subgraph "Core分布式训练全流程"
        %% 系统初始化阶段
        subgraph "系统初始化阶段"
            START([启动Core系统])
            PARSE_CONFIG[解析实验配置<br/>ExperimentConfig]
            BUILD_DFG[构建数据流图<br/>DataFlowGraph]
            INIT_CLUSTER[初始化集群<br/>Ray/Slurm启动]
            START_MASTER[启动Master Worker<br/>全局协调器]
            DEPLOY_WORKERS[部署Worker节点<br/>分布式部署]
        end
        
        %% Worker初始化阶段
        subgraph "Worker初始化阶段"
            INIT_MODEL_W[初始化Model Workers<br/>训练节点池]
            INIT_ROLLOUT_W[初始化Rollout Workers<br/>数据收集池]
            INIT_GEN_MGR[初始化Generation Manager<br/>推理管理器]
            START_GEN_SERVERS[启动Generation Servers<br/>推理服务集群]
            SETUP_COMM[建立通信连接<br/>RPC/ZMQ/TCP]
        end
        
        %% 主训练循环
        subgraph "主训练循环"
            LOOP_START([训练循环开始])
            SCHEDULE_TASKS[任务调度<br/>Master调度]
            
            %% 数据收集阶段
            subgraph "数据收集阶段"
                EXECUTE_ROLLOUT[执行Rollout MFC<br/>数据收集]
                GEN_REQUESTS[生成请求分发<br/>负载均衡]
                PARALLEL_GEN[并行文本生成<br/>多服务器]
                COMPUTE_REWARDS[计算奖励分数<br/>质量评估]
                COLLECT_DATA[收集训练数据<br/>格式化存储]
            end
            
            %% 训练更新阶段
            subgraph "训练更新阶段"
                EXECUTE_TRAIN[执行训练MFC<br/>模型更新]
                COMPUTE_LOSS[计算PPO损失<br/>策略优化]
                BACKWARD_PASS[反向传播<br/>梯度计算]
                SYNC_GRADIENTS[梯度同步<br/>All-Reduce]
                UPDATE_PARAMS[参数更新<br/>优化器步骤]
            end
            
            %% 模型同步阶段
            subgraph "模型同步阶段"
                EXTRACT_WEIGHTS[提取模型权重<br/>训练节点]
                BROADCAST_WEIGHTS[广播权重更新<br/>分发到推理节点]
                UPDATE_GEN_MODELS[更新生成模型<br/>热更新权重]
                VERIFY_SYNC[验证同步完成<br/>版本一致性]
            end
        end
        
        %% 监控和调试
        subgraph "监控和调试"
            HEALTH_CHECK[健康检查<br/>节点状态监控]
            PERF_MONITOR[性能监控<br/>指标收集]
            LOG_COLLECT[日志收集<br/>分布式日志]
            ALERT_SYSTEM[告警系统<br/>异常处理]
        end
    end
    
    %% 流程连接
    START --> PARSE_CONFIG
    PARSE_CONFIG --> BUILD_DFG
    BUILD_DFG --> INIT_CLUSTER
    INIT_CLUSTER --> START_MASTER
    START_MASTER --> DEPLOY_WORKERS
    
    DEPLOY_WORKERS --> INIT_MODEL_W
    DEPLOY_WORKERS --> INIT_ROLLOUT_W
    DEPLOY_WORKERS --> INIT_GEN_MGR
    INIT_GEN_MGR --> START_GEN_SERVERS
    START_GEN_SERVERS --> SETUP_COMM
    
    SETUP_COMM --> LOOP_START
    LOOP_START --> SCHEDULE_TASKS
    SCHEDULE_TASKS --> EXECUTE_ROLLOUT
    
    EXECUTE_ROLLOUT --> GEN_REQUESTS
    GEN_REQUESTS --> PARALLEL_GEN
    PARALLEL_GEN --> COMPUTE_REWARDS
    COMPUTE_REWARDS --> COLLECT_DATA
    
    COLLECT_DATA --> EXECUTE_TRAIN
    EXECUTE_TRAIN --> COMPUTE_LOSS
    COMPUTE_LOSS --> BACKWARD_PASS
    BACKWARD_PASS --> SYNC_GRADIENTS
    SYNC_GRADIENTS --> UPDATE_PARAMS
    
    UPDATE_PARAMS --> EXTRACT_WEIGHTS
    EXTRACT_WEIGHTS --> BROADCAST_WEIGHTS
    BROADCAST_WEIGHTS --> UPDATE_GEN_MODELS
    UPDATE_GEN_MODELS --> VERIFY_SYNC
    
    VERIFY_SYNC --> LOOP_START
    
    %% 监控连接
    SCHEDULE_TASKS --> HEALTH_CHECK
    EXECUTE_ROLLOUT --> PERF_MONITOR
    EXECUTE_TRAIN --> LOG_COLLECT
    VERIFY_SYNC --> ALERT_SYSTEM
    
    %% 样式定义
    classDef init fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef worker_init fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef rollout fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef train fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef sync fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef monitor fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    classDef control fill:#fff8e1,stroke:#fbc02d,stroke-width:3px
    
    class START,PARSE_CONFIG,BUILD_DFG,INIT_CLUSTER,START_MASTER,DEPLOY_WORKERS init
    class INIT_MODEL_W,INIT_ROLLOUT_W,INIT_GEN_MGR,START_GEN_SERVERS,SETUP_COMM worker_init
    class EXECUTE_ROLLOUT,GEN_REQUESTS,PARALLEL_GEN,COMPUTE_REWARDS,COLLECT_DATA rollout
    class EXECUTE_TRAIN,COMPUTE_LOSS,BACKWARD_PASS,SYNC_GRADIENTS,UPDATE_PARAMS train
    class EXTRACT_WEIGHTS,BROADCAST_WEIGHTS,UPDATE_GEN_MODELS,VERIFY_SYNC sync
    class HEALTH_CHECK,PERF_MONITOR,LOG_COLLECT,ALERT_SYSTEM monitor
    class LOOP_START,SCHEDULE_TASKS control
```

## 同步PPO详细执行序列

```mermaid
sequenceDiagram
    participant C as 配置管理器
    participant M as Master Worker
    participant MW as Model Workers
    participant RW as Rollout Workers
    participant GS as Generation Servers
    participant DM as Data Manager
    participant FE as Function Executor
    
    Note over C,FE: Core同步PPO完整执行流程
    
    %% 初始化阶段
    C->>M: 加载实验配置
    M->>M: 构建数据流图(DFG)
    M->>MW: 启动训练节点集群
    M->>RW: 启动数据收集集群
    M->>GS: 启动推理服务集群
    M->>DM: 初始化数据管理系统
    M->>FE: 启动函数执行引擎
    
    Note over M,FE: 系统初始化完成，开始训练循环
    
    loop 同步训练循环
        %% 任务调度阶段
        M->>FE: 执行数据流图
        FE->>DM: 检查数据依赖关系
        DM-->>FE: 数据准备就绪
        
        %% Rollout执行阶段
        FE->>RW: 执行generate_MFC
        RW->>GS: 分发生成请求
        
        par 并行生成处理
            GS->>GS: 服务器1并行生成
        and
            GS->>GS: 服务器2并行生成
        and
            GS->>GS: 服务器N并行生成
        end
        
        GS-->>RW: 聚合生成结果
        RW->>RW: 计算奖励分数
        RW->>RW: 格式化rollout数据
        RW-->>FE: 返回rollout结果
        
        %% 数据传输阶段
        FE->>DM: 重分布rollout数据
        DM->>MW: 传输训练数据到模型节点
        
        %% 训练执行阶段
        FE->>MW: 执行train_step_MFC
        MW->>MW: 计算PPO损失函数
        MW->>MW: 执行反向传播
        
        par 分布式梯度同步
            MW->>MW: All-Reduce梯度聚合
        and
            MW->>MW: 参数更新同步
        end
        
        MW-->>FE: 返回训练统计信息
        
        %% 参考模型计算阶段
        FE->>MW: 执行compute_ref_MFC
        MW->>MW: 计算参考模型logprobs
        MW-->>FE: 返回参考数据
        
        %% 权重同步阶段
        FE-->>M: 完成一轮训练
        M->>MW: 提取最新模型权重
        MW-->>M: 上传权重参数
        M->>GS: 同步权重到生成服务器
        
        par 并行权重更新
            GS->>GS: 服务器1权重更新
        and
            GS->>GS: 服务器2权重更新
        and
            GS->>GS: 服务器N权重更新
        end
        
        GS-->>M: 权重同步完成确认
        M->>M: 更新全局训练状态
        
        Note over M,FE: 一轮同步训练完成
    end
```

## 异步PPO详细执行流程

```mermaid
sequenceDiagram
    participant M as Master Worker
    participant MW1 as Model Worker 1
    participant MW2 as Model Worker 2
    participant RW1 as Rollout Worker 1
    participant RW2 as Rollout Worker 2
    participant GM as Generation Manager
    participant GS1 as Gen Server 1
    participant GS2 as Gen Server 2
    participant B as Buffer System
    
    Note over M,B: Core异步PPO分布式执行流程
    
    %% 系统启动
    M->>MW1: 启动训练节点1
    M->>MW2: 启动训练节点2
    M->>RW1: 启动Rollout节点1
    M->>RW2: 启动Rollout节点2
    M->>GM: 启动生成管理器
    GM->>GS1: 管理生成服务器1
    GM->>GS2: 管理生成服务器2
    M->>B: 初始化分布式缓冲系统
    
    Note over M,B: 异步执行开始，三个并行流程
    
    par 异步Rollout流程1
        loop Rollout Worker 1持续数据收集
            RW1->>GM: 请求可用生成服务器
            GM-->>RW1: 分配GS1服务器
            RW1->>GS1: 提交生成请求批次
            GS1->>GS1: 异步批量文本生成
            GS1-->>RW1: 返回生成结果
            RW1->>RW1: 计算奖励和优势值
            RW1->>B: 存储训练数据到缓冲区
        end
    and 异步Rollout流程2
        loop Rollout Worker 2持续数据收集
            RW2->>GM: 请求可用生成服务器
            GM-->>RW2: 分配GS2服务器
            RW2->>GS2: 提交生成请求批次
            GS2->>GS2: 异步批量文本生成
            GS2-->>RW2: 返回生成结果
            RW2->>RW2: 计算奖励和优势值
            RW2->>B: 存储训练数据到缓冲区
        end
    and 异步训练流程1
        loop Model Worker 1持续模型更新
            MW1->>B: 从缓冲区采样训练批次
            B-->>MW1: 返回训练数据批次
            MW1->>MW1: 计算PPO损失函数
            MW1->>MW1: 执行梯度计算
            MW1->>MW1: 应用参数更新
            MW1->>GM: 上传新的模型权重
            GM->>GS1: 推送权重到生成服务器1
            GS1->>GS1: 热更新模型权重
        end
    and 异步训练流程2
        loop Model Worker 2持续模型更新
            MW2->>B: 从缓冲区采样训练批次
            B-->>MW2: 返回训练数据批次
            MW2->>MW2: 计算PPO损失函数
            MW2->>MW2: 执行梯度计算
            MW2->>MW2: 应用参数更新
            MW2->>GM: 上传新的模型权重
            GM->>GS2: 推送权重到生成服务器2
            GS2->>GS2: 热更新模型权重
        end
    and 系统协调监控流程
        loop Master Worker系统监控
            M->>B: 监控缓冲区状态和数据流
            M->>MW1: 监控训练进度和性能
            M->>MW2: 监控训练进度和性能
            M->>RW1: 监控数据收集速率
            M->>RW2: 监控数据收集速率
            M->>GM: 监控生成服务器健康状态
            M->>M: 动态调整系统参数
            M->>M: 负载均衡和故障处理
        end
    end
    
    Note over M,B: 异步训练持续进行，直到收敛
```

## MFC系统执行深度解析

### MFC调用完整生命周期

```mermaid
flowchart TD
    subgraph "MFC执行生命周期"
        %% MFC定义阶段
        subgraph "MFC定义阶段"
            DEFINE_MFC[定义MFC<br/>MFCDef创建]
            VALIDATE_DEF[验证定义<br/>输入输出规范]
            REGISTER_MFC[注册MFC<br/>添加到注册表]
        end
        
        %% MFC调度阶段
        subgraph "MFC调度阶段"
            SCHEDULE_REQ[调度请求<br/>函数执行器]
            RESOLVE_DEPS[解析依赖<br/>数据依赖关系]
            SELECT_WORKER[选择Worker<br/>负载均衡]
        end
        
        %% MFC执行阶段
        subgraph "MFC执行阶段"
            PREPARE_DATA[准备数据<br/>输入格式化]
            SEND_REQUEST[发送请求<br/>网络传输]
            WORKER_EXEC[Worker执行<br/>模型计算]
            COLLECT_RESULT[收集结果<br/>输出聚合]
        end
        
        %% MFC完成阶段
        subgraph "MFC完成阶段"
            VALIDATE_OUTPUT[验证输出<br/>结果检查]
            UPDATE_BUFFER[更新缓冲区<br/>数据存储]
            NOTIFY_DEPS[通知依赖<br/>下游MFC]
            LOG_STATS[记录统计<br/>性能指标]
        end
    end
    
    DEFINE_MFC --> VALIDATE_DEF
    VALIDATE_DEF --> REGISTER_MFC
    REGISTER_MFC --> SCHEDULE_REQ
    
    SCHEDULE_REQ --> RESOLVE_DEPS
    RESOLVE_DEPS --> SELECT_WORKER
    SELECT_WORKER --> PREPARE_DATA
    
    PREPARE_DATA --> SEND_REQUEST
    SEND_REQUEST --> WORKER_EXEC
    WORKER_EXEC --> COLLECT_RESULT
    
    COLLECT_RESULT --> VALIDATE_OUTPUT
    VALIDATE_OUTPUT --> UPDATE_BUFFER
    UPDATE_BUFFER --> NOTIFY_DEPS
    NOTIFY_DEPS --> LOG_STATS
    
    classDef define fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef schedule fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef execute fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef complete fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class DEFINE_MFC,VALIDATE_DEF,REGISTER_MFC define
    class SCHEDULE_REQ,RESOLVE_DEPS,SELECT_WORKER schedule
    class PREPARE_DATA,SEND_REQUEST,WORKER_EXEC,COLLECT_RESULT execute
    class VALIDATE_OUTPUT,UPDATE_BUFFER,NOTIFY_DEPS,LOG_STATS complete
```

### MFC并行执行协调

```mermaid
sequenceDiagram
    participant FE as Function Executor
    participant DM as Data Manager
    participant MW1 as Model Worker 1
    participant MW2 as Model Worker 2
    participant MW3 as Model Worker 3
    participant BM as Buffer Manager
    
    Note over FE,BM: MFC并行执行协调流程
    
    %% MFC准备阶段
    FE->>DM: 分析数据依赖关系
    DM->>DM: 创建数据重分布计划
    FE->>BM: 分配执行缓冲区
    BM-->>FE: 缓冲区分配完成
    
    %% 并行MFC调度
    FE->>MW1: MFC_REQUEST(train_step, batch_1)
    FE->>MW2: MFC_REQUEST(train_step, batch_2)
    FE->>MW3: MFC_REQUEST(train_step, batch_3)
    
    Note over MW1,MW3: 并行执行训练步骤
    
    par Model Worker 1执行
        MW1->>MW1: 加载模型接口
        MW1->>MW1: 执行前向传播
        MW1->>MW1: 计算损失函数
        MW1->>MW1: 执行反向传播
        MW1->>MW1: 计算梯度
        MW1-->>FE: 返回执行结果1
    and Model Worker 2执行
        MW2->>MW2: 加载模型接口
        MW2->>MW2: 执行前向传播
        MW2->>MW2: 计算损失函数
        MW2->>MW2: 执行反向传播
        MW2->>MW2: 计算梯度
        MW2-->>FE: 返回执行结果2
    and Model Worker 3执行
        MW3->>MW3: 加载模型接口
        MW3->>MW3: 执行前向传播
        MW3->>MW3: 计算损失函数
        MW3->>MW3: 执行反向传播
        MW3->>MW3: 计算梯度
        MW3-->>FE: 返回执行结果3
    end
    
    %% 结果聚合阶段
    FE->>BM: 收集所有执行结果
    BM->>BM: 聚合梯度和统计信息
    BM-->>FE: 返回聚合结果
    
    %% 同步更新阶段
    FE->>MW1: SYNC_REQUEST(aggregated_gradients)
    FE->>MW2: SYNC_REQUEST(aggregated_gradients)
    FE->>MW3: SYNC_REQUEST(aggregated_gradients)
    
    par 参数同步更新
        MW1->>MW1: 应用聚合梯度
        MW1-->>FE: 同步完成1
    and
        MW2->>MW2: 应用聚合梯度
        MW2-->>FE: 同步完成2
    and
        MW3->>MW3: 应用聚合梯度
        MW3-->>FE: 同步完成3
    end
    
    FE->>FE: 更新全局训练状态
    
    Note over FE,BM: 并行MFC执行完成
```

## 分布式数据管道详解

### 数据重分布系统

```mermaid
graph TB
    subgraph "Core数据重分布架构"
        %% 数据源层
        subgraph "数据源层"
            DATASET[原始数据集<br/>HuggingFace/Custom]
            STREAM[流式数据<br/>实时数据源]
            CACHE[缓存数据<br/>预处理结果]
        end
        
        %% 数据分析层
        subgraph "数据分析层"
            ANALYZER[数据分析器<br/>统计特征分析]
            PROFILER[性能分析器<br/>访问模式分析]
            OPTIMIZER[优化器<br/>分布策略优化]
        end
        
        %% 分布规划层
        subgraph "分布规划层"
            PLANNER[重分布规划器<br/>分布方案生成]
            SCHEDULER[调度器<br/>传输任务调度]
            COORDINATOR[协调器<br/>节点间协调]
        end
        
        %% 数据传输层
        subgraph "数据传输层"
            SENDER[数据发送器<br/>并行传输]
            RECEIVER[数据接收器<br/>并行接收]
            MONITOR[传输监控<br/>进度跟踪]
        end
        
        %% 数据存储层
        subgraph "数据存储层"
            LOCAL_BUFFER[本地缓冲区<br/>Worker本地存储]
            SHARED_BUFFER[共享缓冲区<br/>跨节点共享]
            PERSISTENT[持久化存储<br/>检查点保存]
        end
    end
    
    DATASET --> ANALYZER
    STREAM --> PROFILER
    CACHE --> OPTIMIZER
    
    ANALYZER --> PLANNER
    PROFILER --> SCHEDULER
    OPTIMIZER --> COORDINATOR
    
    PLANNER --> SENDER
    SCHEDULER --> RECEIVER
    COORDINATOR --> MONITOR
    
    SENDER --> LOCAL_BUFFER
    RECEIVER --> SHARED_BUFFER
    MONITOR --> PERSISTENT
    
    classDef source fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef analysis fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef planning fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef transport fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef storage fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class DATASET,STREAM,CACHE source
    class ANALYZER,PROFILER,OPTIMIZER analysis
    class PLANNER,SCHEDULER,COORDINATOR planning
    class SENDER,RECEIVER,MONITOR transport
    class LOCAL_BUFFER,SHARED_BUFFER,PERSISTENT storage
```

### 数据传输优化流程

```mermaid
sequenceDiagram
    participant S as 数据源
    participant A as 分析器
    participant P as 规划器
    participant T as 传输器
    participant W1 as Worker 1
    participant W2 as Worker 2
    participant W3 as Worker 3
    
    Note over S,W3: 数据重分布优化流程
    
    %% 数据分析阶段
    S->>A: 提供数据集信息
    A->>A: 分析数据分布特征
    A->>A: 统计访问模式
    A->>A: 计算传输成本
    A-->>P: 数据分析报告
    
    %% 分布规划阶段
    P->>P: 生成候选分布方案
    P->>P: 评估方案性能
    P->>P: 选择最优分布策略
    P-->>T: 传输执行计划
    
    %% 数据传输阶段
    T->>S: 请求数据分片
    S-->>T: 返回数据分片
    
    par 并行数据传输
        T->>W1: 传输分片1
        W1->>W1: 本地存储分片1
        W1-->>T: 确认接收完成
    and
        T->>W2: 传输分片2
        W2->>W2: 本地存储分片2
        W2-->>T: 确认接收完成
    and
        T->>W3: 传输分片3
        W3->>W3: 本地存储分片3
        W3-->>T: 确认接收完成
    end
    
    T->>T: 验证传输完整性
    T->>A: 报告传输统计
    A->>P: 更新性能模型
    
    Note over S,W3: 数据重分布完成
```

## 容错与故障恢复机制

### 多层容错架构

```mermaid
graph TB
    subgraph "Core容错体系"
        %% 检测层
        subgraph "故障检测层"
            HEARTBEAT[心跳检测<br/>节点存活监控]
            HEALTH_CHECK[健康检查<br/>服务状态检测]
            PERF_MONITOR[性能监控<br/>异常行为检测]
        end
        
        %% 决策层
        subgraph "故障决策层"
            FAULT_DETECTOR[故障检测器<br/>异常识别和分类]
            RECOVERY_PLANNER[恢复规划器<br/>恢复策略制定]
            RESOURCE_MANAGER[资源管理器<br/>资源重新分配]
        end
        
        %% 执行层
        subgraph "故障恢复层"
            NODE_REPLACER[节点替换<br/>故障节点替换]
            DATA_RESTORER[数据恢复<br/>数据重建和同步]
            STATE_MIGRATOR[状态迁移<br/>计算状态转移]
        end
        
        %% 验证层
        subgraph "恢复验证层"
            INTEGRITY_CHECKER[完整性检查<br/>数据一致性验证]
            PERFORMANCE_VALIDATOR[性能验证<br/>恢复后性能检查]
            SYSTEM_TESTER[系统测试<br/>端到端功能测试]
        end
    end
    
    HEARTBEAT --> FAULT_DETECTOR
    HEALTH_CHECK --> RECOVERY_PLANNER
    PERF_MONITOR --> RESOURCE_MANAGER
    
    FAULT_DETECTOR --> NODE_REPLACER
    RECOVERY_PLANNER --> DATA_RESTORER
    RESOURCE_MANAGER --> STATE_MIGRATOR
    
    NODE_REPLACER --> INTEGRITY_CHECKER
    DATA_RESTORER --> PERFORMANCE_VALIDATOR
    STATE_MIGRATOR --> SYSTEM_TESTER
    
    classDef detect fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef decide fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef recover fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef validate fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class HEARTBEAT,HEALTH_CHECK,PERF_MONITOR detect
    class FAULT_DETECTOR,RECOVERY_PLANNER,RESOURCE_MANAGER decide
    class NODE_REPLACER,DATA_RESTORER,STATE_MIGRATOR recover
    class INTEGRITY_CHECKER,PERFORMANCE_VALIDATOR,SYSTEM_TESTER validate
```

### 故障恢复执行流程

```mermaid
sequenceDiagram
    participant M as Master Worker
    participant FD as 故障检测器
    participant RP as 恢复规划器
    participant RM as 资源管理器
    participant F as 故障节点
    participant R as 替换节点
    participant D as 数据管理器
    
    Note over M,D: Core故障恢复完整流程
    
    %% 故障检测阶段
    M->>FD: 启动故障监控
    FD->>F: 发送心跳检测
    F-->>FD: 心跳超时/异常响应
    FD->>FD: 确认节点故障
    FD->>M: 报告故障节点
    
    %% 恢复规划阶段
    M->>RP: 请求恢复计划
    RP->>RP: 分析故障影响范围
    RP->>RP: 评估恢复策略选项
    RP->>RP: 制定最优恢复方案
    RP-->>M: 返回恢复计划
    
    %% 资源重分配阶段
    M->>RM: 执行资源重分配
    RM->>RM: 搜索可用替换节点
    RM->>R: 分配新的计算节点
    RM->>D: 准备数据迁移
    RM-->>M: 资源分配完成
    
    %% 状态恢复阶段
    M->>R: 启动替换节点
    R->>R: 初始化节点环境
    M->>D: 开始数据恢复
    
    par 并行恢复操作
        D->>R: 传输检查点数据
        R->>R: 恢复计算状态
    and
        D->>D: 验证数据完整性
        D->>R: 同步最新数据
    end
    
    R-->>M: 节点恢复完成
    D-->>M: 数据恢复完成
    
    %% 验证阶段
    M->>R: 执行健康检查
    R-->>M: 健康检查通过
    M->>M: 更新节点注册表
    M->>M: 恢复训练流程
    
    Note over M,D: 故障恢复完成，系统正常运行
```

## 性能调优与监控

### 性能瓶颈分析

```mermaid
graph LR
    subgraph "Core性能瓶颈识别"
        subgraph "计算瓶颈"
            COMP_CPU[CPU利用率<br/>计算密集度]
            COMP_GPU[GPU利用率<br/>并行效率]
            COMP_MEM[内存带宽<br/>数据访问]
        end
        
        subgraph "通信瓶颈"
            COMM_NET[网络带宽<br/>数据传输]
            COMM_LAT[通信延迟<br/>同步等待]
            COMM_PROTO[协议开销<br/>消息处理]
        end
        
        subgraph "存储瓶颈"
            STOR_IO[磁盘I/O<br/>数据读写]
            STOR_CACHE[缓存命中率<br/>数据局部性]
            STOR_DIST[数据分布<br/>负载均衡]
        end
        
        subgraph "系统瓶颈"
            SYS_SCHED[任务调度<br/>资源分配]
            SYS_SYNC[同步开销<br/>协调成本]
            SYS_FAULT[故障处理<br/>恢复时间]
        end
        
        COMP_CPU <--> COMM_NET
        COMP_GPU <--> COMM_LAT
        COMP_MEM <--> COMM_PROTO
        
        COMM_NET <--> STOR_IO
        COMM_LAT <--> STOR_CACHE
        COMM_PROTO <--> STOR_DIST
        
        STOR_IO <--> SYS_SCHED
        STOR_CACHE <--> SYS_SYNC
        STOR_DIST <--> SYS_FAULT
    end
    
    classDef compute fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef comm fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef storage fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef system fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class COMP_CPU,COMP_GPU,COMP_MEM compute
    class COMM_NET,COMM_LAT,COMM_PROTO comm
    class STOR_IO,STOR_CACHE,STOR_DIST storage
    class SYS_SCHED,SYS_SYNC,SYS_FAULT system
```

## Core系统执行特色

### 1. 企业级执行保障
- **全链路监控**: 从请求到响应的完整链路追踪
- **自动化运维**: 智能故障检测和自动恢复
- **弹性伸缩**: 根据负载动态调整计算资源
- **版本控制**: 模型和数据的版本管理

### 2. 高性能执行优化
- **异步执行**: 计算和通信的异步并行
- **批处理优化**: 智能批处理策略
- **缓存机制**: 多层缓存优化数据访问
- **预取策略**: 预测性数据预加载

### 3. 分布式协调机制
- **一致性保证**: 分布式状态的强一致性
- **负载均衡**: 动态负载感知和均衡
- **故障隔离**: 故障影响的最小化
- **优雅降级**: 部分故障下的服务可用性

### 4. 可观测性支持
- **实时监控**: 关键指标的实时监控
- **日志聚合**: 分布式日志的统一收集
- **性能分析**: 详细的性能剖析和优化建议
- **告警通知**: 智能告警和故障通知

这个Core执行流程文档展示了一个成熟的工业级分布式RLHF系统的复杂执行机制，为理解和优化大规模语言模型训练提供了详细的技术指导。
