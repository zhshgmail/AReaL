# AReaLite 系统架构功能视图

## 概述

本文档提供了 AReaLite 系统的高层功能视图，以 PlantUML 组件图的形式展示系统的分层架构和主要功能组件。AReaLite 是 AReaL 的轻量级版本，专注于为 AI 研究者提供更直观、易用的强化学习训练接口。

## 架构图文件

- **文件**: `AReaLite_architecture.puml`
- **类型**: PlantUML 组件图
- **描述**: 展示 AReaLite 系统的6层功能架构

## 系统设计理念

AReaLite 遵循以下7个核心设计原则：

1. **原生异步RL训练支持** - 从底层支持生成和训练的解耦
2. **AI中心化设计** - 最小化对系统概念（如"PlacementGroup"）的暴露
3. **PyTorch中心化方法** - 使用原生PyTorch类型，避免不必要的抽象
4. **透明的算法编排** - 使操作流程清晰易懂
5. **开发者友好导航** - 通过IDE的Ctrl+点击轻松访问实现细节
6. **生态兼容性** - 与现有ML/RL工具平滑集成
7. **单文件定制** - 允许在单个文件内修改RL流水线

## 系统分层说明

### 1. 应用层 (Application Layer)
系统的最顶层，为用户提供具体的训练示例和配置管理：
- **示例脚本**: 提供GSM8K、CLEVR等具体任务的完整训练示例
- **配置管理**: 统一的实验配置和参数管理
- **数据集接口**: 提供标准化的数据集加载和预处理接口

### 2. 工作流层 (Workflow Layer)
AReaLite的核心创新层，定义灵活的强化学习数据收集模式：
- **RLVR工作流**: 强化学习价值排序工作流，支持多轮对话和奖励计算
- **多轮对话工作流**: 支持工具调用和多轮交互的对话工作流
- **视觉RLVR工作流**: 支持视觉输入的强化学习工作流
- **工作流基类**: 定义工作流统一接口和数据流规范

### 3. API层 (API Layer)
定义系统的核心接口和抽象，实现接口与实现的分离：
- **引擎API**: 定义训练引擎和推理引擎的抽象接口
- **工作流API**: 定义强化学习数据收集工作流的标准接口
- **环境API**: 定义强化学习环境和奖励函数接口
- **IO结构**: 定义LLM请求、响应和元数据的标准数据结构
- **CLI参数**: 定义命令行参数和配置数据类
- **奖励API**: 定义奖励函数和评估指标的接口

### 4. 引擎层 (Engine Layer)
提供具体的训练和推理引擎实现，支持SPMD模式：
- **FSDP训练引擎**: 基于PyTorch FSDP2的分布式训练引擎
- **PPO算法引擎**: PPO强化学习算法的具体实现
- **SGLang远程引擎**: 连接远程SGLang服务器的推理引擎
- **SFT训练引擎**: 监督微调训练的语言模型引擎
- **HuggingFace基础引擎**: 基于HuggingFace模型的基础训练引擎

### 5. 启动器层 (Launcher Layer)
负责分布式部署和调度，支持多种环境：
- **Ray启动器**: 使用Ray框架进行多节点分布式任务调度
- **Slurm启动器**: 在Slurm集群环境中启动分布式训练任务
- **本地启动器**: 单机或本地环境的任务启动管理
- **SGLang服务器**: 管理SGLang推理服务器的生命周期

### 6. 工具层 (Utils Layer)
提供支持功能和工具，服务于各个上层组件：
- **分布式工具**: 分布式训练的通信和同步工具
- **设备管理**: GPU设备分配和内存管理
- **模型工具**: 模型加载、保存和权重更新工具
- **数据工具**: 数据预处理、批处理和流式处理工具
- **评估器**: 模型性能评估和指标计算
- **统计日志**: 训练过程统计信息和日志记录
- **保存器**: 模型检查点保存和恢复管理
- **网络工具**: 网络通信和HTTP服务相关工具

## 层次关系

系统采用分层架构设计，层次间的主要依赖关系为：
- 应用层 → 工作流层 → API层 → 引擎层 → 启动器层 → 工具层

### 跨层直接依赖

虽然系统整体上是分层的，但存在一些跨层的直接依赖关系：
- **应用层直接实例化引擎层**: 为了简化用户使用，示例脚本直接创建具体的引擎实例
- **工作流层直接调用引擎层**: 工作流在执行过程中直接调用推理引擎
- **引擎层直接使用工具层**: 引擎实现需要直接使用各种工具函数

这些跨层依赖是AReaLite简化设计的体现，旨在减少不必要的中间抽象层。

## 与主AReaL系统的区别

相比于主AReaL系统的复杂架构，AReaLite具有以下特点：

1. **更简单的层次结构**: 从7层减少到6层，去除了复杂的调度层和系统层
2. **工作流层创新**: 引入专门的工作流层，使RL数据收集过程更加灵活和直观
3. **AI研究者友好**: 采用SPMD模式而非worker模式，更符合AI研究者的使用习惯
4. **减少抽象层**: 直接使用PyTorch原生类型，避免过度抽象
5. **单文件定制**: 支持在单个文件内完成算法定制，降低学习成本

## 如何查看图表

1. 使用在线 PlantUML 编辑器：访问 [PlantUML Online](http://www.plantuml.com/plantuml/uml/) 并粘贴 `AReaLite_architecture.puml` 文件内容
2. 使用本地 PlantUML 工具：安装 PlantUML 后运行 `plantuml AReaLite_architecture.puml`
3. 使用支持 PlantUML 的 IDE 插件，如 VS Code 的 PlantUML 扩展

## 使用示例

### 基本RL训练

```bash
# 使用Ray启动器：4节点（每节点4GPU），3节点用于生成，1节点用于训练
python3 -m arealite.launcher.ray examples/arealite/gsm8k_grpo.py \
    --config examples/arealite/configs/gsm8k_grpo.yaml \
    experiment_name=my_experiment \
    trial_name=my_trial \
    allocation_mode=sglang.d12p1t1+d4p1t1 \
    cluster.n_nodes=4 \
    cluster.n_gpus_per_node=4
```

### Slurm集群训练

```bash
# 使用Slurm启动器：16节点（每节点8GPU），12节点用于生成，4节点用于训练
python3 -m arealite.launcher.slurm examples/arealite/gsm8k_grpo.py \
    --config examples/arealite/configs/gsm8k_grpo.yaml \
    experiment_name=my_experiment \
    trial_name=my_trial \
    allocation_mode=sglang.d96p1t1+d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8
```

## 系统特色

该架构图突出展示了 AReaLite 系统的核心特色：

- **轻量级设计**: 相比主AReaL系统，显著减少了架构复杂度
- **工作流中心**: 工作流层的创新设计使RL数据收集更加灵活
- **SPMD友好**: 引擎层支持AI研究者熟悉的SPMD训练模式
- **多后端支持**: 启动器层支持从单机到大规模集群的多种部署环境
- **生态兼容**: 与PyTorch和HuggingFace生态系统深度集成
- **开发者友好**: 透明的实现和单文件定制能力降低使用门槛