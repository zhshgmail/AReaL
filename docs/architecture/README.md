# AReaLite vs Core 架构分析文档索引

本文档集从软件架构角度全面分析了AReaL项目中AReaLite与Core系统的主要区别。

## 文档概览

### 📋 [架构分析总结报告](final_summary_report.md)
**推荐优先阅读** - 高层次的架构对比和战略建议
- 执行摘要和核心发现
- 适用场景分析
- 技术深度评估
- 选择建议

### 📊 [详细架构对比分析](arealite_vs_core_analysis.md)
深入的技术分析，包含：
- 设计理念差异
- 系统架构模式对比
- 代码复杂度分析
- 性能和扩展性评估
- 具体使用建议

### 🔧 [具体代码实例对比](code_examples_comparison.md)
通过真实代码片段展示差异：
- 入口点实现对比
- 配置和组件定义
- 训练逻辑实现
- 数据流处理方式
- 自定义开发体验

### 📈 [架构图表](architecture_diagrams.md)
可视化的架构对比：
- 系统架构图
- 代码复杂度对比图
- 训练流程序列图
- 抽象层次对比

## 快速导航

### 按读者类型

**🎓 AI研究者/学生**
1. 先读 [架构分析总结报告](final_summary_report.md) 了解整体情况
2. 查看 [具体代码实例对比](code_examples_comparison.md) 中的AReaLite示例
3. 建议从AReaLite开始学习

**🏭 工业界研究团队**
1. 重点关注 [详细架构对比分析](arealite_vs_core_analysis.md) 中的性能分析
2. 参考适用场景建议选择合适的系统
3. 考虑从AReaLite原型到Core生产的迁移路径

**⚙️ 系统工程师**
1. 详细阅读 [架构图表](architecture_diagrams.md) 理解系统设计
2. 重点学习Core系统的分布式架构
3. 关注 [详细架构对比分析](arealite_vs_core_analysis.md) 中的技术细节

### 按关注点

**🚀 快速上手**
- [具体代码实例对比](code_examples_comparison.md) → AReaLite示例
- AReaLite入口点：`examples/arealite/gsm8k_grpo.py`

**📐 架构设计**
- [架构图表](architecture_diagrams.md) → 系统架构对比
- [详细架构对比分析](arealite_vs_core_analysis.md) → 设计模式分析

**⚖️ 技术选型**
- [架构分析总结报告](final_summary_report.md) → 选择建议
- [详细架构对比分析](arealite_vs_core_analysis.md) → 适用场景

**🔬 深入研究**
- 所有文档都值得详细阅读
- 配合源码进行实践验证

## 关键概念速查

| 概念 | AReaLite | Core |
|------|----------|------|
| **设计理念** | AI-centric | System-centric |
| **编程模型** | SPMD | Master-Worker |
| **抽象层次** | 3层 | 6层 |
| **自定义方式** | 单文件修改 | 多模块注册 |
| **适用规模** | 1-64 GPU | 64-1000+ GPU |
| **学习曲线** | 1-2天 | 1-2周 |

## 实践建议

### 学习路径
1. **理论学习**: 阅读本文档集，理解两套系统的设计差异
2. **动手实践**: 从AReaLite的GSM8K例子开始，运行完整的训练流程
3. **深入探索**: 尝试修改奖励函数、算法参数，体验自定义开发
4. **规模扩展**: 根据需要考虑Core系统的高级功能

### 选择决策树
```
项目规模 < 64 GPU？
├─ 是 → 团队熟悉PyTorch？
│   ├─ 是 → 选择 AReaLite
│   └─ 否 → 学习后选择 AReaLite
└─ 否 → 有专门的系统工程师？
    ├─ 是 → 选择 Core
    └─ 否 → 考虑外包或使用云服务
```

## 贡献指南

如果您发现文档中的错误或有改进建议，请：

1. 检查相关源码确认问题
2. 提出具体的修改建议
3. 如果可能，提供更好的实例或图表

## 相关资源

- **源码仓库**: [AReaL GitHub](https://github.com/inclusionAI/AReaL)
- **官方文档**: [AReaL Documentation](https://inclusionai.github.io/AReaL/)
- **AReaLite设计文档**: `arealite/README.md`
- **论文**: [AReaL: A Large-Scale Asynchronous Reinforcement Learning System](https://arxiv.org/abs/2505.24298)