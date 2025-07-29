# AReaL System Architecture Documentation

This directory contains comprehensive documentation of the AReaL (Ant Reasoning Reinforcement Learning) system architecture.

## System Overview Diagram

The `system-overview.puml` file contains a comprehensive PlantUML diagram that illustrates the high-level architecture of the AReaL system. This diagram shows how different components interact to provide fully asynchronous reinforcement learning capabilities for large language models.

### Key Architectural Features Illustrated

1. **Fully Asynchronous RL Training**: The diagram highlights the decoupled generation and training architecture that allows continuous model inference without waiting for training completion.

2. **Component Organization**: Shows the actual code structure from the `realhf/` package, including:
   - **API Layer**: CLI arguments, core APIs, and quickstart functionality
   - **Model Implementation**: Backend integrations (SGLang, vLLM, Megatron), neural modules, and parallelism support
   - **Training System**: Core workers, control systems, data management, and communication infrastructure
   - **Scheduler**: Job scheduling and evaluation systems
   - **Infrastructure**: Base services like logging, topology management, and utilities
   - **Implementations**: Concrete agent, dataset, and environment implementations
   - **Experiments**: Both asynchronous and traditional experiment configurations

3. **System Relationships**: Demonstrates key interactions between components:
   - How the controller manages both generation and training workers independently
   - Asynchronous data streaming from rollout workers to training workers
   - Infrastructure services supporting distributed operations
   - Model backend integration with neural components

### Understanding the Asynchronous Architecture

The diagram specifically emphasizes three critical aspects of AReaL's asynchronous design:

- **Generation Server**: Operates continuously without blocking on training completion
- **Model Worker**: Handles parallel training updates with stale data using decoupled PPO
- **Push/Pull Stream**: Enables asynchronous data streaming for maximum throughput

### Files

- `system-overview.puml`: PlantUML source file for the architecture diagram
- `system-overview.png`: Generated PNG image of the architecture diagram (if available)

### Usage

To regenerate the diagram from the PlantUML source:

```bash
# Install PlantUML and graphviz
wget https://github.com/plantuml/plantuml/releases/download/v1.2024.8/plantuml-1.2024.8.jar
sudo apt install graphviz

# Generate diagram
java -jar plantuml-1.2024.8.jar docs/architecture/system-overview.puml
```

This diagram serves as a comprehensive reference for developers and users who want to understand how AReaL achieves its high-performance asynchronous RL training capabilities.