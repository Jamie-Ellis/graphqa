# GraphQA User Guide

## Quick Start

### Installation
```bash
pip install graphqa
```

### Basic Usage
```python
from graphqa import GraphQA

# Initialize with your dataset
agent = GraphQA(dataset_name="amazon")
agent.load_dataset()

# Ask questions in natural language
response = agent.ask("What are the most connected nodes?")
print(response)

agent.shutdown()
```

### Interactive Demo
```bash
python -m graphqa.demo
```

## Core Features

- **Zero Configuration**: Automatic schema discovery
- **Universal Tools**: Works across any graph dataset
- **Natural Language**: Ask questions in plain English
- **Production Ready**: Built-in observability and error handling

## Example Questions

### Discovery
- "What attributes do nodes have?"
- "How many nodes and edges are there?"
- "Show me a sample of the data"

### Analysis  
- "What's the degree distribution?"
- "Which nodes have highest centrality?"
- "Find communities in the graph"

### Domain-Specific (Amazon example)
- "What's the average rating of electronics?"
- "Which product categories are most popular?"
- "Find products with similar characteristics"
