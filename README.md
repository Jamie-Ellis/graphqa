# GraphQA: Natural Language Graph Analysis Framework

> **Ask questions about any graph in natural language - no SQL, no graph query languages, just plain English**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-green.svg)](https://langchain.com/)

> **⚠️ Software Disclaimer**: GraphQA is provided "as is" without warranty of any kind. This is research/experimental software designed for graph analysis and exploration. Use at your own discretion and always validate results for production use cases.

## 📑 Table of Contents

- [🎯 What is GraphQA?](#-what-is-graphqa)
- [🧠 Graph Algorithm Intelligence](#-graph-algorithm-intelligence)
- [🚀 Quick Start](#-quick-start)
- [🗣️ Interactive Chat](#️-interactive-chat)
- [📖 Programmatic Usage](#-programmatic-usage)
- [🔧 Custom Data Loaders](#-custom-data-loaders)
- [🔋 Memory & Performance](#-memory--performance)
- [⚙️ Configuration](#️-configuration)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📁 Project Files Explained](#-project-files-explained)

## 🎯 What is GraphQA?

GraphQA is a **comprehensive graph analysis framework** that transforms complex graph datasets into natural language conversations. It's essentially a **graph algorithm engine with an AI interface** that understands what you're asking and automatically selects the right NetworkX algorithms to answer your questions.

### 🧮 The Graph Algorithm Problem

Traditional graph analysis requires:
- **Deep expertise** in graph theory and algorithms
- **Programming skills** to implement NetworkX code
- **Domain knowledge** to know which algorithms apply when
- **Trial and error** to find the right analysis approach

GraphQA solves this by providing an **intelligent algorithm selection layer** that:
- **Understands natural language** questions about graph data
- **Automatically chooses** appropriate graph algorithms
- **Executes analysis** using proven NetworkX implementations
- **Presents results** in human-readable format

### 🏗️ Core Architecture: NetworkX + AI

**Foundation**: GraphQA is built on [NetworkX](https://networkx.org/), the gold standard for graph analysis in Python. Your entire dataset lives in memory as a NetworkX MultiDiGraph, providing:

- **⚡ Instant access** to 500+ NetworkX algorithms
- **🔬 Research-grade implementations** (PageRank, Louvain, Dijkstra, etc.)
- **📊 Rich data structures** supporting any graph topology
- **🔄 Interactive analysis** with sub-second response times
- **🧮 Mathematical precision** for complex computations

**Intelligence Layer**: AI agents intelligently orchestrate graph algorithms by:

1. **Schema Discovery**: Embedding-based analysis of your graph structure
2. **Question Understanding**: LangChain ReAct agents parse natural language
3. **Algorithm Selection**: Smart routing to optimal NetworkX algorithms  
4. **Execution**: Running analysis with appropriate parameters
5. **Result Synthesis**: Converting algorithm outputs to insights

### 🎯 Graph Analysis Made Accessible

**Traditional NetworkX** requires expertise:
```python
# Manual algorithm selection and interpretation
communities = nx.community.greedy_modularity_communities(G)
centrality = nx.pagerank(G)
# ... complex result analysis ...
```

**GraphQA** makes it conversational:
```
🤔 Ask GraphQA: "Find communities and influential nodes"
📊 Found 1,247 communities. Top nodes: B001, B002, B003...
```

### 🧠 Schema-Agnostic Intelligence

GraphQA makes **zero assumptions** about your data structure:

- **🔍 Automatic Discovery**: embeddings map attributes to semantic concepts
- **🤖 Contextual Understanding**: "expensive" automatically maps to price/cost attributes
- **📊 Algorithm Adaptation**: Community detection parameters adjust to graph size/density

**The Result**: Whether you're analyzing protein interactions, supply chains, or social media networks, GraphQA provides the same powerful natural language interface backed by rigorous graph algorithms.

## 🧠 Graph Algorithm Intelligence

GraphQA's power comes from its **intelligent algorithm selection system** that automatically chooses and configures the right NetworkX algorithms for your questions. Here's what's under the hood:

### 🔧 The Five Universal Tools

GraphQA's ReAct agent uses exactly 5 specialized tools:

#### 1. 🔍 **Graph Explorer** - Schema discovery and search operations
- Embedding-based attribute discovery  
- Multi-attribute filtering and pattern matching
- Sample data retrieval

#### 2. 🎯 **Graph Query** - Targeted data retrieval and neighborhood analysis  
- Attribute-based node/edge filtering
- Multi-hop neighborhood exploration (up to depth N)
- Similarity searches and metric-based queries

#### 3. 📊 **Graph Statistics** - Network measurements and distributions
- Centrality measures (PageRank, betweenness, degree)
- Connectivity analysis and topology metrics
- Attribute distributions and summary statistics

#### 4. 🔬 **Node Analyzer** - Deep analysis of individual nodes
- Node-specific centrality and importance metrics
- Local neighborhood analysis and clustering
- Similarity computation with other nodes

#### 5. 🧠 **Algorithm Selector** - Intelligent algorithm selection
- Community detection (Louvain, greedy modularity, label propagation)  
- Pathfinding algorithms (Dijkstra, shortest paths)
- Network topology analysis and connectivity patterns

### 🎛️ Intelligent Algorithm Selection

GraphQA doesn't just run algorithms randomly - it uses **sophisticated decision logic**:

**Dataset-Aware Selection**:
```python
# GraphQA automatically chooses based on your graph characteristics:
if graph_size == "small" and preference == "comprehensive":
    use_betweenness_centrality()  # O(V³) - thorough but slow
elif graph_size == "large" and preference == "fast":  
    use_degree_centrality()      # O(V) - fast approximation
else:
    use_pagerank()               # O(V+E) - balanced choice
```

**Performance-Optimized Routing**:
- **Small graphs** (< 1K nodes): Use comprehensive algorithms (betweenness centrality, all-pairs shortest paths)
- **Medium graphs** (1K-100K nodes): Balance quality vs speed (PageRank, greedy modularity)  
- **Large graphs** (100K+ nodes): Prioritize scalable algorithms (degree centrality, label propagation)

**Context-Aware Configuration**:
- **Community Detection**: Louvain for undirected graphs, greedy modularity for directed
- **Centrality**: Adapts to graph density and connectivity patterns
- **Pathfinding**: Chooses between Dijkstra, A*, or bidirectional search

### 🎯 Question-to-Algorithm Mapping

Here's how natural language maps to specific graph algorithms:

| Question Type | Selected Algorithms | NetworkX Functions |
|---------------|--------------------|--------------------|
| **"Most connected nodes"** | Degree, PageRank centrality | `nx.degree_centrality()`, `nx.pagerank()` |
| **"Find communities"** | Louvain, greedy modularity | `nx.community.louvain_communities()` |
| **"Shortest path"** | Dijkstra, A* pathfinding | `nx.shortest_path()`, `nx.astar_path()` |
| **"Important nodes"** | Betweenness, closeness centrality | `nx.betweenness_centrality()` |
| **"Network structure"** | Clustering, connectivity analysis | `nx.average_clustering()`, `nx.connected_components()` |
| **"Similar products"** | Structural similarity, attribute matching | `nx.jaccard_coefficient()`, custom similarity |

### 🔬 Algorithm Quality & Validation

GraphQA uses **research-grade implementations**:
- **NetworkX algorithms**: Peer-reviewed, mathematically validated
- **Performance tested**: Benchmarked on real-world datasets
- **Parameter tuned**: Automatically optimized for different graph types
- **Error handling**: Graceful fallbacks for edge cases

**Example Output Quality**:
```
🤔 Ask GraphQA: "Find communities in this network"

🧠 Algorithm Selection:
   • Graph: 29,091 nodes, 60,168 edges (medium, directed)
   • Chosen: Greedy modularity optimization
   • Rationale: Best balance of quality vs performance for this size

📊 Results:
   • Found 2,743 communities (modularity = 0.892)
   • Largest community: 150 nodes
   • Average community size: 10.6 nodes
   • Execution time: 49.8 seconds
```

This intelligent algorithm selection is what makes GraphQA special - it brings **expert-level graph analysis** to anyone who can ask a question in English.

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (recommended: Python 3.10+)
- **OpenAI API Key** for LLM functionality

### Installation

#### Option 1: Interactive Setup (Recommended for New Users)

```bash
# Clone the repository
git clone https://github.com/your-org/graphqa.git
cd graphqa

# Run interactive setup - guides you through everything
python quickstart.py
```

#### Option 2: Automated Setup (For Experienced Users)

```bash
# Clone the repository
git clone https://github.com/your-org/graphqa.git
cd graphqa

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Automated setup
python quickstart.py --auto

# Or use the wrapper script
./setup
```

#### Option 3: Manual Install

```bash
# Clone and install manually
git clone https://github.com/your-org/graphqa.git
cd graphqa

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install GraphQA (core dependencies)
pip install -e .

# Or install with development tools
pip install -e ".[dev]"

# Or install with observability
pip install -e ".[observability]"

# Or install everything
pip install -e ".[all]"
```

### First Run

Set your OpenAI API key:

**Option 1 (Recommended): Use .env file**
```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=your-api-key-here
```

**Option 2: Set environment variable**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Test the installation:

```python
from graphqa import GraphQA

# Initialize GraphQA
agent = GraphQA(dataset_name="amazon")
print("✅ GraphQA ready!")
```

## ⚡ Key Technical Features

- **🔗 NetworkX Integration**: Full compatibility with NetworkX ecosystem and algorithms
- **🧠 Zero Configuration**: Automatic schema discovery - no manual setup required
- **🔍 Embedding-Based Search**: 384-dimensional semantic vectors for intelligent attribute matching
- **🤖 LangChain ReAct Agents**: Multi-step reasoning with tool selection and execution
- **📊 Universal Graph Algorithms**: Community detection, centrality, clustering, pathfinding
- **💾 In-Memory Performance**: Sub-second analysis on graphs with 100K+ nodes
- **🔧 Production Ready**: Built-in observability, error handling, and memory management

## 🗣️ Interactive Chat

### Start the Interactive Demo

**The easiest way to use GraphQA is through the interactive chat interface:**

```bash
# Start interactive GraphQA chat
python -m graphqa.demo

# Or with specific dataset
python -m graphqa.demo --dataset amazon
```

This launches an interactive session where you can:
- Ask questions in natural language
- Get immediate responses about your graph data  
- Explore data without writing code


**Example Interactive Session:**
```
🔍 GraphQA Interactive Demo - Amazon Products Dataset
💡 Ask questions about your graph data in plain English!

🤔 Ask GraphQA: What are the highest rated products?

📊 I found the highest rated products in the Amazon dataset...
[Results with top-rated products including ASINs like B0001234...]

🤔 Ask GraphQA: What categories do these products belong to?

📂 Based on the previous results, here are the categories...
[Category information for the highly rated products]

🤔 Ask GraphQA: Find products under $50 with good ratings

🛍️ I found products under $50 with high ratings...
[Budget-friendly alternatives with ratings above 4.0]
```

### CLI Usage

```bash
# Quick interactive demo
python -m graphqa.cli
```

## 📖 Programmatic Usage

### Basic Natural Language Interface

```python
from graphqa import GraphQA

# Load any graph dataset into memory as NetworkX graph
agent = GraphQA(dataset_name="amazon")  # or "architecture"
agent.load_dataset()  # Loads into NetworkX MultiDiGraph

# Ask questions in natural language  
response = agent.ask("What are the most connected nodes?")
print(response)

# Follow-up questions use conversation context
response = agent.ask("Show me their categories")
print(response)

agent.shutdown()
```

### Direct NetworkX Access

```python
from graphqa import GraphQA

# Load dataset
agent = GraphQA(dataset_name="amazon")
agent.load_dataset()

# Access the underlying NetworkX graph directly
graph = agent.graph  # NetworkX MultiDiGraph object
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")

# Use any NetworkX algorithms directly
import networkx as nx
centrality = nx.degree_centrality(graph)
communities = nx.community.greedy_modularity_communities(graph)

# Or let GraphQA's AI choose the right algorithms
response = agent.ask("Find the most central nodes and detect communities")
print(response)
```

### Custom Data Loading

```python
import networkx as nx
from graphqa import GraphQA

# Create your own NetworkX graph
graph = nx.MultiDiGraph()
graph.add_node("A", type="server", region="us-west-2")
graph.add_node("B", type="database", region="us-east-1")  
graph.add_edge("A", "B", relationship="connects_to", weight=1.0)

# Load into GraphQA for natural language analysis
agent = GraphQA()
agent.graph = graph
agent._discover_and_setup_schema()  # Auto-discover schema

# Now ask questions about your custom graph
response = agent.ask("What nodes are in the us-west-2 region?")
print(response)
```

## 🔧 Custom Data Loaders

GraphQA is designed to work with **any graph dataset**. You can create custom data loaders to import your specific data format into the GraphQA ecosystem.

### 🏗️ Data Loader Architecture  

All data loaders inherit from `BaseGraphLoader`, which provides:
- **Automatic schema discovery** using embedding-based analysis
- **Validation and error handling** for robust data loading
- **Metadata generation** for intelligent tool configuration
- **Universal interface** that works with all GraphQA tools

### 📖 BaseGraphLoader Interface

```python
from graphqa.loaders import BaseGraphLoader
import networkx as nx

class MyCustomLoader(BaseGraphLoader):
    """Custom loader for your specific data format"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.data_source = self.config.get('data_source', 'my_data.json')
    
    def load_graph(self) -> nx.MultiDiGraph:
        """
        REQUIRED: Load your data and return a NetworkX MultiDiGraph
        This is the core method you must implement
        """
        graph = nx.MultiDiGraph()
        
        # Your custom data loading logic here
        # Example: loading from JSON, CSV, database, API, etc.
        
        return graph
    
    def get_dataset_description(self) -> str:
        """REQUIRED: Describe what this dataset contains"""
        return "My custom dataset containing..."
    
    def get_sample_queries(self) -> List[str]:
        """REQUIRED: Provide example questions for this dataset"""
        return [
            "What are the most connected entities?",
            "Find clusters in the data",
            "Show me the network structure"
        ]
    
    # Optional: Override these methods for custom behavior
    def get_dataset_name(self) -> str:
        return "MyCustomDataset"
    
    def get_data_sources(self) -> List[str]:
        return [self.data_source]
```

### 🎯 Detailed Implementation Guide

#### Step 1: Set Up Your Loader Class

```python
import json
import pandas as pd
import networkx as nx
from graphqa.loaders import BaseGraphLoader
from pathlib import Path

class SocialNetworkLoader(BaseGraphLoader):
    """Example: Load social network data from CSV files"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Configure data sources
        self.users_file = self.config.get('users_file', 'users.csv')
        self.connections_file = self.config.get('connections_file', 'connections.csv')
        self.max_users = self.config.get('max_users', None)
        
        self.logger.info(f"Social network loader initialized")
```

#### Step 2: Implement the Core Loading Logic

```python
    def load_graph(self) -> nx.MultiDiGraph:
        """Load social network data into NetworkX graph"""
        graph = nx.MultiDiGraph()
        
        try:
            # Load user data (nodes)
            users_df = pd.read_csv(self.users_file)
            if self.max_users:
                users_df = users_df.head(self.max_users)
            
            # Add nodes with attributes
            for _, user in users_df.iterrows():
                graph.add_node(
                    user['user_id'],
                    name=user.get('name', ''),
                    age=user.get('age', 0),
                    location=user.get('location', ''),
                    followers=user.get('followers', 0),
                    # Add any other attributes from your data
                )
            
            # Load connections (edges)
            connections_df = pd.read_csv(self.connections_file)
            
            for _, conn in connections_df.iterrows():
                source = conn['user_id']
                target = conn['friend_id']
                
                # Only add edge if both nodes exist
                if graph.has_node(source) and graph.has_node(target):
                    graph.add_edge(
                        source, target,
                        relationship_type='friendship',
                        weight=conn.get('strength', 1.0),
                        created_date=conn.get('date', '')
                    )
            
            self.logger.info(f"Loaded {graph.number_of_nodes()} users, {graph.number_of_edges()} connections")
            return graph
            
        except Exception as e:
            raise LoaderError(f"Failed to load social network data: {e}")
```

#### Step 3: Provide Metadata and Examples

```python
    def get_dataset_description(self) -> str:
        return (
            "Social Network dataset containing user profiles and friendship connections. "
            "Includes user demographics, follower counts, and relationship data for "
            "analyzing social influence and community structures."
        )
    
    def get_sample_queries(self) -> List[str]:
        return [
            "Who are the most popular users?",
            "Find friend groups and communities",
            "What's the average number of connections?",
            "Show me users in the same location",
            "Which users have the most influence?",
            "Find the shortest path between two users",
            "Analyze age distribution in the network",
            "Identify users with unusual connection patterns"
        ]
    
    def get_dataset_name(self) -> str:
        return "Social Network"
```

### 🔧 Advanced Data Loader Features

#### Handle Multiple Data Sources

```python
class MultiSourceLoader(BaseGraphLoader):
    """Example: Combine data from multiple sources"""
    
    def load_graph(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        
        # Load from database
        graph = self._load_from_database(graph)
        
        # Enrich with API data
        graph = self._enrich_with_api(graph)
        
        # Add computed features
        graph = self._add_computed_features(graph)
        
        return graph
    
    def _load_from_database(self, graph):
        # Your database loading logic
        return graph
    
    def _enrich_with_api(self, graph):
        # API enrichment logic
        return graph
    
    def _add_computed_features(self, graph):
        # Add computed node/edge attributes
        for node in graph.nodes():
            graph.nodes[node]['degree'] = graph.degree(node)
        return graph
```

#### Data Validation and Error Handling

```python
    def load_graph(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        
        # Validate data sources exist
        for source in self.get_data_sources():
            if not Path(source).exists():
                raise LoaderError(f"Data source not found: {source}")
        
        try:
            # Load with progress tracking
            total_steps = 3
            
            self.logger.info("Step 1/3: Loading nodes...")
            self._load_nodes(graph)
            
            self.logger.info("Step 2/3: Loading edges...")
            self._load_edges(graph)
            
            self.logger.info("Step 3/3: Validating graph...")
            self._validate_graph(graph)
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise LoaderError(f"Cannot load dataset: {e}")
    
    def _validate_graph(self, graph):
        """Custom validation logic"""
        if graph.number_of_nodes() == 0:
            raise LoaderError("No nodes loaded - check your data")
        
        # Add your specific validation rules
        isolated_nodes = [n for n in graph.nodes() if graph.degree(n) == 0]
        if len(isolated_nodes) > graph.number_of_nodes() * 0.5:
            self.logger.warning(f"Many isolated nodes: {len(isolated_nodes)}")
```

### 🚀 Using Your Custom Loader

#### Option 1: Direct Usage

```python
from graphqa import GraphQA
from my_loaders import SocialNetworkLoader

# Create custom config
config = {
    'users_file': 'my_social_data/users.csv',
    'connections_file': 'my_social_data/connections.csv',
    'max_users': 10000  # Limit for testing
}

# Initialize GraphQA with custom loader
loader = SocialNetworkLoader(config)
agent = GraphQA()
agent.loader = loader
agent.load_dataset()

# Now use natural language on your data!
response = agent.ask("Find the most influential users in the network")
print(response)
```

#### Option 2: Register Your Loader

```python
# Register your loader with GraphQA
from graphqa.loaders import register_loader

register_loader("social_network", SocialNetworkLoader)

# Now use it like built-in loaders
agent = GraphQA(dataset_name="social_network", config=config)
agent.load_dataset()
```

### 📋 Data Format Requirements

Your loader can handle **any data format**, but the output must be:

**✅ Required**:
- `nx.MultiDiGraph` object (supports any graph topology)
- Nodes with string IDs (can be any string: "user123", "product_B001", etc.)
- Basic error handling for missing files/invalid data

**🎯 Recommended**:
- **Node attributes**: Add meaningful properties (name, category, price, etc.)
- **Edge attributes**: Include relationship types, weights, timestamps
- **Data validation**: Check for duplicates, missing values, format issues
- **Progress logging**: Use `self.logger.info()` for status updates
- **Memory management**: Handle large datasets with chunking/streaming

**🔧 Best Practices**:
- **Attribute naming**: Use descriptive names (`product_title` vs `title`)
- **Data types**: Keep consistent types (all prices as float, all dates as strings)
- **Missing data**: Handle nulls gracefully (empty string vs None vs 0)
- **Performance**: Use pandas/numpy for large data processing
- **Documentation**: Provide clear dataset descriptions and sample queries

### 💡 Example Loaders

GraphQA includes reference implementations you can learn from:
- **`AmazonProductLoader`**: JSON files, product relationships, review data
- **`ArchitectureLoader`**: Cloud infrastructure, JSON configuration files

Check `src/graphqa/loaders/` for complete working examples!

## 🔋 Memory & Performance

### In-Memory Processing Benefits

GraphQA loads your entire graph into system memory as a NetworkX object, providing:

- **⚡ Sub-second queries**: No database round-trips or network latency
- **🧮 Rich algorithms**: Full NetworkX algorithm library available
- **🔄 Interactive analysis**: Immediate follow-up questions and exploration
- **📊 Complex analytics**: Multi-step analysis without data movement

### Memory Requirements

| Dataset Size | Memory Usage | Load Time | Recommended For |
|-------------|--------------|-----------|-----------------|
| **Small** (< 1K nodes) | < 50MB | < 5s | Development, testing |
| **Medium** (1K-10K nodes) | 50-200MB | 5-30s | Demos, prototypes |
| **Large** (10K-100K nodes) | 200MB-2GB | 30s-5min | Research, production |
| **Very Large** (100K+ nodes) | 2GB+ | 5+ min | High-memory servers |

### Performance Characteristics

- **Amazon Dataset (Full)**: ~200K products, ~1M reviews → 2GB RAM, 2-5min load
- **Amazon Dataset (Demo)**: ~5K products, ~10K reviews → 200MB RAM, 10-30s load
- **Query Performance**: Most questions answered in < 5 seconds
- **Memory Management**: Automatic cleanup, configurable limits

### Scaling Considerations

GraphQA is optimized for **interactive analysis** rather than big data processing:

✅ **Perfect for**: Research, prototyping, dashboard analytics, graph exploration  
✅ **Good for**: Production analytics on medium-large graphs (< 1M nodes)  
⚠️ **Consider alternatives for**: Massive graphs (> 10M nodes), streaming data, production OLTP

For larger datasets, consider:
- **Sampling strategies** (built-in test modes)
- **Graph databases** (Neo4j, Amazon Neptune) for storage + GraphQA for analysis
- **Distributed processing** (Apache Spark GraphX) for preprocessing

## ⚙️ Configuration

### Default Settings (Optimized for Demos)

GraphQA ships with **test mode enabled by default** to ensure fast loading and prevent memory issues:

- **Amazon dataset**: Loads 5,000 products (vs 200,000+ full dataset)
- **Loading time**: 10-30 seconds (vs 2-5 minutes for full dataset)  
- **Memory usage**: 100-200MB (vs 1-2GB for full dataset)

### Configuration File

GraphQA automatically loads `config.yaml` from the current directory. Edit this file to customize behavior:

```yaml
datasets:
  amazon_products:
    config:
      # DEMO MODE (default - fast & safe)
      test_mode: true
      max_products: 5000
      max_reviews: 10000
      
      # FULL DATASET (requires more resources)
      # test_mode: false
      # max_products: null
      # max_reviews: null
```

### Configuration Templates

Choose the right template for your use case:

```bash
# For demos and quick testing (recommended)
cp config-templates/demo.yaml config.yaml

# For research and full analysis
cp config-templates/full-analysis.yaml config.yaml
```

| Template | Products | Load Time | Memory | Use Case |
|----------|----------|-----------|---------|----------|
| **demo.yaml** | 5,000 | 10-30s | 200MB | Demos, tutorials, quick tests |
| **full-analysis.yaml** | 200,000+ | 2-5min | 2GB | Research, production analysis |

### Custom Configuration

```python
from graphqa import GraphQA

# Option 1: Use custom config file
agent = GraphQA(dataset_name="amazon", config_file="my_config.yaml")

# Option 2: Programmatic configuration
from graphqa.config import UniversalRetrieverConfig
config = UniversalRetrieverConfig()
config.datasets['amazon_products'].config['test_mode'] = False  # Full dataset
agent = GraphQA(dataset_name="amazon", config=config)

agent.load_dataset()
```

## 🛠️ Troubleshooting

### Common Issues

**Import Error: "No module named 'graphqa'"**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate
pip install -e .
```

**OpenAI API Error**
```bash
# Option 1: Check your .env file
cat .env  # Should contain: OPENAI_API_KEY=your-key-here

# Option 2: Set environment variable
export OPENAI_API_KEY="your-key-here"
```

**Memory Issues with Large Graphs**
```bash
# Option 1: Edit config.yaml to reduce limits
# datasets.amazon_products.config.max_products: 2000

# Option 2: Use smaller test mode
python -c "
from graphqa import GraphQA
from graphqa.config import UniversalRetrieverConfig
config = UniversalRetrieverConfig()
config.datasets['amazon_products'].config['max_products'] = 1000
agent = GraphQA(dataset_name='amazon', config=config)
"
```

**Slow Loading Performance**
```bash
# Check your current settings
grep -A 5 "test_mode" config.yaml

# For faster loading, ensure test mode is enabled
# test_mode: true (default)
# max_products: 5000 (vs 200,000+ full)
```

**Missing Dependencies**
```bash
# Install optional dependencies
pip install langfuse  # For observability
```

**Environment Variables Not Loading**
```bash
# Check if .env file exists and has correct format
cat .env
# Should contain: OPENAI_API_KEY=your-actual-key-here

# Reload .env in current session
python -c "from dotenv import load_dotenv; load_dotenv(override=True); import os; print('Loaded:', bool(os.getenv('OPENAI_API_KEY')))"
```

**Performance Issues / Out of Memory**
```bash
# Switch to demo mode (lighter load)
cp config-templates/demo.yaml config.yaml

# Or manually reduce limits in config.yaml
# test_mode: true
# max_products: 1000  # Even smaller
```

### Getting Help

1. **Check the logs**: GraphQA provides detailed logging
2. **Try the examples**: Run files in `examples/` directory  
3. **Open an issue**: [GitHub Issues](https://github.com/your-org/graphqa/issues)

## 📚 Documentation

- [User Guide](docs/user-guide.md) - Complete usage documentation
- [Examples](examples/) - Real-world usage examples

## 🤝 Contributing

```bash
# Development setup
git clone https://github.com/your-org/graphqa.git
cd graphqa

# Interactive setup (recommended)
python quickstart.py

# Or manual setup
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"  # Installs dev dependencies from pyproject.toml

# Run tests
pytest

# Format code
black src/
isort src/
```

## 📁 Project Files Explained

### Setup Files
| File | Purpose | When to Use |
|------|---------|-------------|
| **`quickstart.py`** | Interactive guided setup with options | New users, need help with API keys/config |
| **`setup`** | Automated setup script | CI/CD, experienced users |
| **`setup.py`** | Python package definition (legacy) | Used by pip (don't run directly) |

### Dependencies
| File | Purpose | Usage |
|------|---------|-------|
| **`pyproject.toml`** | Modern Python dependencies & project config | `pip install -e ".[dev]"` |

**Why no requirements.txt?** We use `pyproject.toml` (modern Python standard) instead of legacy `requirements.txt` files. This eliminates redundancy and follows current best practices.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

**Need help?** Open an issue or check our [troubleshooting guide](docs/user-guide.md#troubleshooting).
