"""
GraphQA: Natural Language Graph Analysis Framework

A schema-agnostic graph analysis framework that can answer natural language questions
about any graph dataset using advanced algorithms and LLM-powered analysis.

Core Philosophy: "Any Graph, Any Question, Minimal Integration"

Key Components:
- Universal data loaders for easy dataset integration
- Schema discovery that adapts to any graph structure  
- Universal analysis tools that work across domains
- Natural language agent for intelligent interaction
"""

# Core agent
from .agent import UniversalRetrievalAgent as GraphQA

# Configuration
from .config import UniversalRetrieverConfig as GraphQAConfig

# Data structures
from .data_structures import (
    AttributeType, AttributeInfo, SchemaInfo,
    DatasetInfo, LoadResult, AnalysisResult
)

# Loaders
from .loaders import (
    BaseGraphLoader, AmazonProductLoader, ArchitectureLoader
)

# Universal tools
from .tools import (
    UniversalGraphExplorer,
    UniversalGraphQuery, 
    UniversalGraphStats,
    UniversalNodeAnalyzer,
    UniversalAlgorithmSelector
)

__version__ = "2.0.0"

__all__ = [
    # Main agent
    "UniversalRetrievalAgent",
    
    # Configuration
    "UniversalRetrieverConfig",
    
    # Data structures
    "AttributeType", "AttributeInfo", "SchemaInfo",
    "DatasetInfo", "LoadResult", "AnalysisResult",
    
    # Loaders
    "BaseGraphLoader", "AmazonProductLoader", "ArchitectureLoader",
    
    # Universal tools
    "UniversalGraphExplorer", "UniversalGraphQuery", "UniversalGraphStats",
    "UniversalNodeAnalyzer", "UniversalAlgorithmSelector"
]

# Easy imports for common usage patterns
def create_agent(dataset_name: str = "amazon", **kwargs):
    """
    Create a UniversalRetrievalAgent with sensible defaults.
    
    Args:
        dataset_name: Dataset to load ("amazon", "architecture", etc.)
        **kwargs: Additional arguments passed to UniversalRetrievalAgent
        
    Returns:
        Configured UniversalRetrievalAgent instance
    """
    agent = UniversalRetrievalAgent(dataset_name=dataset_name, **kwargs)
    agent.load_dataset()
    return agent

def load_dataset(dataset_name: str, config: dict = None):
    """
    Load a dataset and return the graph with schema.
    
    Args:
        dataset_name: Name of dataset ("amazon", "architecture", etc.)
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (graph, schema, loader)
    """
    if dataset_name == "amazon":
        loader = AmazonProductLoader(config)
    elif dataset_name == "architecture":
        loader = ArchitectureLoader(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    graph = loader.load_graph()
    schema = loader.discover_schema(graph)
    
    return graph, schema, loader 