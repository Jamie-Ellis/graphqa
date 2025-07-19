"""
Core data structures for the Universal Graph Retriever.

This module defines the fundamental data types used throughout the system
for representing datasets, schema information, and analysis results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import networkx as nx


class AttributeType(Enum):
    """Types of attributes that can be discovered in graph data"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical" 
    TEXT = "text"
    BOOLEAN = "boolean"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass
class AttributeInfo:
    """Information about a discovered attribute in the graph"""
    name: str
    attribute_type: AttributeType
    coverage: float  # Percentage of nodes that have this attribute
    unique_count: int
    sample_values: List[Any] = field(default_factory=list)
    
    # Type-specific information
    stats: Optional[Dict[str, float]] = None  # For numeric: min, max, avg, std
    categories: Optional[List[str]] = None    # For categorical: list of categories
    is_high_cardinality: bool = False        # True if too many unique values
    
    def __post_init__(self):
        """Set high cardinality flag based on unique count"""
        if self.attribute_type == AttributeType.CATEGORICAL and self.unique_count > 50:
            self.is_high_cardinality = True


@dataclass
class SchemaInfo:
    """Complete schema information about a graph dataset"""
    dataset_name: str
    node_count: int
    edge_count: int
    is_directed: bool
    
    # Attribute information
    node_attributes: Dict[str, AttributeInfo] = field(default_factory=dict)
    edge_attributes: Dict[str, AttributeInfo] = field(default_factory=dict)
    
    # Representative samples
    sample_nodes: List[Dict[str, Any]] = field(default_factory=list)
    sample_edges: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis suggestions
    suggested_queries: List[str] = field(default_factory=list)
    recommended_searches: List[str] = field(default_factory=list)
    
    def get_searchable_attributes(self) -> List[str]:
        """Get attributes suitable for search operations"""
        searchable = []
        for name, info in self.node_attributes.items():
            if info.attribute_type in [AttributeType.TEXT, AttributeType.CATEGORICAL]:
                if not info.is_high_cardinality:
                    searchable.append(name)
        return searchable
    
    def get_numeric_attributes(self) -> List[str]:
        """Get numeric attributes suitable for range operations"""
        return [name for name, info in self.node_attributes.items() 
                if info.attribute_type == AttributeType.NUMERIC]
    
    def get_categorical_attributes(self) -> List[str]:
        """Get categorical attributes suitable for grouping"""
        return [name for name, info in self.node_attributes.items()
                if info.attribute_type == AttributeType.CATEGORICAL and not info.is_high_cardinality]


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset"""
    name: str
    description: str
    loader_class: str
    
    # Graph information
    graph: Optional[nx.MultiDiGraph] = None
    schema: Optional[SchemaInfo] = None
    
    # Loading information  
    is_loaded: bool = False
    load_time_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    
    @property
    def node_count(self) -> int:
        """Get node count from graph or schema"""
        if self.graph:
            return self.graph.number_of_nodes()
        elif self.schema:
            return self.schema.node_count
        return 0
    
    @property  
    def edge_count(self) -> int:
        """Get edge count from graph or schema"""
        if self.graph:
            return self.graph.number_of_edges()
        elif self.schema:
            return self.schema.edge_count
        return 0


@dataclass
class LoadResult:
    """Result of a dataset loading operation"""
    success: bool
    dataset_info: Optional[DatasetInfo] = None
    error_message: Optional[str] = None
    load_time_seconds: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of a graph analysis operation"""
    operation: str
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    
    def add_interpretation(self, interpretation: str):
        """Add human-readable interpretation of results"""
        self.metadata["interpretation"] = interpretation
    
    def add_recommendations(self, recommendations: List[str]):
        """Add recommended follow-up actions"""
        self.metadata["recommendations"] = recommendations


@dataclass
class NodeAnalysis:
    """Detailed analysis of a single node"""
    node_id: str
    attributes: Dict[str, Any]
    
    # Connectivity information
    in_degree: int
    out_degree: int
    total_degree: int
    
    # Importance metrics
    pagerank_score: Optional[float] = None
    betweenness_centrality: Optional[float] = None
    degree_centrality: Optional[float] = None
    
    # Relationship information
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)
    
    # Analysis flags
    is_isolated: bool = False
    is_hub: bool = False
    is_bridge: bool = False
    
    # Context information
    neighborhood_size: int = 0
    clustering_coefficient: Optional[float] = None 