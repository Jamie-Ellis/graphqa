"""
Universal Graph Statistics Tool - Schema-aware statistical analysis for any graph dataset
"""

import json
import logging
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import networkx as nx
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..data_structures import SchemaInfo, AttributeType

logger = logging.getLogger(__name__)


class UniversalStatsInput(BaseModel):
    """Input schema for universal graph statistics"""
    
    stat_type: str = Field(
        description="Statistics category (distribution, centrality, connectivity, topology, summary)"
    )
    groupby: Optional[List[str]] = Field(
        default=None, description="Grouping attributes"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Node/edge filters"
    )
    top_k: Optional[int] = Field(default=10, description="Top K results to return")


class UniversalGraphStats(BaseTool):
    """Universal tool for computing statistics and insights from any graph dataset"""
    
    name: str = "universal_graph_stats"
    description: str = """📊 UNIVERSAL GRAPH STATISTICS - COMPREHENSIVE ANALYSIS FOR ANY DATASET

Compute detailed statistics and insights from any graph dataset with automatic schema adaptation.
Works seamlessly with Amazon products, cloud architecture, social networks, or any graph data.

🔧 SMART SCHEMA ADAPTATION: Automatically discovers available attributes and computes relevant statistics.

📈 STATISTICAL ANALYSIS TYPES:

📊 distribution - Attribute value distributions and counts
   Format: {{"stat_type": "distribution", "groupby": ["category"], "top_k": 15}}
   Returns: Count distributions for specified attributes

🌟 centrality - Node importance rankings and centrality metrics
   Format: {{"stat_type": "centrality", "top_k": 20}}
   Returns: PageRank, degree centrality, betweenness centrality rankings

🔗 connectivity - Connection patterns and network structure analysis
   Format: {{"stat_type": "connectivity", "filters": {{"min_degree": 5}}}}
   Returns: Connectivity metrics, isolated nodes, component analysis

🌐 topology - Graph-level structural metrics
   Format: {{"stat_type": "topology"}}
   Returns: Density, clustering, diameter, component counts

📋 summary - Comprehensive overview of the entire dataset
   Format: {{"stat_type": "summary"}}
   Returns: Complete statistical summary including all major metrics

🎯 attribute_analysis - Deep dive into specific attributes
   Format: {{"stat_type": "attribute_analysis", "groupby": ["brand", "price_range"]}}
   Returns: Detailed analysis of specified attributes with correlations

📋 PARAMETER REQUIREMENTS:
- Required: "stat_type" (string) - must be one of the types above
- Optional: "groupby" (list of strings) - attribute names for grouping
- Optional: "filters" (dict) - filter criteria for nodes
- Optional: "top_k" (integer) - number of top results to return (default: 10)

All statistics automatically adapt to your dataset's schema. No hardcoded assumptions!
"""
    
    args_schema: type[BaseModel] = UniversalStatsInput
    graph: nx.MultiDiGraph = None
    graph_schema: Optional[SchemaInfo] = None
    dataset_name: str = "Unknown"
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, graph: nx.MultiDiGraph, graph_schema: Optional[SchemaInfo] = None, dataset_name: str = "Unknown", **kwargs):
        """Initialize with graph and optional schema information"""
        super().__init__(
            graph=graph,
            graph_schema=graph_schema,
            dataset_name=dataset_name,
            **kwargs
        )
        object.__setattr__(self, 'logger', logging.getLogger(f"{__name__}.{self.__class__.__name__}"))
    
    def _run(self, stat_type: str, groupby: Optional[List[str]] = None, 
             filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> str:
        """Execute universal graph statistics analysis"""
        
        start_time = time.time()
        self.logger.info(f"Computing statistics: {stat_type}, groupby: {groupby}, filters: {filters}")
        
        try:
            # Route to appropriate statistics handler
            if stat_type == "distribution":
                result = self._compute_distributions(groupby, filters, top_k)
            elif stat_type == "centrality":
                result = self._compute_centrality(filters, top_k)
            elif stat_type == "connectivity":
                result = self._compute_connectivity(filters, top_k)
            elif stat_type == "topology":
                result = self._compute_topology()
            elif stat_type == "summary":
                result = self._compute_summary(top_k)
            elif stat_type == "attribute_analysis":
                result = self._compute_attribute_analysis(groupby, filters, top_k)
            else:
                return json.dumps({
                    "error": f"Unknown statistics type: {stat_type}",
                    "available_types": [
                        "distribution", "centrality", "connectivity", 
                        "topology", "summary", "attribute_analysis"
                    ]
                })
            
            execution_time = time.time() - start_time
            result["execution_time_seconds"] = round(execution_time, 3)
            result["dataset_name"] = self.dataset_name
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            self.logger.error(f"Statistics computation failed: {str(e)}")
            return json.dumps({
                "error": f"Statistics computation failed: {str(e)}",
                "stat_type": stat_type,
                "parameters": {"groupby": groupby, "filters": filters, "top_k": top_k}
            })
    
    def _compute_distributions(self, groupby: Optional[List[str]], filters: Optional[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        """Compute attribute value distributions"""
        if not groupby:
            # Discover interesting attributes to analyze
            groupby = self._discover_categorical_attributes()[:3]  # Top 3 most interesting
        
        distributions = {}
        
        for attribute in groupby:
            # Check if this is an edge attribute
            if self._is_edge_attribute(attribute):
                distributions[attribute] = self._analyze_edge_attribute(attribute, top_k)
            else:
                # Analyze as node attribute
                attribute_counts = Counter()
                total_nodes = 0
                
                for node_id, attrs in self._filtered_nodes(filters):
                    total_nodes += 1
                    if attribute in attrs:
                        try:
                            value = attrs[attribute]
                            # Handle unhashable types
                            if isinstance(value, (list, dict)):
                                value = str(value)
                            attribute_counts[value] += 1
                        except TypeError:
                            # Skip unhashable values
                            continue
                
                # Convert to sorted list
                sorted_counts = sorted(attribute_counts.items(), key=lambda x: x[1], reverse=True)
                
                distributions[attribute] = {
                    "total_values": sum(attribute_counts.values()),
                    "unique_values": len(attribute_counts),
                    "coverage": sum(attribute_counts.values()) / total_nodes if total_nodes > 0 else 0,
                    "top_values": sorted_counts[:top_k],
                    "value_distribution": dict(sorted_counts[:top_k])
                }
        
        return {
            "stat_type": "distribution",
            "attributes_analyzed": groupby,
            "total_nodes": self.graph.number_of_nodes(),
            "distributions": distributions
        }
    
    def _compute_centrality(self, filters: Optional[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        """Compute centrality metrics and rankings"""
        
        # Get filtered subgraph if filters specified
        if filters:
            filtered_graph = self._create_filtered_subgraph(filters)
        else:
            filtered_graph = self.graph
        
        centrality_metrics = {}
        
        # Degree Centrality
        degree_centrality = nx.degree_centrality(filtered_graph)
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
        centrality_metrics["degree_centrality"] = {
            "top_nodes": [{"node_id": node, "score": score, "attributes": dict(self.graph.nodes[node])} 
                         for node, score in top_degree],
            "average": sum(degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0
        }
        
        # PageRank (if graph is not too large)
        if filtered_graph.number_of_nodes() <= 10000:
            try:
                pagerank = nx.pagerank(filtered_graph)
                top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]
                centrality_metrics["pagerank"] = {
                    "top_nodes": [{"node_id": node, "score": score, "attributes": dict(self.graph.nodes[node])} 
                                 for node, score in top_pagerank],
                    "average": sum(pagerank.values()) / len(pagerank) if pagerank else 0
                }
            except:
                centrality_metrics["pagerank"] = {"error": "PageRank computation failed (possibly due to graph structure)"}
        
        # Betweenness Centrality (for smaller graphs)
        if filtered_graph.number_of_nodes() <= 1000:
            try:
                betweenness = nx.betweenness_centrality(filtered_graph)
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_k]
                centrality_metrics["betweenness_centrality"] = {
                    "top_nodes": [{"node_id": node, "score": score, "attributes": dict(self.graph.nodes[node])} 
                                 for node, score in top_betweenness],
                    "average": sum(betweenness.values()) / len(betweenness) if betweenness else 0
                }
            except:
                centrality_metrics["betweenness_centrality"] = {"error": "Betweenness centrality computation failed"}
        
        return {
            "stat_type": "centrality",
            "graph_size": {"nodes": filtered_graph.number_of_nodes(), "edges": filtered_graph.number_of_edges()},
            "centrality_metrics": centrality_metrics
        }
    
    def _compute_connectivity(self, filters: Optional[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        """Compute connectivity patterns and analysis"""
        
        # Get filtered subgraph if filters specified
        if filters:
            filtered_graph = self._create_filtered_subgraph(filters)
        else:
            filtered_graph = self.graph
        
        connectivity_analysis = {}
        
        # Degree distribution
        degrees = [filtered_graph.degree(node) for node in filtered_graph.nodes()]
        degree_counts = Counter(degrees)
        connectivity_analysis["degree_distribution"] = {
            "min_degree": min(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "average_degree": sum(degrees) / len(degrees) if degrees else 0,
            "degree_counts": dict(sorted(degree_counts.items())[:top_k])
        }
        
        # Isolated nodes (degree 0)
        isolated_nodes = [node for node in filtered_graph.nodes() if filtered_graph.degree(node) == 0]
        connectivity_analysis["isolated_nodes"] = {
            "count": len(isolated_nodes),
            "sample": isolated_nodes[:top_k]
        }
        
        # High-degree nodes (hubs)
        high_degree_nodes = sorted(
            [(node, filtered_graph.degree(node)) for node in filtered_graph.nodes()],
            key=lambda x: x[1], reverse=True
        )[:top_k]
        connectivity_analysis["hub_nodes"] = [
            {"node_id": node, "degree": degree, "attributes": dict(self.graph.nodes[node])} 
            for node, degree in high_degree_nodes
        ]
        
        # Connected components analysis
        if filtered_graph.is_directed():
            # For directed graphs, use weakly connected components
            components = list(nx.weakly_connected_components(filtered_graph))
        else:
            components = list(nx.connected_components(filtered_graph))
        
        component_sizes = [len(comp) for comp in components]
        connectivity_analysis["connected_components"] = {
            "total_components": len(components),
            "largest_component_size": max(component_sizes) if component_sizes else 0,
            "component_size_distribution": dict(Counter(component_sizes))
        }
        
        return {
            "stat_type": "connectivity",
            "graph_size": {"nodes": filtered_graph.number_of_nodes(), "edges": filtered_graph.number_of_edges()},
            "connectivity_analysis": connectivity_analysis
        }
    
    def _compute_topology(self) -> Dict[str, Any]:
        """Compute graph topology metrics"""
        
        topology_metrics = {}
        
        # Basic metrics
        topology_metrics["basic"] = {
            "number_of_nodes": self.graph.number_of_nodes(),
            "number_of_edges": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "is_multigraph": self.graph.is_multigraph()
        }
        
        # Density
        if self.graph.number_of_nodes() > 1:
            density = nx.density(self.graph)
            topology_metrics["density"] = density
        else:
            topology_metrics["density"] = 0
        
        # Clustering (for undirected graphs or smaller directed graphs)
        try:
            if not self.graph.is_directed() or self.graph.number_of_nodes() <= 1000:
                avg_clustering = nx.average_clustering(self.graph)
                topology_metrics["average_clustering"] = avg_clustering
        except:
            topology_metrics["average_clustering"] = "computation_failed"
        
        # Diameter and radius (for smaller connected graphs)
        try:
            if self.graph.number_of_nodes() <= 500:
                if self.graph.is_directed():
                    # Use weakly connected components
                    components = list(nx.weakly_connected_components(self.graph))
                else:
                    components = list(nx.connected_components(self.graph))
                
                if components:
                    largest_component = max(components, key=len)
                    subgraph = self.graph.subgraph(largest_component)
                    
                    if not self.graph.is_directed():
                        diameter = nx.diameter(subgraph)
                        radius = nx.radius(subgraph)
                        topology_metrics["diameter"] = diameter
                        topology_metrics["radius"] = radius
        except:
            topology_metrics["diameter"] = "computation_failed"
            topology_metrics["radius"] = "computation_failed"
        
        # Transitivity (global clustering coefficient)
        try:
            if not self.graph.is_directed():
                transitivity = nx.transitivity(self.graph)
                topology_metrics["transitivity"] = transitivity
        except:
            topology_metrics["transitivity"] = "computation_failed"
        
        return {
            "stat_type": "topology",
            "topology_metrics": topology_metrics
        }
    
    def _compute_summary(self, top_k: int) -> Dict[str, Any]:
        """Compute comprehensive summary statistics"""
        
        summary = {
            "stat_type": "summary",
            "dataset_overview": {
                "dataset_name": self.dataset_name,
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "graph_type": "directed" if self.graph.is_directed() else "undirected",
                "is_multigraph": self.graph.is_multigraph()
            }
        }
        
        # Schema information if available
        if self.graph_schema:
            summary["schema_info"] = {
                "node_attributes": len(self.graph_schema.node_attributes),
                "edge_attributes": len(self.graph_schema.edge_attributes),
                "top_node_attributes": list(self.graph_schema.node_attributes.keys())[:top_k],
                "top_edge_attributes": list(self.graph_schema.edge_attributes.keys())[:top_k]
            }
        
        # Quick topology metrics
        if self.graph.number_of_nodes() > 0:
            degrees = [self.graph.degree(node) for node in self.graph.nodes()]
            summary["connectivity_summary"] = {
                "average_degree": sum(degrees) / len(degrees),
                "max_degree": max(degrees),
                "min_degree": min(degrees),
                "isolated_nodes": sum(1 for d in degrees if d == 0)
            }
        
        # Discover and analyze top categorical attributes
        categorical_attrs = self._discover_categorical_attributes()[:3]
        if categorical_attrs:
            summary["attribute_distributions"] = {}
            for attr in categorical_attrs:
                value_counts = Counter()
                for node_id, attrs in self.graph.nodes(data=True):
                    if attr in attrs:
                        try:
                            value = attrs[attr]
                            if isinstance(value, (list, dict)):
                                value = str(value)
                            value_counts[value] += 1
                        except TypeError:
                            continue
                
                top_values = value_counts.most_common(top_k)
                summary["attribute_distributions"][attr] = {
                    "unique_values": len(value_counts),
                    "top_values": top_values
                }
        
        return summary
    
    def _compute_attribute_analysis(self, groupby: Optional[List[str]], filters: Optional[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        """Detailed analysis of specific attributes"""
        
        if not groupby:
            groupby = self._discover_categorical_attributes()[:2]
        
        attribute_analysis = {}
        
        for attribute in groupby:
            analysis = {
                "attribute_name": attribute,
                "analysis": {}
            }
            
            # Basic distribution
            value_counts = Counter()
            numeric_values = []
            total_nodes = 0
            nodes_with_attribute = 0
            
            for node_id, attrs in self._filtered_nodes(filters):
                total_nodes += 1
                if attribute in attrs:
                    nodes_with_attribute += 1
                    value = attrs[attribute]
                    
                    # Try to treat as numeric
                    try:
                        if isinstance(value, (int, float)):
                            numeric_values.append(float(value))
                        elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                            numeric_values.append(float(value))
                    except:
                        pass
                    
                    # Count for categorical analysis
                    try:
                        if isinstance(value, (list, dict)):
                            value = str(value)
                        value_counts[value] += 1
                    except TypeError:
                        continue
            
            # Coverage
            analysis["analysis"]["coverage"] = nodes_with_attribute / total_nodes if total_nodes > 0 else 0
            
            # Categorical analysis
            analysis["analysis"]["categorical"] = {
                "unique_values": len(value_counts),
                "top_values": value_counts.most_common(top_k),
                "distribution": dict(value_counts.most_common(top_k))
            }
            
            # Numeric analysis if applicable
            if numeric_values:
                analysis["analysis"]["numeric"] = {
                    "count": len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "average": sum(numeric_values) / len(numeric_values),
                    "median": sorted(numeric_values)[len(numeric_values) // 2]
                }
            
            attribute_analysis[attribute] = analysis
        
        return {
            "stat_type": "attribute_analysis",
            "attributes_analyzed": groupby,
            "total_nodes": self.graph.number_of_nodes(),
            "attribute_analysis": attribute_analysis
        }
    
    def _discover_categorical_attributes(self) -> List[str]:
        """Discover interesting categorical attributes to analyze"""
        if self.graph_schema and self.graph_schema.node_attributes:
            # Use schema information if available
            categorical_attrs = []
            for attr_name, attr_info in self.graph_schema.node_attributes.items():
                if attr_info.attribute_type in [AttributeType.CATEGORICAL, AttributeType.TEXT]:
                    categorical_attrs.append(attr_name)
            return categorical_attrs[:10]  # Limit to top 10
        
        # Fallback: sample nodes to discover attributes
        sample_nodes = list(self.graph.nodes(data=True))[:100]  # Sample first 100
        attr_counts = Counter()
        
        for node_id, attrs in sample_nodes:
            for attr_name in attrs.keys():
                attr_counts[attr_name] += 1
        
        # Return most common attributes
        return [attr for attr, count in attr_counts.most_common(10)]
    
    def _filtered_nodes(self, filters: Optional[Dict[str, Any]]):
        """Iterator over nodes that match the given filters"""
        for node_id, attrs in self.graph.nodes(data=True):
            if not filters:
                yield node_id, attrs
                continue
            
            matches = True
            for filter_attr, filter_value in filters.items():
                if filter_attr not in attrs:
                    matches = False
                    break
                
                attr_value = attrs[filter_attr]
                
                # Handle different filter types
                if isinstance(filter_value, dict):
                    if "min" in filter_value and attr_value < filter_value["min"]:
                        matches = False
                        break
                    if "max" in filter_value and attr_value > filter_value["max"]:
                        matches = False
                        break
                elif attr_value != filter_value:
                    matches = False
                    break
            
            if matches:
                yield node_id, attrs
    
    def _create_filtered_subgraph(self, filters: Dict[str, Any]) -> nx.MultiDiGraph:
        """Create a subgraph containing only nodes that match the filters"""
        matching_nodes = [node_id for node_id, attrs in self._filtered_nodes(filters)]
        return self.graph.subgraph(matching_nodes).copy()
    
    def _is_edge_attribute(self, attribute: str) -> bool:
        """Check if an attribute is an edge attribute by sampling edges"""
        if self.graph.number_of_edges() == 0:
            return False
        
        # Sample a few edges to check if the attribute exists
        edge_sample = list(self.graph.edges(data=True))[:10]
        for u, v, data in edge_sample:
            if attribute in data:
                return True
        return False
    
    def _analyze_edge_attribute(self, attribute: str, top_k: int) -> Dict[str, Any]:
        """Analyze an edge attribute distribution"""
        attribute_counts = Counter()
        total_edges = 0
        
        for u, v, data in self.graph.edges(data=True):
            total_edges += 1
            if attribute in data:
                try:
                    value = data[attribute]
                    # Handle unhashable types
                    if isinstance(value, (list, dict)):
                        value = str(value)
                    attribute_counts[value] += 1
                except TypeError:
                    # Skip unhashable values
                    continue
        
        # Convert to sorted list
        sorted_counts = sorted(attribute_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "type": "edge_attribute",
            "total_values": sum(attribute_counts.values()),
            "unique_values": len(attribute_counts),
            "coverage": sum(attribute_counts.values()) / total_edges if total_edges > 0 else 0,
            "top_values": sorted_counts[:top_k],
            "value_distribution": dict(sorted_counts[:top_k])
        } 