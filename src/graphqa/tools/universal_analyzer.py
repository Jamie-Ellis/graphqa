"""
Universal Node Analyzer Tool - Deep analysis of individual nodes in any graph dataset
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import networkx as nx
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..data_structures import SchemaInfo, AttributeType

logger = logging.getLogger(__name__)


class UniversalAnalyzerInput(BaseModel):
    """Input schema for universal node analysis"""
    
    analysis_type: str = Field(description="Type of analysis to execute")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {}, description="Analysis parameters"
    )
    include_metrics: Optional[bool] = Field(
        default=True, description="Include detailed metrics"
    )
    include_insights: Optional[bool] = Field(
        default=True, description="Include AI-generated insights"
    )


class UniversalNodeAnalyzer(BaseTool):
    """Universal node analysis tool with deep insights for any graph dataset"""
    
    name: str = "universal_node_analyzer"
    description: str = """🔍 UNIVERSAL NODE ANALYSIS - DEEP DIVE FOR ANY DATASET

Perform comprehensive analysis of individual nodes with automatic schema adaptation.
Works seamlessly with Amazon products, social networks, or any graph data.

🎯 INTELLIGENT ANALYSIS: Automatically discovers node properties and relationships for deep insights.

🔍 UNIVERSAL ANALYSIS TYPES:

🎯 analyze_node - Complete node analysis with all available insights
   Format: {{"analysis_type": "analyze_node", "parameters": {{"node_id": "B00123ABCD"}}}}
   Returns: Full node details, connections, metrics, relationships, and insights

📈 node_metrics - Calculate centrality and importance metrics for a node
   Format: {{"analysis_type": "node_metrics", "parameters": {{"node_id": "B00123ABCD"}}}}
   Returns: Degree, PageRank, betweenness centrality, clustering coefficient

🔗 node_relationships - Analyze all relationships and connections
   Format: {{"analysis_type": "node_relationships", "parameters": {{"node_id": "B00123ABCD", "depth": 2}}}}
   Returns: Direct and indirect connections, relationship patterns, network position

🎭 node_similarity - Find nodes similar to the target node
   Format: {{"analysis_type": "node_similarity", "parameters": {{"node_id": "B00123ABCD", "top_k": 10}}}}
   Returns: Most similar nodes based on attributes and network position

🌟 node_influence - Analyze the node's influence and reach in the network
   Format: {{"analysis_type": "node_influence", "parameters": {{"node_id": "B00123ABCD"}}}}
   Returns: Influence metrics, reachability, and network impact analysis

🏷️ attribute_analysis - Deep dive into the node's attributes
   Format: {{"analysis_type": "attribute_analysis", "parameters": {{"node_id": "B00123ABCD"}}}}
   Returns: Detailed analysis of all node attributes with context and insights

🎯 PARAMETER REQUIREMENTS:
- All operations require: "analysis_type" (string)
- All operations require: "parameters" dict with "node_id" (string)
- Optional: "top_k" for similarity/influence analysis (default: 10)

All analysis automatically adapts to your dataset's schema. No hardcoded assumptions!
"""
    
    args_schema: type[BaseModel] = UniversalAnalyzerInput
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
    
    def _run(self, analysis_type: str, parameters: Optional[Dict[str, Any]] = None, 
             include_metrics: bool = True, include_insights: bool = True) -> str:
        """Execute universal node analysis"""
        
        if parameters is None:
            parameters = {}
            
        start_time = time.time()
        self.logger.info(f"Analyzing node: {analysis_type} with params: {parameters}")
        
        try:
            # Route to appropriate analysis handler
            if analysis_type == "analyze_node":
                result = self._analyze_node(parameters, include_metrics, include_insights)
            elif analysis_type == "node_metrics":
                result = self._compute_node_metrics(parameters)
            elif analysis_type == "node_relationships":
                result = self._analyze_node_relationships(parameters)
            elif analysis_type == "node_similarity":
                result = self._find_similar_nodes(parameters)
            elif analysis_type == "node_influence":
                result = self._analyze_node_influence(parameters)
            elif analysis_type == "attribute_analysis":
                result = self._analyze_node_attributes(parameters)
            else:
                return json.dumps({
                    "error": f"Unknown analysis type: {analysis_type}",
                    "available_types": [
                        "analyze_node", "node_metrics", "node_relationships",
                        "node_similarity", "node_influence", "attribute_analysis"
                    ]
                })
            
            execution_time = time.time() - start_time
            result["execution_time_seconds"] = round(execution_time, 3)
            result["dataset_name"] = self.dataset_name
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            self.logger.error(f"Node analysis failed: {str(e)}")
            return json.dumps({
                "error": f"Node analysis failed: {str(e)}",
                "analysis_type": analysis_type,
                "parameters": parameters
            })
    
    def _analyze_node(self, params: Dict[str, Any], include_metrics: bool, include_insights: bool) -> Dict[str, Any]:
        """Complete comprehensive node analysis"""
        node_id = params.get("node_id")
        
        if not node_id or node_id not in self.graph:
            return {"error": f"Node '{node_id}' not found in graph"}
        
        # Get node attributes
        node_attrs = dict(self.graph.nodes[node_id])
        
        analysis = {
            "analysis_type": "analyze_node",
            "node_id": node_id,
            "node_attributes": node_attrs,
            "basic_info": self._get_basic_node_info(node_id)
        }
        
        if include_metrics:
            analysis["metrics"] = self._compute_node_metrics({"node_id": node_id})["metrics"]
        
        # Relationship analysis
        analysis["relationships"] = self._get_relationship_summary(node_id)
        
        # Neighborhood analysis
        analysis["neighborhood"] = self._analyze_neighborhood(node_id)
        
        if include_insights:
            analysis["insights"] = self._generate_node_insights(node_id, node_attrs)
        
        return analysis
    
    def _compute_node_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute centrality and importance metrics for a node"""
        node_id = params.get("node_id")
        
        if not node_id or node_id not in self.graph:
            return {"error": f"Node '{node_id}' not found in graph"}
        
        metrics = {}
        
        # Basic metrics
        metrics["degree"] = self.graph.degree(node_id)
        metrics["in_degree"] = self.graph.in_degree(node_id) if self.graph.is_directed() else None
        metrics["out_degree"] = self.graph.out_degree(node_id) if self.graph.is_directed() else None
        
        # Centrality metrics (compute if graph is not too large)
        if self.graph.number_of_nodes() <= 10000:
            try:
                # Degree centrality
                degree_centrality = nx.degree_centrality(self.graph)
                metrics["degree_centrality"] = degree_centrality.get(node_id, 0)
                
                # PageRank
                pagerank = nx.pagerank(self.graph)
                metrics["pagerank"] = pagerank.get(node_id, 0)
                
                # Closeness centrality (for smaller graphs)
                if self.graph.number_of_nodes() <= 1000:
                    closeness = nx.closeness_centrality(self.graph)
                    metrics["closeness_centrality"] = closeness.get(node_id, 0)
                
            except Exception as e:
                metrics["centrality_error"] = f"Could not compute centrality: {str(e)}"
        
        # Clustering coefficient
        try:
            if not self.graph.is_directed():
                clustering = nx.clustering(self.graph, node_id)
                metrics["clustering_coefficient"] = clustering
        except:
            pass
        
        # Local metrics
        neighbors = list(self.graph.neighbors(node_id))
        metrics["neighbor_count"] = len(neighbors)
        
        if neighbors:
            # Average neighbor degree
            neighbor_degrees = [self.graph.degree(neighbor) for neighbor in neighbors]
            metrics["average_neighbor_degree"] = sum(neighbor_degrees) / len(neighbor_degrees)
        
        return {
            "analysis_type": "node_metrics",
            "node_id": node_id,
            "metrics": metrics
        }
    
    def _analyze_node_relationships(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all relationships and connections for a node"""
        node_id = params.get("node_id")
        depth = params.get("depth", 2)
        
        if not node_id or node_id not in self.graph:
            return {"error": f"Node '{node_id}' not found in graph"}
        
        relationships = {
            "analysis_type": "node_relationships",
            "node_id": node_id,
            "depth": depth
        }
        
        # Direct relationships
        direct_neighbors = {}
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            neighbor_attrs = dict(self.graph.nodes[neighbor])
            
            relationship_info = {
                "neighbor_id": neighbor,
                "neighbor_attributes": neighbor_attrs,
                "edge_data": edge_data,
                "neighbor_degree": self.graph.degree(neighbor)
            }
            
            direct_neighbors[neighbor] = relationship_info
        
        relationships["direct_neighbors"] = direct_neighbors
        relationships["direct_neighbor_count"] = len(direct_neighbors)
        
        # Multi-hop analysis
        if depth > 1:
            multi_hop = self._analyze_multi_hop_connections(node_id, depth)
            relationships["multi_hop_analysis"] = multi_hop
        
        # Relationship patterns
        relationships["relationship_patterns"] = self._analyze_relationship_patterns(node_id)
        
        return relationships
    
    def _find_similar_nodes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find nodes similar to the target node"""
        node_id = params.get("node_id")
        top_k = params.get("top_k", 10)
        similarity_type = params.get("similarity_type", "attribute")  # attribute, structural, hybrid
        
        if not node_id or node_id not in self.graph:
            return {"error": f"Node '{node_id}' not found in graph"}
        
        target_attrs = dict(self.graph.nodes[node_id])
        similar_nodes = []
        
        for other_node_id in self.graph.nodes():
            if other_node_id == node_id:
                continue
            
            similarity_score = 0
            
            if similarity_type in ["attribute", "hybrid"]:
                # Attribute similarity
                other_attrs = dict(self.graph.nodes[other_node_id])
                attr_similarity = self._compute_attribute_similarity(target_attrs, other_attrs)
                similarity_score += attr_similarity
            
            if similarity_type in ["structural", "hybrid"]:
                # Structural similarity
                structural_similarity = self._compute_structural_similarity(node_id, other_node_id)
                similarity_score += structural_similarity
                
                if similarity_type == "hybrid":
                    similarity_score /= 2  # Average of attribute and structural
            
            if similarity_score > 0:
                similar_nodes.append({
                    "node_id": other_node_id,
                    "similarity_score": similarity_score,
                    "attributes": dict(self.graph.nodes[other_node_id]),
                    "degree": self.graph.degree(other_node_id)
                })
        
        # Sort by similarity score
        similar_nodes.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "analysis_type": "node_similarity",
            "target_node": node_id,
            "similarity_type": similarity_type,
            "total_candidates": len(similar_nodes),
            "top_similar_nodes": similar_nodes[:top_k]
        }
    
    def _analyze_node_influence(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the node's influence and reach in the network"""
        node_id = params.get("node_id")
        
        if not node_id or node_id not in self.graph:
            return {"error": f"Node '{node_id}' not found in graph"}
        
        influence_analysis = {
            "analysis_type": "node_influence",
            "node_id": node_id
        }
        
        # Basic influence metrics
        degree = self.graph.degree(node_id)
        total_nodes = self.graph.number_of_nodes()
        
        influence_analysis["basic_influence"] = {
            "degree": degree,
            "degree_ratio": degree / total_nodes if total_nodes > 0 else 0,
            "is_hub": degree > (sum(self.graph.degree(n) for n in self.graph.nodes()) / total_nodes) * 2
        }
        
        # Reachability analysis
        reachable_nodes = set()
        try:
            # BFS to find reachable nodes (limited depth for performance)
            visited = set()
            queue = [(node_id, 0)]
            max_depth = 3
            
            while queue:
                current_node, depth = queue.pop(0)
                if depth >= max_depth or current_node in visited:
                    continue
                
                visited.add(current_node)
                reachable_nodes.add(current_node)
                
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
            
            influence_analysis["reachability"] = {
                "reachable_nodes_count": len(reachable_nodes),
                "reachability_ratio": len(reachable_nodes) / total_nodes if total_nodes > 0 else 0,
                "max_depth_analyzed": max_depth
            }
        except:
            influence_analysis["reachability"] = {"error": "Could not compute reachability"}
        
        # Network position analysis
        try:
            if self.graph.number_of_nodes() <= 1000:
                betweenness = nx.betweenness_centrality(self.graph)
                node_betweenness = betweenness.get(node_id, 0)
                avg_betweenness = sum(betweenness.values()) / len(betweenness)
                
                influence_analysis["network_position"] = {
                    "betweenness_centrality": node_betweenness,
                    "is_bridge_node": node_betweenness > avg_betweenness * 2,
                    "position_percentile": sum(1 for v in betweenness.values() if v < node_betweenness) / len(betweenness)
                }
        except:
            influence_analysis["network_position"] = {"error": "Could not compute network position"}
        
        return influence_analysis
    
    def _analyze_node_attributes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deep dive into the node's attributes with context"""
        node_id = params.get("node_id")
        
        if not node_id or node_id not in self.graph:
            return {"error": f"Node '{node_id}' not found in graph"}
        
        node_attrs = dict(self.graph.nodes[node_id])
        
        attribute_analysis = {
            "analysis_type": "attribute_analysis",
            "node_id": node_id,
            "total_attributes": len(node_attrs),
            "attribute_details": {}
        }
        
        # Analyze each attribute
        for attr_name, attr_value in node_attrs.items():
            attr_analysis = {
                "value": attr_value,
                "type": type(attr_value).__name__
            }
            
            # Compute attribute statistics across the graph
            all_values = []
            nodes_with_attr = 0
            
            for other_node, other_attrs in self.graph.nodes(data=True):
                if attr_name in other_attrs:
                    nodes_with_attr += 1
                    all_values.append(other_attrs[attr_name])
            
            attr_analysis["prevalence"] = {
                "nodes_with_attribute": nodes_with_attr,
                "coverage": nodes_with_attr / self.graph.number_of_nodes()
            }
            
            # Value analysis
            if attr_value is not None:
                try:
                    # Count occurrences of this value
                    same_value_count = sum(1 for v in all_values if v == attr_value)
                    attr_analysis["value_frequency"] = {
                        "count": same_value_count,
                        "frequency": same_value_count / len(all_values) if all_values else 0,
                        "is_rare": same_value_count < len(all_values) * 0.05,  # Less than 5%
                        "is_unique": same_value_count == 1
                    }
                except:
                    attr_analysis["value_frequency"] = {"error": "Could not compute frequency"}
            
            # Numeric analysis if applicable
            try:
                if isinstance(attr_value, (int, float)):
                    numeric_values = [v for v in all_values if isinstance(v, (int, float))]
                    if numeric_values:
                        avg_value = sum(numeric_values) / len(numeric_values)
                        attr_analysis["numeric_context"] = {
                            "average": avg_value,
                            "min": min(numeric_values),
                            "max": max(numeric_values),
                            "percentile": sum(1 for v in numeric_values if v < attr_value) / len(numeric_values),
                            "above_average": attr_value > avg_value
                        }
            except:
                pass
            
            attribute_analysis["attribute_details"][attr_name] = attr_analysis
        
        return attribute_analysis
    
    def _get_basic_node_info(self, node_id: str) -> Dict[str, Any]:
        """Get basic information about the node"""
        return {
            "total_attributes": len(self.graph.nodes[node_id]),
            "degree": self.graph.degree(node_id),
            "in_degree": self.graph.in_degree(node_id) if self.graph.is_directed() else None,
            "out_degree": self.graph.out_degree(node_id) if self.graph.is_directed() else None,
            "is_isolated": self.graph.degree(node_id) == 0
        }
    
    def _get_relationship_summary(self, node_id: str) -> Dict[str, Any]:
        """Get a summary of the node's relationships"""
        neighbors = list(self.graph.neighbors(node_id))
        
        # Analyze edge types if available
        edge_types = {}
        for neighbor in neighbors:
            edges = self.graph.get_edge_data(node_id, neighbor)
            if edges:
                for edge_data in edges.values():
                    edge_type = edge_data.get("type", "untyped")
                    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        return {
            "total_neighbors": len(neighbors),
            "edge_types": edge_types,
            "has_self_loop": self.graph.has_edge(node_id, node_id),
            "sample_neighbors": neighbors[:5]  # First 5 neighbors
        }
    
    def _analyze_neighborhood(self, node_id: str) -> Dict[str, Any]:
        """Analyze the local neighborhood around the node"""
        neighbors = list(self.graph.neighbors(node_id))
        
        if not neighbors:
            return {"message": "Node has no neighbors"}
        
        # Neighbor degrees
        neighbor_degrees = [self.graph.degree(neighbor) for neighbor in neighbors]
        
        # Common neighbors analysis
        common_neighbors = {}
        for i, neighbor1 in enumerate(neighbors):
            for neighbor2 in neighbors[i+1:]:
                if self.graph.has_edge(neighbor1, neighbor2):
                    pair = tuple(sorted([neighbor1, neighbor2]))
                    common_neighbors[pair] = True
        
        return {
            "neighbor_count": len(neighbors),
            "avg_neighbor_degree": sum(neighbor_degrees) / len(neighbor_degrees),
            "max_neighbor_degree": max(neighbor_degrees),
            "min_neighbor_degree": min(neighbor_degrees),
            "connected_neighbor_pairs": len(common_neighbors),
            "neighborhood_density": len(common_neighbors) / (len(neighbors) * (len(neighbors) - 1) / 2) if len(neighbors) > 1 else 0
        }
    
    def _generate_node_insights(self, node_id: str, node_attrs: Dict[str, Any]) -> List[str]:
        """Generate intelligent insights about the node"""
        insights = []
        
        degree = self.graph.degree(node_id)
        total_nodes = self.graph.number_of_nodes()
        avg_degree = sum(self.graph.degree(n) for n in self.graph.nodes()) / total_nodes
        
        # Degree-based insights
        if degree == 0:
            insights.append("This node is isolated with no connections to other nodes.")
        elif degree > avg_degree * 3:
            insights.append(f"This node is a major hub with {degree} connections, significantly above the network average of {avg_degree:.1f}.")
        elif degree > avg_degree:
            insights.append(f"This node is well-connected with {degree} connections, above the network average.")
        else:
            insights.append(f"This node has {degree} connections, which is below the network average.")
        
        # Attribute-based insights
        if node_attrs:
            insights.append(f"This node has {len(node_attrs)} attributes providing rich metadata.")
            
            # Look for unique or rare attributes
            for attr_name, attr_value in node_attrs.items():
                if isinstance(attr_value, str) and len(attr_value) > 50:
                    insights.append(f"The '{attr_name}' attribute contains detailed information ({len(attr_value)} characters).")
                elif isinstance(attr_value, (list, dict)):
                    insights.append(f"The '{attr_name}' attribute contains structured data ({type(attr_value).__name__}).")
        
        # Network position insights
        neighbors = list(self.graph.neighbors(node_id))
        if neighbors:
            neighbor_degrees = [self.graph.degree(neighbor) for neighbor in neighbors]
            avg_neighbor_degree = sum(neighbor_degrees) / len(neighbor_degrees)
            
            if avg_neighbor_degree > avg_degree * 2:
                insights.append("This node is connected to highly influential nodes in the network.")
            elif avg_neighbor_degree < avg_degree * 0.5:
                insights.append("This node is primarily connected to less central nodes in the network.")
        
        return insights
    
    def _compute_attribute_similarity(self, attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> float:
        """Compute similarity between two sets of attributes"""
        common_attrs = set(attrs1.keys()) & set(attrs2.keys())
        
        if not common_attrs:
            return 0.0
        
        matching_attrs = 0
        for attr in common_attrs:
            if attrs1[attr] == attrs2[attr]:
                matching_attrs += 1
        
        return matching_attrs / len(common_attrs)
    
    def _compute_structural_similarity(self, node1: str, node2: str) -> float:
        """Compute structural similarity between two nodes"""
        neighbors1 = set(self.graph.neighbors(node1))
        neighbors2 = set(self.graph.neighbors(node2))
        
        if not neighbors1 and not neighbors2:
            return 1.0  # Both isolated
        
        if not neighbors1 or not neighbors2:
            return 0.0  # One isolated, one not
        
        # Jaccard similarity of neighborhoods
        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_multi_hop_connections(self, node_id: str, max_depth: int) -> Dict[str, Any]:
        """Analyze multi-hop connections from the node"""
        connections_by_depth = {}
        visited = set()
        current_level = {node_id}
        
        for depth in range(1, max_depth + 1):
            next_level = set()
            for current_node in current_level:
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited and neighbor != node_id:
                        next_level.add(neighbor)
                        visited.add(neighbor)
            
            connections_by_depth[depth] = {
                "count": len(next_level),
                "nodes": list(next_level)[:10]  # Limit for performance
            }
            current_level = next_level
            
            if not current_level:
                break
        
        return {
            "max_depth_reached": max(connections_by_depth.keys()) if connections_by_depth else 0,
            "total_reachable": sum(info["count"] for info in connections_by_depth.values()),
            "connections_by_depth": connections_by_depth
        }
    
    def _analyze_relationship_patterns(self, node_id: str) -> Dict[str, Any]:
        """Analyze patterns in the node's relationships"""
        patterns = {}
        
        # Analyze edge attributes if available
        edge_attributes = {}
        neighbors = list(self.graph.neighbors(node_id))
        
        for neighbor in neighbors:
            edges = self.graph.get_edge_data(node_id, neighbor)
            if edges:
                for edge_data in edges.values():
                    for attr_name, attr_value in edge_data.items():
                        if attr_name not in edge_attributes:
                            edge_attributes[attr_name] = []
                        edge_attributes[attr_name].append(attr_value)
        
        # Summarize edge attribute patterns
        for attr_name, values in edge_attributes.items():
            try:
                from collections import Counter
                value_counts = Counter(values)
                patterns[f"edge_{attr_name}"] = {
                    "unique_values": len(value_counts),
                    "most_common": value_counts.most_common(3)
                }
            except:
                pass
        
        return patterns 