"""
Universal Graph Query Tool - Schema-aware querying for any graph dataset
"""

import json
import logging
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Union

import networkx as nx
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..data_structures import SchemaInfo, AttributeType

logger = logging.getLogger(__name__)


class UniversalQueryInput(BaseModel):
    """Input schema for universal graph queries"""
    
    query_type: str = Field(description="Type of query to execute")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {}, description="Query parameters"
    )
    limit: Optional[int] = Field(default=100, description="Result limit")
    sort_desc: Optional[bool] = Field(
        default=False, description="Sort results in descending order"
    )


class UniversalGraphQuery(BaseTool):
    """Universal graph querying tool that works with any graph dataset"""
    
    name: str = "universal_graph_query"
    description: str = """🎯 UNIVERSAL GRAPH QUERIES - ANY DATASET, ANY QUESTION

Execute powerful graph analysis on any dataset using dynamic schema discovery.
Works seamlessly with Amazon products, social networks, or any graph data.

🔧 SMART SCHEMA ADAPTATION: Automatically discovers and uses available attributes for querying.

📊 UNIVERSAL QUERY TYPES:

🔍 find_by_attribute - Find nodes by any attribute value
   Format: {{"query_type": "find_by_attribute", "parameters": {{"attribute": "category", "value": "Electronics"}}, "limit": 50}}
   Returns: Nodes matching the attribute value

🏷️ find_by_pattern - Find nodes matching attribute patterns
   Format: {{"query_type": "find_by_pattern", "parameters": {{"attribute": "title", "pattern": "smartphone"}}, "limit": 100}}
   Returns: Nodes with attribute values containing the pattern

🔗 find_by_relationship - Find nodes based on their connections
   Format: {{"query_type": "find_by_relationship", "parameters": {{"relationship_type": "also_bought", "min_connections": 5}}, "limit": 75}}
   Returns: Nodes with specified relationship patterns

📈 find_by_metric - Find nodes by computed metrics (degree, centrality, etc.)
   Format: {{"query_type": "find_by_metric", "parameters": {{"metric": "degree", "min_value": 10}}, "limit": 50}}
   Returns: Nodes meeting metric criteria

🎯 find_similar - Find nodes similar to a given node
   Format: {{"query_type": "find_similar", "parameters": {{"node_id": "B00123", "similarity_attributes": ["category", "brand"]}}, "limit": 20}}
   Returns: Nodes with similar attribute values

🌐 explore_neighborhood - Explore the neighborhood around a node
   Format: {{"query_type": "explore_neighborhood", "parameters": {{"node_id": "B00123", "depth": 2, "min_edge_weight": 0.5}}, "limit": 50}}
   Returns: Nodes within specified distance from the target node

📊 aggregate_by_attribute - Group and count by attribute values
   Format: {{"query_type": "aggregate_by_attribute", "parameters": {{"group_by": "category", "metric": "count"}}, "limit": 25}}
   Returns: Aggregated counts by attribute values

🎲 sample_nodes - Get a random sample of nodes
   Format: {{"query_type": "sample_nodes", "parameters": {{"criteria": {{"attribute": "price", "min": 10}}}}, "limit": 100}}
   Returns: Random sample meeting optional criteria

All queries automatically adapt to your dataset's schema. Attribute names are discovered dynamically!
"""
    
    args_schema: type[BaseModel] = UniversalQueryInput
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
    
    def _run(self, query_type: str, parameters: Optional[Dict[str, Any]] = None, 
             limit: int = 100, sort_desc: bool = False) -> str:
        """Execute a universal graph query"""
        
        if parameters is None:
            parameters = {}
            
        start_time = time.time()
        self.logger.info(f"Executing universal query: {query_type} with params: {parameters}")
        
        try:
            # Route to appropriate query handler
            if query_type == "find_by_attribute":
                result = self._find_by_attribute(parameters, limit, sort_desc)
            elif query_type == "find_by_pattern":
                result = self._find_by_pattern(parameters, limit, sort_desc)
            elif query_type == "find_by_relationship":
                result = self._find_by_relationship(parameters, limit, sort_desc)
            elif query_type == "find_by_metric":
                result = self._find_by_metric(parameters, limit, sort_desc)
            elif query_type == "find_similar":
                result = self._find_similar(parameters, limit, sort_desc)
            elif query_type == "explore_neighborhood":
                result = self._explore_neighborhood(parameters, limit, sort_desc)
            elif query_type == "aggregate_by_attribute":
                result = self._aggregate_by_attribute(parameters, limit, sort_desc)
            elif query_type == "sample_nodes":
                result = self._sample_nodes(parameters, limit, sort_desc)
            else:
                return json.dumps({
                    "error": f"Unknown query type: {query_type}",
                    "available_types": [
                        "find_by_attribute", "find_by_pattern", "find_by_relationship",
                        "find_by_metric", "find_similar", "explore_neighborhood", 
                        "aggregate_by_attribute", "sample_nodes"
                    ]
                })
            
            execution_time = time.time() - start_time
            result["execution_time_seconds"] = round(execution_time, 3)
            result["dataset_name"] = self.dataset_name
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            return json.dumps({
                "error": f"Query execution failed: {str(e)}",
                "query_type": query_type,
                "parameters": parameters
            })
    
    def _find_by_attribute(self, params: Dict[str, Any], limit: int, sort_desc: bool) -> Dict[str, Any]:
        """Find nodes by attribute value"""
        attribute = params.get("attribute")
        value = params.get("value")
        
        if not attribute or value is None:
            return {"error": "Both 'attribute' and 'value' parameters required"}
        
        matches = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attribute in attrs and attrs[attribute] == value:
                matches.append({
                    "node_id": node_id,
                    "attributes": dict(attrs),
                    "degree": self.graph.degree(node_id)
                })
        
        # Sort by degree if requested
        if sort_desc:
            matches.sort(key=lambda x: x.get("degree", 0), reverse=True)
        
        return {
            "query_type": "find_by_attribute",
            "attribute": attribute,
            "value": value,
            "total_matches": len(matches),
            "results": matches[:limit]
        }
    
    def _find_by_pattern(self, params: Dict[str, Any], limit: int, sort_desc: bool) -> Dict[str, Any]:
        """Find nodes by attribute pattern matching"""
        attribute = params.get("attribute")
        pattern = params.get("pattern", "").lower()
        
        if not attribute or not pattern:
            return {"error": "Both 'attribute' and 'pattern' parameters required"}
        
        matches = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attribute in attrs:
                attr_value = str(attrs[attribute]).lower()
                if pattern in attr_value:
                    matches.append({
                        "node_id": node_id,
                        "attribute_value": attrs[attribute],
                        "attributes": dict(attrs),
                        "degree": self.graph.degree(node_id)
                    })
        
        if sort_desc:
            matches.sort(key=lambda x: x.get("degree", 0), reverse=True)
        
        return {
            "query_type": "find_by_pattern",
            "attribute": attribute,
            "pattern": pattern,
            "total_matches": len(matches),
            "results": matches[:limit]
        }
    
    def _find_by_relationship(self, params: Dict[str, Any], limit: int, sort_desc: bool) -> Dict[str, Any]:
        """Find nodes based on relationship patterns"""
        relationship_type = params.get("relationship_type")
        min_connections = params.get("min_connections", 1)
        
        matches = []
        for node_id in self.graph.nodes():
            # Count relationships of specified type
            connection_count = 0
            for neighbor in self.graph.neighbors(node_id):
                edges = self.graph.get_edge_data(node_id, neighbor)
                if edges:
                    for edge_data in edges.values():
                        if relationship_type is None or edge_data.get("type") == relationship_type:
                            connection_count += 1
            
            if connection_count >= min_connections:
                node_attrs = dict(self.graph.nodes[node_id])
                matches.append({
                    "node_id": node_id,
                    "connection_count": connection_count,
                    "attributes": node_attrs,
                    "degree": self.graph.degree(node_id)
                })
        
        if sort_desc:
            matches.sort(key=lambda x: x.get("connection_count", 0), reverse=True)
        
        return {
            "query_type": "find_by_relationship",
            "relationship_type": relationship_type,
            "min_connections": min_connections,
            "total_matches": len(matches),
            "results": matches[:limit]
        }
    
    def _find_by_metric(self, params: Dict[str, Any], limit: int, sort_desc: bool) -> Dict[str, Any]:
        """Find nodes by computed graph metrics"""
        metric = params.get("metric", "degree")
        min_value = params.get("min_value", 0)
        max_value = params.get("max_value")
        
        matches = []
        
        if metric == "degree":
            for node_id in self.graph.nodes():
                degree = self.graph.degree(node_id)
                if degree >= min_value and (max_value is None or degree <= max_value):
                    matches.append({
                        "node_id": node_id,
                        "metric_value": degree,
                        "attributes": dict(self.graph.nodes[node_id])
                    })
        elif metric == "betweenness_centrality":
            centrality = nx.betweenness_centrality(self.graph)
            for node_id, centrality_score in centrality.items():
                if centrality_score >= min_value and (max_value is None or centrality_score <= max_value):
                    matches.append({
                        "node_id": node_id,
                        "metric_value": centrality_score,
                        "attributes": dict(self.graph.nodes[node_id])
                    })
        elif metric == "pagerank":
            pagerank = nx.pagerank(self.graph)
            for node_id, pr_score in pagerank.items():
                if pr_score >= min_value and (max_value is None or pr_score <= max_value):
                    matches.append({
                        "node_id": node_id,
                        "metric_value": pr_score,
                        "attributes": dict(self.graph.nodes[node_id])
                    })
        
        if sort_desc:
            matches.sort(key=lambda x: x.get("metric_value", 0), reverse=True)
        
        return {
            "query_type": "find_by_metric",
            "metric": metric,
            "min_value": min_value,
            "max_value": max_value,
            "total_matches": len(matches),
            "results": matches[:limit]
        }
    
    def _find_similar(self, params: Dict[str, Any], limit: int, sort_desc: bool) -> Dict[str, Any]:
        """Find nodes similar to a given node"""
        node_id = params.get("node_id")
        similarity_attributes = params.get("similarity_attributes", [])
        
        if not node_id or node_id not in self.graph:
            return {"error": f"Node '{node_id}' not found in graph"}
        
        target_attrs = dict(self.graph.nodes[node_id])
        matches = []
        
        for other_node_id in self.graph.nodes():
            if other_node_id == node_id:
                continue
                
            other_attrs = dict(self.graph.nodes[other_node_id])
            similarity_score = 0
            
            if similarity_attributes:
                # Use specified attributes for similarity
                matching_attrs = 0
                for attr in similarity_attributes:
                    if (attr in target_attrs and attr in other_attrs and 
                        target_attrs[attr] == other_attrs[attr]):
                        matching_attrs += 1
                similarity_score = matching_attrs / len(similarity_attributes) if similarity_attributes else 0
            else:
                # Use all common attributes
                common_attrs = set(target_attrs.keys()) & set(other_attrs.keys())
                if common_attrs:
                    matching_attrs = sum(1 for attr in common_attrs 
                                       if target_attrs[attr] == other_attrs[attr])
                    similarity_score = matching_attrs / len(common_attrs)
            
            if similarity_score > 0:
                matches.append({
                    "node_id": other_node_id,
                    "similarity_score": similarity_score,
                    "attributes": other_attrs
                })
        
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "query_type": "find_similar",
            "target_node": node_id,
            "similarity_attributes": similarity_attributes,
            "total_matches": len(matches),
            "results": matches[:limit]
        }
    
    def _explore_neighborhood(self, params: Dict[str, Any], limit: int, sort_desc: bool) -> Dict[str, Any]:
        """Explore the neighborhood around a node"""
        node_id = params.get("node_id")
        depth = params.get("depth", 1)
        min_edge_weight = params.get("min_edge_weight", 0)
        
        if not node_id or node_id not in self.graph:
            return {"error": f"Node '{node_id}' not found in graph"}
        
        # BFS to find neighbors within depth
        visited = set()
        current_level = {node_id}
        all_neighbors = []
        
        for level in range(depth + 1):
            if level == 0:
                continue
                
            next_level = set()
            for current_node in current_level:
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        # Check edge weight if specified
                        edges = self.graph.get_edge_data(current_node, neighbor)
                        max_weight = 0
                        if edges:
                            for edge_data in edges.values():
                                weight = edge_data.get("weight", 1)
                                max_weight = max(max_weight, weight)
                        
                        if max_weight >= min_edge_weight:
                            neighbor_attrs = dict(self.graph.nodes[neighbor])
                            all_neighbors.append({
                                "node_id": neighbor,
                                "distance": level,
                                "edge_weight": max_weight,
                                "attributes": neighbor_attrs,
                                "degree": self.graph.degree(neighbor)
                            })
                            next_level.add(neighbor)
                            visited.add(neighbor)
            
            current_level = next_level
        
        if sort_desc:
            all_neighbors.sort(key=lambda x: x.get("degree", 0), reverse=True)
        
        return {
            "query_type": "explore_neighborhood",
            "center_node": node_id,
            "depth": depth,
            "min_edge_weight": min_edge_weight,
            "total_neighbors": len(all_neighbors),
            "results": all_neighbors[:limit]
        }
    
    def _aggregate_by_attribute(self, params: Dict[str, Any], limit: int, sort_desc: bool) -> Dict[str, Any]:
        """Group and aggregate nodes by attribute values"""
        group_by = params.get("group_by")
        metric = params.get("metric", "count")
        
        if not group_by:
            return {"error": "'group_by' parameter required"}
        
        groups = defaultdict(list)
        
        # Group nodes by attribute value
        for node_id, attrs in self.graph.nodes(data=True):
            if group_by in attrs:
                value = attrs[group_by]
                groups[value].append(node_id)
        
        # Calculate aggregated metrics
        results = []
        for group_value, node_list in groups.items():
            if metric == "count":
                metric_value = len(node_list)
            elif metric == "avg_degree":
                degrees = [self.graph.degree(node) for node in node_list]
                metric_value = sum(degrees) / len(degrees) if degrees else 0
            elif metric == "total_degree":
                metric_value = sum(self.graph.degree(node) for node in node_list)
            else:
                metric_value = len(node_list)  # Default to count
            
            results.append({
                "group_value": group_value,
                "metric_value": metric_value,
                "node_count": len(node_list),
                "sample_nodes": node_list[:3]  # Show first 3 nodes as examples
            })
        
        # Sort by metric value
        results.sort(key=lambda x: x["metric_value"], reverse=sort_desc)
        
        return {
            "query_type": "aggregate_by_attribute",
            "group_by": group_by,
            "metric": metric,
            "total_groups": len(results),
            "results": results[:limit]
        }
    
    def _sample_nodes(self, params: Dict[str, Any], limit: int, sort_desc: bool) -> Dict[str, Any]:
        """Get a random sample of nodes, optionally with criteria"""
        criteria = params.get("criteria", {})
        
        # Filter nodes based on criteria
        eligible_nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            meets_criteria = True
            
            for attr, conditions in criteria.items():
                if attr not in attrs:
                    meets_criteria = False
                    break
                
                attr_value = attrs[attr]
                
                # Handle different condition types
                if isinstance(conditions, dict):
                    if "min" in conditions and attr_value < conditions["min"]:
                        meets_criteria = False
                        break
                    if "max" in conditions and attr_value > conditions["max"]:
                        meets_criteria = False
                        break
                    if "equals" in conditions and attr_value != conditions["equals"]:
                        meets_criteria = False
                        break
                else:
                    # Direct value comparison
                    if attr_value != conditions:
                        meets_criteria = False
                        break
            
            if meets_criteria:
                eligible_nodes.append({
                    "node_id": node_id,
                    "attributes": dict(attrs),
                    "degree": self.graph.degree(node_id)
                })
        
        # Random sample
        import random
        if len(eligible_nodes) > limit:
            sampled = random.sample(eligible_nodes, limit)
        else:
            sampled = eligible_nodes
        
        if sort_desc:
            sampled.sort(key=lambda x: x.get("degree", 0), reverse=True)
        
        return {
            "query_type": "sample_nodes",
            "criteria": criteria,
            "total_eligible": len(eligible_nodes),
            "sample_size": len(sampled),
            "results": sampled
        } 