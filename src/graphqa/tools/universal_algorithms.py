"""
Universal Algorithm Selector - Intelligent algorithm selection for any graph dataset
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..data_structures import SchemaInfo, AttributeType

logger = logging.getLogger(__name__)


class UniversalAlgorithmInput(BaseModel):
    """Input schema for universal algorithm selection"""
    
    analysis_goal: str = Field(description="What you want to analyze or discover")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {}, description="Additional parameters"
    )
    performance_preference: Optional[str] = Field(
        default="balanced", description="Performance preference: fast, balanced, or comprehensive"
    )


class UniversalAlgorithmSelector(BaseTool):
    """Universal algorithm selector that chooses optimal analysis approaches for any dataset"""
    
    name: str = "universal_algorithm_selector"
    description: str = """🧠 UNIVERSAL ALGORITHM SELECTOR - INTELLIGENT ANALYSIS FOR ANY DATASET

Automatically selects the best analysis algorithms and approaches based on your dataset characteristics and analysis goals.
Works seamlessly with Amazon products, social networks, or any graph data.

🔧 SMART ALGORITHM SELECTION: Analyzes your dataset and chooses optimal algorithms for your specific goals.

🎯 ANALYSIS GOALS:

🔍 community_detection - Find groups and clusters in your data
   Format: {{"analysis_goal": "community_detection", "performance_preference": "balanced"}}
   Returns: Community detection algorithm recommendations and results

🌟 influential_nodes - Identify the most important nodes
   Format: {{"analysis_goal": "influential_nodes", "parameters": {{"top_k": 20}}}}
   Returns: Best centrality measures and most influential nodes

🛣️ shortest_paths - Find optimal paths between nodes
   Format: {{"analysis_goal": "shortest_paths", "parameters": {{"source": "node1", "target": "node2"}}}}
   Returns: Shortest path algorithms and actual paths

🔗 connectivity_analysis - Analyze network connectivity patterns
   Format: {{"analysis_goal": "connectivity_analysis"}}
   Returns: Connectivity algorithms and network structure analysis

📊 graph_similarity - Compare nodes or subgraphs for similarity
   Format: {{"analysis_goal": "graph_similarity", "parameters": {{"node_list": ["node1", "node2"]}}}}
   Returns: Similarity algorithms and comparison results

🌐 network_topology - Understand overall network structure
   Format: {{"analysis_goal": "network_topology"}}
   Returns: Topology analysis algorithms and structural insights

🎭 anomaly_detection - Find unusual patterns or outliers
   Format: {{"analysis_goal": "anomaly_detection"}}
   Returns: Anomaly detection algorithms and unusual nodes/patterns

📈 evolution_analysis - Analyze changes over time (if temporal data exists)
   Format: {{"analysis_goal": "evolution_analysis"}}
   Returns: Temporal analysis algorithms and trend detection

All algorithm selection automatically adapts to your dataset size, structure, and characteristics!
"""
    
    args_schema: type[BaseModel] = UniversalAlgorithmInput
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
    
    def _run(self, analysis_goal: str, parameters: Optional[Dict[str, Any]] = None, 
             performance_preference: str = "balanced") -> str:
        """Select and execute optimal algorithms for the analysis goal"""
        
        if parameters is None:
            parameters = {}
            
        start_time = time.time()
        self.logger.info(f"Selecting algorithms for: {analysis_goal} with preference: {performance_preference}")
        
        try:
            # Analyze dataset characteristics
            dataset_characteristics = self._analyze_dataset_characteristics()
            
            # Route to appropriate algorithm selector
            if analysis_goal == "community_detection":
                result = self._select_community_detection(parameters, performance_preference, dataset_characteristics)
            elif analysis_goal == "influential_nodes":
                result = self._select_influential_nodes(parameters, performance_preference, dataset_characteristics)
            elif analysis_goal == "shortest_paths":
                result = self._select_shortest_paths(parameters, performance_preference, dataset_characteristics)
            elif analysis_goal == "connectivity_analysis":
                result = self._select_connectivity_analysis(parameters, performance_preference, dataset_characteristics)
            elif analysis_goal == "graph_similarity":
                result = self._select_graph_similarity(parameters, performance_preference, dataset_characteristics)
            elif analysis_goal == "network_topology":
                result = self._select_network_topology(parameters, performance_preference, dataset_characteristics)
            elif analysis_goal == "anomaly_detection":
                result = self._select_anomaly_detection(parameters, performance_preference, dataset_characteristics)
            elif analysis_goal == "evolution_analysis":
                result = self._select_evolution_analysis(parameters, performance_preference, dataset_characteristics)
            else:
                return json.dumps({
                    "error": f"Unknown analysis goal: {analysis_goal}",
                    "available_goals": [
                        "community_detection", "influential_nodes", "shortest_paths",
                        "connectivity_analysis", "graph_similarity", "network_topology",
                        "anomaly_detection", "evolution_analysis"
                    ]
                })
            
            execution_time = time.time() - start_time
            result["execution_time_seconds"] = round(execution_time, 3)
            result["dataset_name"] = self.dataset_name
            result["dataset_characteristics"] = dataset_characteristics
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            self.logger.error(f"Algorithm selection failed: {str(e)}")
            return json.dumps({
                "error": f"Algorithm selection failed: {str(e)}",
                "analysis_goal": analysis_goal,
                "parameters": parameters
            })
    
    def _analyze_dataset_characteristics(self) -> Dict[str, Any]:
        """Analyze key characteristics of the dataset to inform algorithm selection"""
        characteristics = {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "is_multigraph": self.graph.is_multigraph()
        }
        
        # Size category
        node_count = characteristics["node_count"]
        if node_count < 100:
            characteristics["size_category"] = "small"
        elif node_count < 10000:
            characteristics["size_category"] = "medium"
        elif node_count < 100000:
            characteristics["size_category"] = "large"
        else:
            characteristics["size_category"] = "very_large"
        
        # Density
        if node_count > 1:
            characteristics["density"] = nx.density(self.graph)
            if characteristics["density"] > 0.1:
                characteristics["density_category"] = "dense"
            elif characteristics["density"] > 0.01:
                characteristics["density_category"] = "medium"
            else:
                characteristics["density_category"] = "sparse"
        
        # Degree distribution
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        if degrees:
            characteristics["avg_degree"] = sum(degrees) / len(degrees)
            characteristics["max_degree"] = max(degrees)
            characteristics["min_degree"] = min(degrees)
        
        # Connectivity
        if self.graph.is_directed():
            characteristics["connected_components"] = nx.number_weakly_connected_components(self.graph)
        else:
            characteristics["connected_components"] = nx.number_connected_components(self.graph)
        
        # Attribute richness
        if self.graph_schema:
            characteristics["node_attribute_count"] = len(self.graph_schema.node_attributes)
            characteristics["edge_attribute_count"] = len(self.graph_schema.edge_attributes)
        else:
            # Sample to estimate
            sample_nodes = list(self.graph.nodes(data=True))[:10]
            if sample_nodes:
                avg_attrs = sum(len(attrs) for _, attrs in sample_nodes) / len(sample_nodes)
                characteristics["estimated_node_attributes"] = avg_attrs
        
        return characteristics
    
    def _select_community_detection(self, params: Dict[str, Any], preference: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal community detection algorithms"""
        algorithms = []
        results = {}
        
        size_category = characteristics.get("size_category", "medium")
        is_directed = characteristics.get("is_directed", False)
        
        # Algorithm selection based on characteristics
        if size_category in ["small", "medium"]:
            if not is_directed:
                # Louvain for undirected graphs
                try:
                    import networkx.algorithms.community as nx_comm
                    communities = nx_comm.louvain_communities(self.graph)
                    algorithms.append("louvain")
                    results["louvain"] = {
                        **self._summarize_communities(communities),
                        "modularity": nx_comm.modularity(self.graph, communities)
                    }
                except:
                    pass
        
        # Greedy modularity (works for most cases)
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(self.graph)
            algorithms.append("greedy_modularity")
            results["greedy_modularity"] = {
                **self._summarize_communities(communities),
                "modularity": nx_comm.modularity(self.graph, communities)
            }
        except:
            pass
        
        # Label propagation (fast for large graphs)
        if size_category in ["large", "very_large"] or preference == "fast":
            try:
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.label_propagation_communities(self.graph)
                algorithms.append("label_propagation")
                results["label_propagation"] = self._summarize_communities(communities)
            except:
                pass
        
        return {
            "analysis_goal": "community_detection",
            "selected_algorithms": algorithms,
            "algorithm_results": results,
            "recommendation": self._get_community_recommendation(algorithms, results, characteristics)
        }
    
    def _summarize_communities(self, communities) -> Dict[str, Any]:
        """Smart summarization of community detection results to prevent token overflow"""
        community_list = [list(community) for community in communities]
        community_sizes = [len(community) for community in community_list]
        
        # Sort communities by size (largest first)
        communities_with_sizes = list(zip(community_list, community_sizes))
        communities_with_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Size distribution statistics
        size_counts = {}
        for size in community_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        # Smart sampling: show top 10 largest communities with limited members
        sample_communities = []
        for community, size in communities_with_sizes[:10]:
            sample_communities.append({
                "size": size,
                "sample_members": community[:5],  # Show only first 5 members
                "has_more": size > 5
            })
        
        return {
            "community_count": len(community_list),
            "size_distribution": {
                "largest_community": max(community_sizes) if community_sizes else 0,
                "smallest_community": min(community_sizes) if community_sizes else 0,
                "average_size": sum(community_sizes) / len(community_sizes) if community_sizes else 0,
                "median_size": sorted(community_sizes)[len(community_sizes)//2] if community_sizes else 0
            },
            "sample_communities": sample_communities,
            "total_communities_shown": min(10, len(community_list)),
            "note": f"Showing top 10 largest communities out of {len(community_list)} total (with max 5 members each to prevent token overflow)"
        }
    
    def _select_influential_nodes(self, params: Dict[str, Any], preference: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal algorithms for finding influential nodes"""
        algorithms = []
        results = {}
        top_k = params.get("top_k", 10)
        
        size_category = characteristics.get("size_category", "medium")
        
        # Degree centrality (always fast)
        degree_centrality = nx.degree_centrality(self.graph)
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
        algorithms.append("degree_centrality")
        results["degree_centrality"] = {
            "top_nodes": [{"node_id": node, "score": score} for node, score in top_degree],
            "computation_complexity": "O(V)"
        }
        
        # PageRank (good for most cases)
        if size_category != "very_large" or preference != "fast":
            try:
                pagerank = nx.pagerank(self.graph)
                top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]
                algorithms.append("pagerank")
                results["pagerank"] = {
                    "top_nodes": [{"node_id": node, "score": score} for node, score in top_pagerank],
                    "computation_complexity": "O(V + E)"
                }
            except:
                pass
        
        # Betweenness centrality (computationally expensive)
        if size_category in ["small", "medium"] and preference != "fast":
            try:
                betweenness = nx.betweenness_centrality(self.graph)
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_k]
                algorithms.append("betweenness_centrality")
                results["betweenness_centrality"] = {
                    "top_nodes": [{"node_id": node, "score": score} for node, score in top_betweenness],
                    "computation_complexity": "O(V^3)"
                }
            except:
                pass
        
        # Closeness centrality (medium complexity)
        if size_category in ["small", "medium"] and preference == "comprehensive":
            try:
                closeness = nx.closeness_centrality(self.graph)
                top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_k]
                algorithms.append("closeness_centrality")
                results["closeness_centrality"] = {
                    "top_nodes": [{"node_id": node, "score": score} for node, score in top_closeness],
                    "computation_complexity": "O(V^2)"
                }
            except:
                pass
        
        return {
            "analysis_goal": "influential_nodes",
            "selected_algorithms": algorithms,
            "algorithm_results": results,
            "recommendation": self._get_influence_recommendation(algorithms, results, characteristics)
        }
    
    def _select_shortest_paths(self, params: Dict[str, Any], preference: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal shortest path algorithms"""
        algorithms = []
        results = {}
        
        source = params.get("source")
        target = params.get("target")
        
        if not source or not target:
            return {"error": "Both 'source' and 'target' parameters required for shortest path analysis"}
        
        if source not in self.graph or target not in self.graph:
            return {"error": f"Source '{source}' or target '{target}' not found in graph"}
        
        # Single source shortest path
        try:
            if self.graph.is_directed():
                path = nx.shortest_path(self.graph, source, target)
                length = nx.shortest_path_length(self.graph, source, target)
            else:
                path = nx.shortest_path(self.graph, source, target)
                length = nx.shortest_path_length(self.graph, source, target)
            
            algorithms.append("dijkstra")
            results["dijkstra"] = {
                "path": path,
                "path_length": length,
                "algorithm": "Dijkstra's algorithm"
            }
        except nx.NetworkXNoPath:
            results["dijkstra"] = {"error": "No path exists between source and target"}
        except:
            results["dijkstra"] = {"error": "Could not compute shortest path"}
        
        # All shortest paths (if comprehensive analysis requested)
        if preference == "comprehensive":
            try:
                all_paths = list(nx.all_shortest_paths(self.graph, source, target))
                algorithms.append("all_shortest_paths")
                results["all_shortest_paths"] = {
                    "path_count": len(all_paths),
                    "paths": all_paths[:5],  # Limit to first 5 paths
                    "note": f"Found {len(all_paths)} shortest paths total"
                }
            except:
                pass
        
        return {
            "analysis_goal": "shortest_paths",
            "source": source,
            "target": target,
            "selected_algorithms": algorithms,
            "algorithm_results": results
        }
    
    def _select_connectivity_analysis(self, params: Dict[str, Any], preference: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Select algorithms for connectivity analysis"""
        algorithms = []
        results = {}
        
        # Connected components
        if self.graph.is_directed():
            # Weakly connected components
            weak_components = list(nx.weakly_connected_components(self.graph))
            algorithms.append("weakly_connected_components")
            results["weakly_connected_components"] = {
                "component_count": len(weak_components),
                "largest_component_size": max(len(comp) for comp in weak_components) if weak_components else 0,
                "component_sizes": [len(comp) for comp in weak_components]
            }
            
            # Strongly connected components
            strong_components = list(nx.strongly_connected_components(self.graph))
            algorithms.append("strongly_connected_components")
            results["strongly_connected_components"] = {
                "component_count": len(strong_components),
                "largest_component_size": max(len(comp) for comp in strong_components) if strong_components else 0,
                "component_sizes": [len(comp) for comp in strong_components]
            }
        else:
            # Connected components for undirected graphs
            components = list(nx.connected_components(self.graph))
            algorithms.append("connected_components")
            results["connected_components"] = {
                "component_count": len(components),
                "largest_component_size": max(len(comp) for comp in components) if components else 0,
                "component_sizes": [len(comp) for comp in components]
            }
        
        # Node connectivity (for smaller graphs)
        if characteristics.get("size_category") in ["small", "medium"] and preference != "fast":
            try:
                node_connectivity = nx.node_connectivity(self.graph)
                algorithms.append("node_connectivity")
                results["node_connectivity"] = {
                    "connectivity": node_connectivity,
                    "description": "Minimum number of nodes that need to be removed to disconnect the graph"
                }
            except:
                pass
        
        # Edge connectivity
        if characteristics.get("size_category") in ["small", "medium"] and preference == "comprehensive":
            try:
                edge_connectivity = nx.edge_connectivity(self.graph)
                algorithms.append("edge_connectivity")
                results["edge_connectivity"] = {
                    "connectivity": edge_connectivity,
                    "description": "Minimum number of edges that need to be removed to disconnect the graph"
                }
            except:
                pass
        
        return {
            "analysis_goal": "connectivity_analysis",
            "selected_algorithms": algorithms,
            "algorithm_results": results
        }
    
    def _select_graph_similarity(self, params: Dict[str, Any], preference: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Select algorithms for graph similarity analysis"""
        algorithms = []
        results = {}
        
        node_list = params.get("node_list", [])
        
        if len(node_list) < 2:
            return {"error": "At least 2 nodes required for similarity analysis"}
        
        # Structural similarity
        for i, node1 in enumerate(node_list):
            for node2 in node_list[i+1:]:
                if node1 in self.graph and node2 in self.graph:
                    # Jaccard similarity of neighborhoods
                    neighbors1 = set(self.graph.neighbors(node1))
                    neighbors2 = set(self.graph.neighbors(node2))
                    
                    if neighbors1 or neighbors2:
                        intersection = len(neighbors1 & neighbors2)
                        union = len(neighbors1 | neighbors2)
                        jaccard_sim = intersection / union if union > 0 else 0
                    else:
                        jaccard_sim = 1.0  # Both isolated
                    
                    pair_key = f"{node1}-{node2}"
                    if "structural_similarity" not in results:
                        results["structural_similarity"] = {}
                        algorithms.append("structural_similarity")
                    
                    results["structural_similarity"][pair_key] = {
                        "jaccard_similarity": jaccard_sim,
                        "common_neighbors": len(neighbors1 & neighbors2),
                        "total_unique_neighbors": len(neighbors1 | neighbors2)
                    }
        
        # Attribute similarity (if nodes have attributes)
        attribute_similarities = {}
        for i, node1 in enumerate(node_list):
            for node2 in node_list[i+1:]:
                if node1 in self.graph and node2 in self.graph:
                    attrs1 = dict(self.graph.nodes[node1])
                    attrs2 = dict(self.graph.nodes[node2])
                    
                    common_attrs = set(attrs1.keys()) & set(attrs2.keys())
                    if common_attrs:
                        matching_attrs = sum(1 for attr in common_attrs if attrs1[attr] == attrs2[attr])
                        attr_similarity = matching_attrs / len(common_attrs)
                        
                        pair_key = f"{node1}-{node2}"
                        attribute_similarities[pair_key] = {
                            "attribute_similarity": attr_similarity,
                            "common_attributes": len(common_attrs),
                            "matching_attributes": matching_attrs
                        }
        
        if attribute_similarities:
            algorithms.append("attribute_similarity")
            results["attribute_similarity"] = attribute_similarities
        
        return {
            "analysis_goal": "graph_similarity",
            "node_list": node_list,
            "selected_algorithms": algorithms,
            "algorithm_results": results
        }
    
    def _select_network_topology(self, params: Dict[str, Any], preference: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Select algorithms for network topology analysis"""
        algorithms = []
        results = {}
        
        # Basic topology metrics
        basic_metrics = {
            "nodes": characteristics["node_count"],
            "edges": characteristics["edge_count"],
            "density": characteristics.get("density", 0),
            "is_directed": characteristics["is_directed"]
        }
        algorithms.append("basic_topology")
        results["basic_topology"] = basic_metrics
        
        # Clustering coefficient
        if not characteristics["is_directed"] or characteristics.get("size_category") != "very_large":
            try:
                avg_clustering = nx.average_clustering(self.graph)
                algorithms.append("clustering_analysis")
                results["clustering_analysis"] = {
                    "average_clustering": avg_clustering,
                    "description": "Measure of how much nodes cluster together"
                }
            except:
                pass
        
        # Degree distribution
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        if degrees:
            from collections import Counter
            degree_dist = Counter(degrees)
            algorithms.append("degree_distribution")
            results["degree_distribution"] = {
                "min_degree": min(degrees),
                "max_degree": max(degrees),
                "average_degree": sum(degrees) / len(degrees),
                "degree_distribution": dict(degree_dist.most_common(10))
            }
        
        # Small world properties (for smaller graphs)
        if characteristics.get("size_category") in ["small", "medium"] and preference == "comprehensive":
            try:
                if not characteristics["is_directed"]:
                    # Check if graph is connected
                    if nx.is_connected(self.graph):
                        diameter = nx.diameter(self.graph)
                        avg_path_length = nx.average_shortest_path_length(self.graph)
                        algorithms.append("small_world_analysis")
                        results["small_world_analysis"] = {
                            "diameter": diameter,
                            "average_path_length": avg_path_length,
                            "clustering_coefficient": results.get("clustering_analysis", {}).get("average_clustering", 0)
                        }
            except:
                pass
        
        return {
            "analysis_goal": "network_topology",
            "selected_algorithms": algorithms,
            "algorithm_results": results
        }
    
    def _select_anomaly_detection(self, params: Dict[str, Any], preference: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Select algorithms for anomaly detection"""
        algorithms = []
        results = {}
        
        # Degree-based anomalies
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        if degrees:
            avg_degree = sum(degrees) / len(degrees)
            std_degree = (sum((d - avg_degree) ** 2 for d in degrees) / len(degrees)) ** 0.5
            
            # Nodes with unusually high or low degree
            anomalous_nodes = []
            for node in self.graph.nodes():
                node_degree = self.graph.degree(node)
                if abs(node_degree - avg_degree) > 2 * std_degree:
                    anomalous_nodes.append({
                        "node_id": node,
                        "degree": node_degree,
                        "anomaly_score": abs(node_degree - avg_degree) / std_degree if std_degree > 0 else 0,
                        "anomaly_type": "high_degree" if node_degree > avg_degree else "low_degree"
                    })
            
            algorithms.append("degree_anomaly_detection")
            results["degree_anomaly_detection"] = {
                "anomalous_nodes": sorted(anomalous_nodes, key=lambda x: x["anomaly_score"], reverse=True)[:10],
                "total_anomalies": len(anomalous_nodes),
                "threshold": f"{avg_degree:.2f} ± {2*std_degree:.2f}"
            }
        
        # Isolated nodes
        isolated_nodes = [node for node in self.graph.nodes() if self.graph.degree(node) == 0]
        if isolated_nodes:
            algorithms.append("isolation_detection")
            results["isolation_detection"] = {
                "isolated_nodes": isolated_nodes[:20],  # Limit output
                "total_isolated": len(isolated_nodes)
            }
        
        return {
            "analysis_goal": "anomaly_detection",
            "selected_algorithms": algorithms,
            "algorithm_results": results
        }
    
    def _select_evolution_analysis(self, params: Dict[str, Any], preference: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Select algorithms for evolution analysis (temporal data)"""
        algorithms = []
        results = {}
        
        # Check for temporal attributes
        temporal_attributes = []
        sample_nodes = list(self.graph.nodes(data=True))[:10]
        
        for _, attrs in sample_nodes:
            for attr_name, attr_value in attrs.items():
                if any(keyword in attr_name.lower() for keyword in ['time', 'date', 'created', 'updated', 'timestamp']):
                    temporal_attributes.append(attr_name)
        
        if temporal_attributes:
            algorithms.append("temporal_attribute_analysis")
            results["temporal_attribute_analysis"] = {
                "temporal_attributes_found": list(set(temporal_attributes)),
                "analysis": "Temporal attributes detected - time-based analysis possible"
            }
        else:
            results["temporal_analysis"] = {
                "message": "No obvious temporal attributes found in the dataset",
                "suggestion": "Consider adding timestamp information for evolution analysis"
            }
        
        return {
            "analysis_goal": "evolution_analysis",
            "selected_algorithms": algorithms,
            "algorithm_results": results
        }
    
    def _get_community_recommendation(self, algorithms: List[str], results: Dict[str, Any], characteristics: Dict[str, Any]) -> str:
        """Generate recommendation for community detection"""
        if not algorithms:
            return "No suitable community detection algorithms could be applied to this dataset."
        
        best_algorithm = algorithms[0]
        
        if "louvain" in algorithms:
            best_algorithm = "louvain"
            recommendation = "Louvain algorithm recommended for this undirected graph. It provides good modularity optimization."
        elif "greedy_modularity" in algorithms:
            best_algorithm = "greedy_modularity"
            recommendation = "Greedy modularity optimization recommended. Good balance of quality and performance."
        else:
            recommendation = f"Using {best_algorithm} algorithm. Consider the trade-offs between speed and quality."
        
        if best_algorithm in results:
            community_count = results[best_algorithm].get("community_count", 0)
            recommendation += f" Found {community_count} communities in your dataset."
        
        return recommendation
    
    def _get_influence_recommendation(self, algorithms: List[str], results: Dict[str, Any], characteristics: Dict[str, Any]) -> str:
        """Generate recommendation for influence analysis"""
        if not algorithms:
            return "No suitable influence algorithms could be applied to this dataset."
        
        size_category = characteristics.get("size_category", "medium")
        
        if size_category in ["small", "medium"] and "betweenness_centrality" in algorithms:
            return "Betweenness centrality recommended for identifying bridge nodes that control information flow."
        elif "pagerank" in algorithms:
            return "PageRank recommended for identifying globally influential nodes based on network structure."
        else:
            return "Degree centrality recommended as a fast measure of local influence and connectivity." 