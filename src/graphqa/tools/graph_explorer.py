"""
Universal Graph Explorer Tool

Schema-agnostic graph exploration and search tool that works with any graph structure.
Provides schema discovery, flexible search, and adaptive querying capabilities.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
from collections import Counter
import networkx as nx
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..data_structures import SchemaInfo, AttributeInfo, AttributeType, AnalysisResult

logger = logging.getLogger(__name__)


class GraphExplorerInput(BaseModel):
    """Input schema for graph explorer operations"""
    operation: str = Field(description="The operation to perform")
    attribute: Optional[str] = Field(default=None, description="Attribute name for searches")
    value: Optional[Union[str, int, float]] = Field(default=None, description="Search value")
    values: Optional[List[str]] = Field(default=None, description="Multiple values for search")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Multi-attribute filters")
    query: Optional[str] = Field(default=None, description="Text search query")
    fields: Optional[List[str]] = Field(default=None, description="Fields to search in")
    min_value: Optional[Union[int, float]] = Field(default=None, description="Minimum value for range")
    max_value: Optional[Union[int, float]] = Field(default=None, description="Maximum value for range")
    limit: Optional[int] = Field(default=50, description="Maximum results to return")
    node_id: Optional[str] = Field(default=None, description="Node ID for neighbor search")
    depth: Optional[int] = Field(default=1, description="Depth for neighbor search")


class UniversalGraphExplorer(BaseTool):
    """
    Universal graph exploration tool with automatic schema discovery.
    
    This tool adapts to any graph structure by first discovering available
    attributes and then providing targeted search and analysis capabilities.
    """
    
    name: str = "graph_explorer"
    description: str = """🔍 UNIVERSAL GRAPH EXPLORER - SCHEMA DISCOVERY & SEARCH

Discover dataset schema and perform flexible searches on any graph structure.
Automatically adapts to available attributes and data types.

📊 SCHEMA DISCOVERY:
- discover_schema: Get complete dataset overview
  Format: {"operation": "discover_schema"}
- search_schema_by_query: Find relevant attributes using natural language  
  Format: {"operation": "search_schema_by_query", "query": "find price-related attributes"}
- sample_data: See example nodes and edges
  Format: {"operation": "sample_data", "limit": 10}

🔍 SEARCH OPERATIONS:
- find_by_attribute: Search by specific field value
  Format: {"operation": "find_by_attribute", "attribute": "category", "value": "Electronics"}
- search_text: Text search across multiple fields (use "query" parameter)
  Format: {"operation": "search_text", "query": "smartphone", "fields": ["title", "description"], "limit": 50}
- filter_nodes: Multi-attribute filtering
  Format: {"operation": "filter_nodes", "filters": {"brand": "Apple", "price": 100}, "limit": 25}
- range_search: Numeric range queries
  Format: {"operation": "range_search", "attribute": "price", "min_value": 10, "max_value": 100}

📈 ATTRIBUTE ANALYSIS:
- get_node_attributes: List all node attributes
  Format: {"operation": "get_node_attributes"}
- get_edge_attributes: List all edge attributes  
  Format: {"operation": "get_edge_attributes"}
- attribute_analysis: Analyze specific attribute
  Format: {"operation": "attribute_analysis", "attribute": "price"}
- count_by_attribute: Count nodes by attribute values
  Format: {"operation": "count_by_attribute", "attribute": "category"}
- top_values: Get most common values for an attribute
  Format: {"operation": "top_values", "attribute": "brand", "limit": 10}

🌐 RELATIONSHIP ANALYSIS:
- get_neighbors: Find neighboring nodes
  Format: {"operation": "get_neighbors", "node_id": "B00123", "depth": 2, "limit": 20}
- find_by_values: Find nodes matching multiple values
  Format: {"operation": "find_by_values", "attribute": "category", "values": ["Electronics", "Books"]}

CRITICAL: Always use "query" parameter for search_text, not "text"!"""

    args_schema: type[BaseModel] = GraphExplorerInput
    graph: nx.MultiDiGraph = None
    graph_schema: Optional[SchemaInfo] = None
    dataset_name: str = "Unknown"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, graph: nx.MultiDiGraph, dataset_name: str = "Unknown", **kwargs):
        """Initialize the graph explorer with a graph"""
        # Initialize with the graph and dataset_name as field values
        super().__init__(
            graph=graph,
            dataset_name=dataset_name,
            graph_schema=None,
            **kwargs
        )
        # Set logger using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'logger', logging.getLogger(f"{__name__}.{self.__class__.__name__}"))
        
        # Initialize embedding-based schema search
        object.__setattr__(self, 'schema_searcher', None)
        self._init_schema_search()

    def _run(
        self,
        operation: str,
        attribute: Optional[str] = None,
        value: Optional[Union[str, int, float]] = None,
        values: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        fields: Optional[List[str]] = None,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        limit: int = 50,
        node_id: Optional[str] = None,
        depth: int = 1,
    ) -> Dict[str, Any]:
        """Execute graph exploration operations"""
        
        start_time = time.time()
        
        try:
            # Ensure schema is discovered
            if self.graph_schema is None:
                self._discover_schema()
            
            # Route to appropriate operation
            if operation == "discover_schema":
                result = self._get_complete_schema()
            elif operation == "search_schema_by_query":
                result = self._search_schema_by_query(query)
            elif operation == "get_node_attributes":
                result = self._get_node_attribute_list()
            elif operation == "get_edge_attributes":
                result = self._get_edge_attribute_list()
            elif operation == "sample_data":
                result = self._get_sample_data(limit)
            elif operation == "attribute_analysis":
                result = self._analyze_attribute(attribute)
            elif operation == "find_by_attribute":
                result = self._find_by_attribute(attribute, value, limit)
            elif operation == "search_text":
                result = self._search_text(query, fields, limit)
            elif operation == "filter_nodes":
                result = self._filter_nodes(filters, limit)
            elif operation == "range_search":
                result = self._range_search(attribute, min_value, max_value, limit)
            elif operation == "find_by_values":
                result = self._find_by_values(attribute, values, limit)
            elif operation == "count_by_attribute":
                result = self._count_by_attribute(attribute)
            elif operation == "top_values":
                result = self._get_top_values(attribute, limit)
            elif operation == "get_neighbors":
                result = self._get_neighbors(node_id, depth, limit)
            elif operation == "attribute_distribution":
                result = self._get_attribute_distribution(attribute)
            else:
                result = {"error": f"Unknown operation: {operation}"}
            
            # Add execution metadata
            result["execution_time"] = time.time() - start_time
            result["operation"] = operation
            
            return result
            
        except Exception as e:
            self.logger.error(f"Graph explorer operation failed: {e}")
            return {
                "error": f"Operation '{operation}' failed: {str(e)}",
                "execution_time": time.time() - start_time
            }

    def _init_schema_search(self):
        """Initialize embedding-based schema search with graceful fallback"""
        try:
            # Only initialize if we have a schema
            if hasattr(self, 'graph_schema') and self.graph_schema is not None:
                # Check if we have any attributes to search
                total_attrs = len(self.graph_schema.node_attributes) + len(self.graph_schema.edge_attributes)
                if total_attrs == 0:
                    self.logger.debug("No attributes available for schema search")
                    return
                
                from ..schema.embedder import SchemaEmbedder
                from ..schema.search_engine import QuerySchemaSearcher
                
                embedder = SchemaEmbedder()
                embedder.initialize_schema_embeddings(self.graph_schema)
                
                object.__setattr__(self, 'schema_searcher', QuerySchemaSearcher(embedder))
                self.logger.info("✅ Embedding-based schema search initialized")
            else:
                self.logger.debug("Schema not available yet for embedding initialization")
                
        except Exception as e:
            self.logger.warning(f"Could not initialize embedding search: {e}")
            self.logger.info("Schema search will fall back to basic exploration")
            object.__setattr__(self, 'schema_searcher', None)

    def _discover_schema(self):
        """Discover and cache schema information"""
        self.logger.info("Discovering graph schema...")
        
        self.graph_schema = SchemaInfo(
            dataset_name=self.dataset_name,
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
            is_directed=self.graph.is_directed()
        )
        
        # Analyze node attributes
        self.graph_schema.node_attributes = self._analyze_node_attributes()
        
        # Analyze edge attributes
        self.graph_schema.edge_attributes = self._analyze_edge_attributes()
        
        # Generate recommendations
        self.graph_schema.recommended_searches = self._generate_search_recommendations()

    def _analyze_node_attributes(self) -> Dict[str, AttributeInfo]:
        """Analyze all node attributes"""
        attributes = {}
        
        # Sample nodes for analysis
        sample_size = min(1000, self.graph.number_of_nodes())
        sample_nodes = list(self.graph.nodes(data=True))[:sample_size]
        
        # Collect all attribute names
        all_attrs = set()
        for node_id, attrs in sample_nodes:
            all_attrs.update(attrs.keys())
        
        # Analyze each attribute
        for attr_name in all_attrs:
            values = [attrs.get(attr_name) for node_id, attrs in sample_nodes 
                     if attr_name in attrs and attrs[attr_name] is not None]
            
            if not values:
                continue
            
            attr_info = self._create_attribute_info(attr_name, values, len(sample_nodes))
            attributes[attr_name] = attr_info
        
        return attributes

    def _analyze_edge_attributes(self) -> Dict[str, AttributeInfo]:
        """Analyze all edge attributes"""
        attributes = {}
        
        # Sample edges for analysis
        sample_size = min(1000, self.graph.number_of_edges())
        sample_edges = list(self.graph.edges(data=True))[:sample_size]
        
        # Collect all attribute names
        all_attrs = set()
        for u, v, attrs in sample_edges:
            all_attrs.update(attrs.keys())
        
        # Analyze each attribute
        for attr_name in all_attrs:
            values = [attrs.get(attr_name) for u, v, attrs in sample_edges
                     if attr_name in attrs and attrs[attr_name] is not None]
            
            if not values:
                continue
            
            attr_info = self._create_attribute_info(attr_name, values, len(sample_edges))
            attributes[attr_name] = attr_info
        
        return attributes

    def _create_attribute_info(self, name: str, values: List[Any], total_count: int) -> AttributeInfo:
        """Create AttributeInfo from values"""
        # Handle unhashable types (lists, dicts) gracefully
        try:
            unique_values = list(set(values))
        except TypeError:
            # Handle unhashable types by converting to strings for uniqueness check
            self.logger.warning(f"Failed to analyze attribute {name}: unhashable type")
            str_values = [str(v) for v in values]
            unique_values = list(set(str_values))[:10]  # Limit sample to prevent overflow
        
        attr_type = self._infer_attribute_type(values)
        
        attr_info = AttributeInfo(
            name=name,
            attribute_type=attr_type,
            coverage=len(values) / total_count,
            unique_count=len(unique_values),
            sample_values=unique_values[:10]  # Limit to prevent LLM overflow
        )
        
        # Add type-specific information
        if attr_type == AttributeType.NUMERIC:
            numeric_values = [float(v) for v in values if self._is_numeric(v)]
            if numeric_values:
                attr_info.stats = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "avg": sum(numeric_values) / len(numeric_values)
                }
        elif attr_type == AttributeType.CATEGORICAL:
            if len(unique_values) <= 20:
                attr_info.categories = unique_values
            else:
                # High cardinality - show top categories
                try:
                    value_counts = Counter(values)
                    attr_info.categories = [item for item, count in value_counts.most_common(10)]
                    attr_info.is_high_cardinality = True
                except TypeError:
                    # Handle unhashable types in Counter
                    self.logger.warning(f"Failed to count values for attribute {name}: unhashable type")
                    str_value_counts = Counter(str(v) for v in values)
                    attr_info.categories = [item for item, count in str_value_counts.most_common(10)]
                    attr_info.is_high_cardinality = True
        
        return attr_info

    def _infer_attribute_type(self, values: List[Any]) -> AttributeType:
        """Infer attribute type from sample values"""
        if not values:
            return AttributeType.UNKNOWN
        
        # Check numeric
        numeric_count = sum(1 for v in values[:20] if self._is_numeric(v))
        if numeric_count > len(values[:20]) * 0.8:
            return AttributeType.NUMERIC
        
        # Check boolean
        bool_values = {"true", "false", "1", "0", "yes", "no"}
        if all(str(v).lower() in bool_values for v in values[:10]):
            return AttributeType.BOOLEAN
        
        # Check list
        if any(isinstance(v, (list, tuple)) for v in values[:5]):
            return AttributeType.LIST
        
        # Categorical vs text
        try:
            unique_ratio = len(set(values)) / len(values)
        except TypeError:
            # Handle unhashable types
            unique_ratio = len(set(str(v) for v in values)) / len(values)
        avg_length = sum(len(str(v)) for v in values[:20]) / min(20, len(values))
        
        if unique_ratio < 0.5 and avg_length < 50:
            return AttributeType.CATEGORICAL
        else:
            return AttributeType.TEXT

    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _get_complete_schema(self) -> Dict[str, Any]:
        """Get complete schema information"""
        return {
            "dataset_name": self.graph_schema.dataset_name,
            "graph_info": {
                "node_count": self.graph_schema.node_count,
                "edge_count": self.graph_schema.edge_count,
                "is_directed": self.graph_schema.is_directed
            },
            "node_attributes": {
                name: {
                    "type": info.attribute_type.value,
                    "coverage": info.coverage,
                    "unique_count": info.unique_count,
                    "sample_values": info.sample_values,
                    "stats": info.stats,
                    "categories": info.categories,
                    "is_high_cardinality": info.is_high_cardinality
                }
                for name, info in self.graph_schema.node_attributes.items()
            },
            "edge_attributes": {
                name: {
                    "type": info.attribute_type.value,
                    "coverage": info.coverage,
                    "unique_count": info.unique_count,
                    "sample_values": info.sample_values
                }
                for name, info in self.graph_schema.edge_attributes.items()
            },
            "search_recommendations": self.graph_schema.recommended_searches,
            "searchable_attributes": self.graph_schema.get_searchable_attributes(),
            "numeric_attributes": self.graph_schema.get_numeric_attributes(),
            "categorical_attributes": self.graph_schema.get_categorical_attributes()
        }

    def _find_by_attribute(self, attribute: str, value: Any, limit: int) -> Dict[str, Any]:
        """Find nodes by attribute value"""
        if not attribute:
            return {"error": "Attribute name is required"}
        
        matches = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attribute in attrs and attrs[attribute] == value:
                matches.append({
                    "node_id": node_id,
                    "attributes": attrs
                })
                if len(matches) >= limit:
                    break
        
        return {
            "results": matches,
            "total_found": len(matches),
            "search_criteria": {"attribute": attribute, "value": value},
            "has_more": len(matches) == limit
        }

    def _search_text(self, query: str, fields: Optional[List[str]], limit: int) -> Dict[str, Any]:
        """Search for text across specified fields"""
        if not query:
            return {"error": "Search query is required"}
        
        if not fields:
            # Auto-detect text fields
            fields = [name for name, info in self.graph_schema.node_attributes.items()
                     if info.attribute_type == AttributeType.TEXT]
        
        query_lower = query.lower()
        matches = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            matched_fields = []
            for field in fields:
                if field in attrs and isinstance(attrs[field], str):
                    if query_lower in attrs[field].lower():
                        matched_fields.append(field)
            
            if matched_fields:
                matches.append({
                    "node_id": node_id,
                    "attributes": attrs,
                    "matched_fields": matched_fields,
                    "match_score": len(matched_fields)
                })
                
                if len(matches) >= limit:
                    break
        
        # Sort by match score
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        
        return {
            "results": matches,
            "total_found": len(matches),
            "search_query": query,
            "searched_fields": fields
        }

    def _count_by_attribute(self, attribute: str) -> Dict[str, Any]:
        """Count nodes by attribute values"""
        if not attribute:
            return {"error": "Attribute name is required"}
        
        value_counts = Counter()
        total_nodes = 0
        
        for node_id, attrs in self.graph.nodes(data=True):
            total_nodes += 1
            if attribute in attrs:
                value = attrs[attribute]
                # Handle unhashable types (lists, dicts) by converting to string
                try:
                    value_counts[value] += 1
                except TypeError:
                    # Convert unhashable types to string for counting
                    value_counts[str(value)] += 1
        
        # Convert to sorted list
        sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "attribute": attribute,
            "value_counts": sorted_counts,
            "total_nodes": total_nodes,
            "nodes_with_attribute": sum(value_counts.values()),
            "unique_values": len(value_counts),
            "coverage": sum(value_counts.values()) / total_nodes if total_nodes > 0 else 0
        }

    def _generate_search_recommendations(self) -> List[str]:
        """Generate search recommendations based on schema"""
        recommendations = []
        
        # Add attribute-based recommendations
        searchable = self.graph_schema.get_searchable_attributes()
        if searchable:
            recommendations.append(f"Search by: {', '.join(searchable[:3])}")
        
        numeric = self.graph_schema.get_numeric_attributes()
        if numeric:
            recommendations.append(f"Filter by range: {', '.join(numeric[:3])}")
        
        categorical = self.graph_schema.get_categorical_attributes()
        if categorical:
            recommendations.append(f"Group by: {', '.join(categorical[:3])}")
        
        return recommendations

    # Additional helper methods for other operations...
    def _get_sample_data(self, limit: int) -> Dict[str, Any]:
        """Get sample nodes and edges"""
        sample_nodes = []
        sample_edges = []
        
        # Get sample nodes
        node_list = list(self.graph.nodes(data=True))
        node_sample_size = min(limit, len(node_list))
        for i in range(0, len(node_list), max(1, len(node_list) // node_sample_size))[:node_sample_size]:
            node_id, attrs = node_list[i]
            sample_nodes.append({"node_id": node_id, "attributes": attrs})
        
        # Get sample edges
        edge_list = list(self.graph.edges(data=True))
        if edge_list:
            edge_sample_size = min(limit, len(edge_list))
            for i in range(0, len(edge_list), max(1, len(edge_list) // edge_sample_size))[:edge_sample_size]:
                u, v, attrs = edge_list[i]
                sample_edges.append({"source": u, "target": v, "attributes": attrs})
        
        return {
            "sample_nodes": sample_nodes,
            "sample_edges": sample_edges,
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges()
        }

    def _get_neighbors(self, node_id: str, depth: int, limit: int) -> Dict[str, Any]:
        """Get node neighbors"""
        if not node_id or node_id not in self.graph:
            return {"error": f"Node '{node_id}' not found in graph"}
        
        neighbors = set()
        current_layer = {node_id}
        
        for d in range(depth):
            next_layer = set()
            for node in current_layer:
                # Get predecessors and successors
                preds = set(self.graph.predecessors(node))
                succs = set(self.graph.successors(node))
                next_layer.update(preds | succs)
            
            neighbors.update(next_layer)
            current_layer = next_layer - neighbors  # Only new nodes for next iteration
            
            if len(neighbors) >= limit:
                break
        
        # Remove the original node from neighbors
        neighbors.discard(node_id)
        
        # Get node details for neighbors
        neighbor_details = []
        for neighbor in list(neighbors)[:limit]:
            attrs = self.graph.nodes.get(neighbor, {})
            neighbor_details.append({"node_id": neighbor, "attributes": attrs})
        
        return {
            "source_node": node_id,
            "depth": depth,
            "neighbors": neighbor_details,
            "total_found": len(neighbors),
            "has_more": len(neighbors) > limit
        }

    def _get_node_attribute_list(self) -> Dict[str, Any]:
        """Get list of all node attributes"""
        if not self.graph_schema:
            return {"error": "Schema not discovered yet"}
        
        return {
            "node_attributes": list(self.graph_schema.node_attributes.keys()),
            "total_count": len(self.graph_schema.node_attributes)
        }

    def _get_edge_attribute_list(self) -> Dict[str, Any]:
        """Get list of all edge attributes"""
        if not self.graph_schema:
            return {"error": "Schema not discovered yet"}
        
        return {
            "edge_attributes": list(self.graph_schema.edge_attributes.keys()),
            "total_count": len(self.graph_schema.edge_attributes)
        }

    def _analyze_attribute(self, attribute: str) -> Dict[str, Any]:
        """Analyze a specific attribute in detail"""
        if not attribute:
            return {"error": "Attribute name is required"}
        
        if not self.graph_schema:
            return {"error": "Schema not discovered yet"}
        
        attr_info = self.graph_schema.node_attributes.get(attribute)
        if not attr_info:
            return {"error": f"Attribute '{attribute}' not found"}
        
        return {
            "attribute": attribute,
            "type": attr_info.attribute_type.value,
            "coverage": attr_info.coverage,
            "unique_count": attr_info.unique_count,
            "sample_values": attr_info.sample_values,
            "stats": attr_info.stats,
            "categories": attr_info.categories,
            "is_high_cardinality": attr_info.is_high_cardinality
        }

    def _filter_nodes(self, filters: Dict[str, Any], limit: int) -> Dict[str, Any]:
        """Filter nodes by multiple attributes"""
        if not filters:
            return {"error": "Filters are required"}
        
        matches = []
        for node_id, attrs in self.graph.nodes(data=True):
            match = True
            for attr_name, attr_value in filters.items():
                if attr_name not in attrs or attrs[attr_name] != attr_value:
                    match = False
                    break
            
            if match:
                matches.append({"node_id": node_id, "attributes": attrs})
                if len(matches) >= limit:
                    break
        
        return {
            "results": matches,
            "total_found": len(matches),
            "filters": filters,
            "has_more": len(matches) == limit
        }

    def _range_search(self, attribute: str, min_value: Union[int, float], max_value: Union[int, float], limit: int) -> Dict[str, Any]:
        """Search nodes by numeric range"""
        if not attribute:
            return {"error": "Attribute name is required"}
        
        if min_value is None and max_value is None:
            return {"error": "At least one of min_value or max_value is required"}
        
        matches = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attribute in attrs:
                try:
                    value = float(attrs[attribute])
                    in_range = True
                    
                    if min_value is not None and value < min_value:
                        in_range = False
                    if max_value is not None and value > max_value:
                        in_range = False
                    
                    if in_range:
                        matches.append({"node_id": node_id, "attributes": attrs, "value": value})
                        if len(matches) >= limit:
                            break
                            
                except (ValueError, TypeError):
                    continue
        
        return {
            "results": matches,
            "total_found": len(matches),
            "search_criteria": {"attribute": attribute, "min_value": min_value, "max_value": max_value},
            "has_more": len(matches) == limit
        }

    def _find_by_values(self, attribute: str, values: List[str], limit: int) -> Dict[str, Any]:
        """Find nodes where attribute matches any of the provided values"""
        if not attribute or not values:
            return {"error": "Attribute name and values list are required"}
        
        matches = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attribute in attrs and attrs[attribute] in values:
                matches.append({"node_id": node_id, "attributes": attrs})
                if len(matches) >= limit:
                    break
        
        return {
            "results": matches,
            "total_found": len(matches),
            "search_criteria": {"attribute": attribute, "values": values},
            "has_more": len(matches) == limit
        }

    def _get_top_values(self, attribute: str, limit: int) -> Dict[str, Any]:
        """Get most common values for an attribute"""
        if not attribute:
            return {"error": "Attribute name is required"}
        
        value_counts = Counter()
        for node_id, attrs in self.graph.nodes(data=True):
            if attribute in attrs:
                value_counts[attrs[attribute]] += 1
        
        top_values = value_counts.most_common(limit)
        
        return {
            "attribute": attribute,
            "top_values": [{"value": value, "count": count} for value, count in top_values],
            "total_unique_values": len(value_counts)
        }

    def _get_attribute_distribution(self, attribute: str) -> Dict[str, Any]:
        """Get distribution analysis for an attribute"""
        if not attribute:
            return {"error": "Attribute name is required"}
        
        values = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attribute in attrs and attrs[attribute] is not None:
                values.append(attrs[attribute])
        
        if not values:
            return {"error": f"No values found for attribute '{attribute}'"}
        
        # Basic distribution statistics
        unique_values = set(values)
        value_counts = Counter(values)
        
        result = {
            "attribute": attribute,
            "total_values": len(values),
            "unique_values": len(unique_values),
            "most_common": value_counts.most_common(10),
            "coverage": len(values) / self.graph.number_of_nodes()
        }
        
        # Add numeric statistics if applicable
        numeric_values = []
        for value in values[:100]:  # Sample for performance
            try:
                numeric_values.append(float(value))
            except (ValueError, TypeError):
                break
        
        if len(numeric_values) > len(values) * 0.8:  # Mostly numeric
            result["numeric_stats"] = {
                "min": min(numeric_values),
                "max": max(numeric_values),
                "avg": sum(numeric_values) / len(numeric_values),
                "median": sorted(numeric_values)[len(numeric_values) // 2]
            }
        
        return result

    def _search_schema_by_query(self, query: str) -> Dict[str, Any]:
        """
        🔍 Search schema using natural language query (AI-powered)
        
        This method uses embedding similarity to find schema attributes
        relevant to the user's natural language question.
        """
        if not query:
            return {
                "error": "Query is required",
                "suggestion": "Ask a question about the data, e.g., 'What attributes are related to pricing?'"
            }
        
        # Ensure schema is discovered
        if self.graph_schema is None:
            self._discover_schema()
        
        # Try to initialize schema search if not already done
        if self.schema_searcher is None:
            self._init_schema_search()
        
        # Use embedding-based search if available
        if self.schema_searcher is not None:
            try:
                return self.schema_searcher.search_relevant_schema(query, top_k=5)
            except Exception as e:
                self.logger.error(f"Schema search failed: {e}")
                return {
                    "error": f"AI schema search failed: {str(e)}",
                    "fallback": "Use 'discover_schema' for complete schema overview"
                }
        
        # Fallback to basic keyword search
        return self._fallback_schema_search(query)
    
    def _fallback_schema_search(self, query: str) -> Dict[str, Any]:
        """Fallback schema search using basic keyword matching"""
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        relevant_attributes = []
        
        # Search node attributes
        for attr_name, attr_info in self.graph_schema.node_attributes.items():
            score = 0
            attr_lower = attr_name.lower()
            
            # Exact word matches
            if attr_lower in query_words:
                score += 10
            
            # Partial matches
            for word in query_words:
                if word in attr_lower:
                    score += 5
            
            if score > 0:
                relevant_attributes.append({
                    "name": attr_name,
                    "type": "node_attribute",
                    "relevance_score": f"{score/10:.1f}",
                    "attribute_type": str(attr_info.attribute_type),
                    "coverage": f"{attr_info.coverage:.1%}",
                    "samples": getattr(attr_info, 'sample_values', [])[:3]
                })
        
        # Search edge attributes  
        for attr_name, attr_info in self.graph_schema.edge_attributes.items():
            score = 0
            attr_lower = attr_name.lower()
            
            if attr_lower in query_words:
                score += 10
            
            for word in query_words:
                if word in attr_lower:
                    score += 5
            
            if score > 0:
                relevant_attributes.append({
                    "name": attr_name,
                    "type": "edge_attribute",
                    "relevance_score": f"{score/10:.1f}",
                    "attribute_type": str(attr_info.attribute_type),
                    "coverage": f"{attr_info.coverage:.1%}",
                    "samples": getattr(attr_info, 'sample_values', [])[:3]
                })
        
        # Sort by relevance
        relevant_attributes.sort(key=lambda x: float(x["relevance_score"]), reverse=True)
        
        return {
            "query": query,
            "relevant_attributes": relevant_attributes[:5],  # Top 5
            "search_method": "keyword_fallback",
            "suggestion": "Install sentence-transformers for AI-powered schema search",
            "found_count": len(relevant_attributes)
        } 