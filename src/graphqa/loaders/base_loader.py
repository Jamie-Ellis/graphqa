"""
Base loader interface for the Universal Graph Retriever.

This module defines the abstract base class that all dataset loaders must implement,
ensuring consistent behavior and metadata provision across different data sources.
"""

from abc import ABC, abstractmethod
import networkx as nx
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..data_structures import (
    DatasetInfo, SchemaInfo, LoadResult, AttributeInfo, AttributeType
)

logger = logging.getLogger(__name__)


class BaseGraphLoader(ABC):
    """
    Abstract base class for all graph dataset loaders.
    
    This interface standardizes how different datasets are loaded and validated,
    ensuring consistent behavior across diverse data sources like Amazon products,
    cloud architectures, research papers, etc.
    
    Key Design Principles:
    1. Schema Discovery: Automatically analyze and report graph structure
    2. Validation: Ensure data quality and flag issues
    3. Metadata Generation: Provide comprehensive dataset information
    4. Error Handling: Graceful failure with informative messages
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the loader with configuration.
        
        Args:
            config: Loader-specific configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def load_graph(self) -> nx.MultiDiGraph:
        """
        Load the dataset and return a NetworkX graph.
        
        This method must be implemented by each loader to handle the specific
        data format and conversion logic for their domain.
        
        Returns:
            NetworkX MultiDiGraph with loaded data
            
        Raises:
            LoaderError: If data cannot be loaded or is invalid
        """
        pass
    
    @abstractmethod
    def get_dataset_description(self) -> str:
        """
        Get a human-readable description of this dataset.
        
        Returns:
            Description string explaining what this dataset contains
        """
        pass
    
    @abstractmethod 
    def get_sample_queries(self) -> List[str]:
        """
        Get domain-appropriate sample questions for this dataset.
        
        These queries should showcase the types of analysis that make sense
        for this particular domain and data structure.
        
        Returns:
            List of example natural language questions
        """
        pass
    
    def load_dataset(self) -> LoadResult:
        """
        Load the complete dataset with timing and error handling.
        
        This method orchestrates the loading process, including validation,
        schema discovery, and metadata generation.
        
        Returns:
            LoadResult with success status and dataset information
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting dataset load: {self.__class__.__name__}")
            
            # Load the graph
            graph = self.load_graph()
            
            # Validate the loaded graph
            validation_result = self.validate_graph(graph)
            if not validation_result["valid"]:
                return LoadResult(
                    success=False,
                    error_message=f"Graph validation failed: {validation_result['errors']}",
                    load_time_seconds=time.time() - start_time
                )
            
            # Discover schema information
            schema = self.discover_schema(graph)
            
            # Create dataset info
            dataset_info = DatasetInfo(
                name=self.get_dataset_name(),
                description=self.get_dataset_description(),
                loader_class=self.__class__.__name__,
                graph=graph,
                schema=schema,
                is_loaded=True,
                load_time_seconds=time.time() - start_time,
                config=self.config,
                data_sources=self.get_data_sources()
            )
            
            # Add sample queries to schema
            schema.suggested_queries = self.get_sample_queries()
            
            self.logger.info(
                f"Successfully loaded {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges in {dataset_info.load_time_seconds:.2f}s"
            )
            
            return LoadResult(
                success=True,
                dataset_info=dataset_info,
                load_time_seconds=dataset_info.load_time_seconds,
                warnings=validation_result.get("warnings", [])
            )
            
        except Exception as e:
            self.logger.error(f"Dataset loading failed: {str(e)}")
            return LoadResult(
                success=False,
                error_message=str(e),
                load_time_seconds=time.time() - start_time
            )
    
    def validate_graph(self, graph: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Validate the loaded graph for quality and consistency.
        
        Args:
            graph: The NetworkX graph to validate
            
        Returns:
            Dict with validation results, errors, and warnings
        """
        errors = []
        warnings = []
        
        # Basic structural validation
        if graph.number_of_nodes() == 0:
            errors.append("Graph contains no nodes")
        
        if graph.number_of_edges() == 0:
            warnings.append("Graph contains no edges")
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(graph))
        if len(isolated_nodes) > graph.number_of_nodes() * 0.5:
            warnings.append(f"High number of isolated nodes: {len(isolated_nodes)}")
        
        # Check for missing node attributes
        node_with_no_attrs = 0
        for node_id, attrs in graph.nodes(data=True):
            if not attrs:
                node_with_no_attrs += 1
        
        if node_with_no_attrs > 0:
            warnings.append(f"{node_with_no_attrs} nodes have no attributes")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def discover_schema(self, graph: nx.MultiDiGraph) -> SchemaInfo:
        """
        Discover and analyze the schema of the loaded graph.
        
        This performs comprehensive analysis of node and edge attributes,
        determining types, distributions, and providing sample data.
        
        Args:
            graph: The NetworkX graph to analyze
            
        Returns:
            SchemaInfo with complete schema analysis
        """
        try:
            schema = SchemaInfo(
                dataset_name=self.get_dataset_name(),
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
                is_directed=graph.is_directed()
            )
            
            # Analyze node attributes
            try:
                schema.node_attributes = self._analyze_node_attributes(graph)
            except Exception as e:
                self.logger.warning(f"Failed to analyze node attributes: {e}")
                schema.node_attributes = {}
            
            # Analyze edge attributes  
            try:
                schema.edge_attributes = self._analyze_edge_attributes(graph)
            except Exception as e:
                self.logger.warning(f"Failed to analyze edge attributes: {e}")
                schema.edge_attributes = {}
            
            # Get representative samples
            try:
                schema.sample_nodes = self._get_sample_nodes(graph, limit=5)
            except Exception as e:
                self.logger.warning(f"Failed to get sample nodes: {e}")
                schema.sample_nodes = []
            
            try:
                schema.sample_edges = self._get_sample_edges(graph, limit=5)
            except Exception as e:
                self.logger.warning(f"Failed to get sample edges: {e}")
                schema.sample_edges = []
            
            # Generate search recommendations
            try:
                schema.recommended_searches = self._generate_search_recommendations(schema)
            except Exception as e:
                self.logger.warning(f"Failed to generate search recommendations: {e}")
                schema.recommended_searches = []
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Schema discovery failed: {e}")
            # Return minimal schema on failure
            return SchemaInfo(
                dataset_name=self.get_dataset_name(),
                node_count=graph.number_of_nodes() if graph else 0,
                edge_count=graph.number_of_edges() if graph else 0,
                is_directed=graph.is_directed() if graph else True,
                node_attributes={},
                edge_attributes={},
                sample_nodes=[],
                sample_edges=[],
                recommended_searches=[]
            )
    
    def _analyze_node_attributes(self, graph: nx.MultiDiGraph) -> Dict[str, AttributeInfo]:
        """Analyze all node attributes in the graph"""
        return self._analyze_attributes(
            data_iterator=graph.nodes(data=True),
            sample_size=min(1000, graph.number_of_nodes()),
            total_count=graph.number_of_nodes()
        )
    
    def _analyze_edge_attributes(self, graph: nx.MultiDiGraph) -> Dict[str, AttributeInfo]:
        """Analyze all edge attributes in the graph"""
        return self._analyze_attributes(
            data_iterator=[(u, v, attrs) for u, v, attrs in graph.edges(data=True)],
            sample_size=min(1000, graph.number_of_edges()),
            total_count=graph.number_of_edges(),
            is_edge=True
        )
    
    def _analyze_attributes(self, data_iterator, sample_size: int, total_count: int, is_edge: bool = False) -> Dict[str, AttributeInfo]:
        """Generic attribute analysis for nodes or edges"""
        attribute_info = {}
        
        try:
            # Sample data for analysis
            sample_data = list(data_iterator)[:sample_size]
            
            if not sample_data:
                return attribute_info
            
            # Extract attributes
            if is_edge:
                all_attributes = set()
                for u, v, attrs in sample_data:
                    all_attributes.update(attrs.keys())
            else:
                all_attributes = set()
                for node_id, attrs in sample_data:
                    all_attributes.update(attrs.keys())
            
            # Analyze each attribute
            for attr_name in all_attributes:
                try:
                    if is_edge:
                        values = [attrs.get(attr_name) for u, v, attrs in sample_data 
                                 if attr_name in attrs and attrs[attr_name] is not None]
                    else:
                        values = [attrs.get(attr_name) for node_id, attrs in sample_data
                                 if attr_name in attrs and attrs[attr_name] is not None]
                    
                    if not values:
                        continue
                    
                    # Determine attribute type and analyze
                    attr_type = self._infer_attribute_type(values)
                    
                    # Handle unhashable types for uniqueness calculation
                    try:
                        unique_values = list(set(values))
                    except TypeError:
                        # For unhashable types (lists, dicts), manually find unique values
                        unique_values = []
                        seen_str = set()
                        for value in values:
                            try:
                                value_str = str(value)
                                if value_str not in seen_str:
                                    unique_values.append(value)
                                    seen_str.add(value_str)
                                    if len(unique_values) >= 50:  # Limit for performance
                                        break
                            except Exception:
                                # Skip values that can't be converted to string
                                continue
                    
                    # Smart sample truncation to prevent context explosion
                    smart_samples = self._create_smart_samples(unique_values[:10])
                    
                    attr_info = AttributeInfo(
                        name=attr_name,
                        attribute_type=attr_type,
                        coverage=len(values) / len(sample_data) if sample_data else 0,
                        unique_count=len(unique_values),
                        sample_values=smart_samples
                    )
                    
                    # Add type-specific information
                    if attr_type == AttributeType.NUMERIC:
                        try:
                            numeric_values = [float(v) for v in values if v is not None]
                            if numeric_values:
                                attr_info.stats = {
                                    "min": min(numeric_values),
                                    "max": max(numeric_values),
                                    "avg": sum(numeric_values) / len(numeric_values)
                                }
                        except (ValueError, TypeError):
                            # If conversion fails, treat as categorical
                            attr_type = AttributeType.CATEGORICAL
                            attr_info.attribute_type = attr_type
                    
                    if attr_type == AttributeType.CATEGORICAL:
                        try:
                            if len(unique_values) <= 20:
                                attr_info.categories = unique_values
                            else:
                                # For high cardinality, show top categories
                                from collections import Counter
                                try:
                                    # Try counting original values first
                                    hashable_values = []
                                    for v in values:
                                        try:
                                            # Test if hashable by trying to add to set
                                            {v}
                                            hashable_values.append(v)
                                        except TypeError:
                                            # Use string representation for unhashable types
                                            hashable_values.append(str(v))
                                    
                                    value_counts = Counter(hashable_values)
                                    top_categories = [item for item, count in value_counts.most_common(10)]
                                except Exception:
                                    # Fallback: use string representations
                                    str_values = [str(v) for v in values]
                                    value_counts = Counter(str_values)
                                    top_categories = [item for item, count in value_counts.most_common(10)]
                                attr_info.categories = top_categories
                                attr_info.is_high_cardinality = True
                        except Exception as e:
                            self.logger.warning(f"Failed to analyze categories for attribute {attr_name}: {e}")
                            attr_info.categories = []
                    
                    attribute_info[attr_name] = attr_info
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze attribute {attr_name}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Failed to analyze attributes: {e}")
            return {}
        
        return attribute_info
    
    def _infer_attribute_type(self, values: List[Any]) -> AttributeType:
        """Infer the type of an attribute from sample values"""
        if not values:
            return AttributeType.UNKNOWN
        
        # Check for numeric
        numeric_count = 0
        for value in values[:20]:  # Sample for type detection
            try:
                float(value)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        if numeric_count > len(values[:20]) * 0.8:  # 80% numeric
            return AttributeType.NUMERIC
        
        # Check for boolean
        boolean_values = {True, False, "true", "false", "True", "False", 1, 0}
        if all(str(v).lower() in ["true", "false", "1", "0"] for v in values[:10]):
            return AttributeType.BOOLEAN
        
        # Check for list/array
        if any(isinstance(v, (list, tuple)) for v in values[:5]):
            return AttributeType.LIST
        
        # Check for categorical vs text
        unique_ratio = len(set(values)) / len(values)
        avg_length = sum(len(str(v)) for v in values[:20]) / min(20, len(values))
        
        if unique_ratio < 0.5 and avg_length < 50:  # Low uniqueness, short values
            return AttributeType.CATEGORICAL
        else:
            return AttributeType.TEXT
    
    def _get_sample_nodes(self, graph: nx.MultiDiGraph, limit: int = 5) -> List[Dict[str, Any]]:
        """Get representative sample nodes"""
        sample_nodes = []
        node_list = list(graph.nodes(data=True))
        
        # Take evenly distributed samples
        if len(node_list) <= limit:
            sample_nodes = [{"node_id": nid, "attributes": attrs} for nid, attrs in node_list]
        else:
            step = len(node_list) // limit
            for i in range(0, len(node_list), step)[:limit]:
                nid, attrs = node_list[i]
                sample_nodes.append({"node_id": nid, "attributes": attrs})
        
        return sample_nodes
    
    def _get_sample_edges(self, graph: nx.MultiDiGraph, limit: int = 5) -> List[Dict[str, Any]]:
        """Get representative sample edges"""
        sample_edges = []
        edge_list = list(graph.edges(data=True))
        
        if len(edge_list) <= limit:
            sample_edges = [{"source": u, "target": v, "attributes": attrs} 
                           for u, v, attrs in edge_list]
        else:
            step = len(edge_list) // limit
            for i in range(0, len(edge_list), step)[:limit]:
                u, v, attrs = edge_list[i]
                sample_edges.append({"source": u, "target": v, "attributes": attrs})
        
        return sample_edges
    
    def _generate_search_recommendations(self, schema: SchemaInfo) -> List[str]:
        """Generate recommended search strategies based on discovered schema"""
        recommendations = []
        
        # Add recommendations based on discovered attributes
        searchable_attrs = schema.get_searchable_attributes()
        if searchable_attrs:
            recommendations.append(f"Try searching by: {', '.join(searchable_attrs[:3])}")
        
        numeric_attrs = schema.get_numeric_attributes()
        if numeric_attrs:
            recommendations.append(f"Use range filtering on: {', '.join(numeric_attrs[:3])}")
        
        categorical_attrs = schema.get_categorical_attributes()
        if categorical_attrs:
            recommendations.append(f"Group by categories: {', '.join(categorical_attrs[:3])}")
        
        # Add graph structure recommendations
        if schema.edge_count > 0:
            recommendations.append("Explore node relationships and connectivity patterns")
            recommendations.append("Use PageRank to find most important nodes")
        
        if schema.node_count > 100:
            recommendations.append("Try community detection to find clusters")
        
        return recommendations
    
    # Abstract methods that subclasses may override
    def get_dataset_name(self) -> str:
        """Get the name of this dataset"""
        return self.__class__.__name__.replace("Loader", "")
    
    def get_data_sources(self) -> List[str]:
        """Get list of data source files/URLs"""
        return []

    def _create_smart_samples(self, values: List[Any]) -> List[str]:
        """
        Create smart sample representations to prevent context explosion.
        
        Args:
            values: List of sample values
            
        Returns:
            List of smart sample strings
        """
        if not values:
            return []
        
        smart_samples = []
        total_chars = 0
        max_total_chars = 300  # Total budget for all samples
        max_sample_chars = 50  # Individual sample limit
        
        for value in values[:5]:  # Max 5 samples
            if total_chars >= max_total_chars:
                break
                
            # Convert to string
            str_val = str(value) if value is not None else "None"
            
            # Apply intelligent truncation
            if len(str_val) <= max_sample_chars:
                sample = str_val
            else:
                # Content-aware truncation
                if self._looks_like_url(str_val):
                    # For URLs: show domain and path structure
                    if '//' in str_val and '/' in str_val[str_val.find('//')+2:]:
                        parts = str_val.split('://', 1)
                        if len(parts) == 2:
                            domain_path = parts[1].split('/', 1)
                            sample = f"{parts[0]}://{domain_path[0]}/..."
                        else:
                            sample = str_val[:max_sample_chars-3] + "..."
                    else:
                        sample = str_val[:max_sample_chars-3] + "..."
                elif self._looks_like_id(str_val):
                    # For IDs: show prefix and suffix
                    if ':' in str_val:
                        parts = str_val.split(':')
                        if len(parts) >= 2:
                            sample = f"{parts[0]}:...:{parts[-1]}"
                        else:
                            sample = str_val[:max_sample_chars-3] + "..."
                    else:
                        # Show beginning and end
                        prefix_len = (max_sample_chars - 3) // 2
                        suffix_len = max_sample_chars - prefix_len - 3
                        sample = f"{str_val[:prefix_len]}...{str_val[-suffix_len:]}"
                else:
                    # Default truncation
                    sample = str_val[:max_sample_chars-3] + "..."
            
            # Check if we have budget for this sample
            if total_chars + len(sample) <= max_total_chars:
                smart_samples.append(sample)
                total_chars += len(sample)
            else:
                break
        
        return smart_samples
    
    def _looks_like_url(self, text: str) -> bool:
        """Check if text looks like a URL"""
        return (text.startswith(('http://', 'https://', 'ftp://')) or 
                '.com' in text or '.org' in text or '.net' in text)
    
    def _looks_like_id(self, text: str) -> bool:
        """Check if text looks like an ID or identifier"""
        return (':' in text or 
                (len(text) > 10 and '-' in text and text.replace('-', '').replace('_', '').isalnum()))


class LoaderError(Exception):
    """Exception raised when dataset loading fails"""
    pass 