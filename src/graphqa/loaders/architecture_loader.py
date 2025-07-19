"""
Architecture Loader for Universal Graph Retriever

Loads cloud infrastructure topology from JSON files with automatic schema discovery.
Maintains backward compatibility with existing architecture analysis workflows.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import networkx as nx

from .base_loader import BaseGraphLoader, LoaderError

logger = logging.getLogger(__name__)


class ArchitectureLoader(BaseGraphLoader):
    """
    Universal loader for cloud architecture datasets.
    
    Loads infrastructure topology from JSON files containing components
    and their relationships. Preserves all existing functionality while
    providing universal schema discovery capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize architecture loader.
        
        Config options:
        - architecture_file: Path to architecture JSON file
        - validate_references: If True, validate component references
        """
        super().__init__(config)
        
        self.architecture_file = self.config.get('architecture_file')
        self.validate_references = self.config.get('validate_references', True)
        
        if not self.architecture_file:
            raise LoaderError("architecture_file must be specified in config")
        
        self.logger.info(f"Architecture loader initialized for: {self.architecture_file}")
    
    def load_graph(self) -> nx.MultiDiGraph:
        """Load architecture data from JSON file"""
        arch_path = Path(self.architecture_file)
        
        if not arch_path.exists():
            raise LoaderError(f"Architecture file not found: {arch_path}")
        
        try:
            with open(arch_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Loading architecture from {arch_path}")
            
            # Build graph from architecture data
            graph = self._build_graph_from_data(data)
            
            self.logger.info(
                f"Architecture loaded: {graph.number_of_nodes()} components, "
                f"{graph.number_of_edges()} relationships"
            )
            
            return graph
            
        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON in architecture file: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load architecture: {e}")
            raise LoaderError(f"Architecture loading failed: {e}")
    
    def _build_graph_from_data(self, data: Dict[str, Any]) -> nx.MultiDiGraph:
        """Build NetworkX graph from architecture JSON data"""
        graph = nx.MultiDiGraph()
        
        # Extract components and relationships from various possible structures
        components = self._extract_components(data)
        relationships = self._extract_relationships(data)
        
        # Add nodes (components)
        for component in components:
            component_id = component.get('id') or component.get('component_id')
            if not component_id:
                self.logger.warning(f"Component missing ID: {component}")
                continue
            
            # Build node attributes
            node_attrs = {}
            for key, value in component.items():
                if key != 'id':  # Don't duplicate the ID
                    node_attrs[key] = value
            
            # Ensure consistent field names while preserving originals
            if 'id' in component:
                node_attrs['component_id'] = component['id']
            
            graph.add_node(component_id, **node_attrs)
        
        # Add edges (relationships)
        relationship_count = 0
        missing_source = 0
        missing_target = 0
        missing_both = 0
        
        for relationship in relationships:
            # Try different field name patterns for source/target
            source = (relationship.get('source') or relationship.get('from') or 
                     relationship.get('sourceComponentId') or relationship.get('source_id'))
            target = (relationship.get('target') or relationship.get('to') or 
                     relationship.get('targetComponentId') or relationship.get('target_id'))
            
            if not source or not target:
                self.logger.warning(f"Relationship missing source/target: {relationship}")
                continue
            
            # Validate references if enabled
            if self.validate_references:
                source_exists = graph.has_node(source)
                target_exists = graph.has_node(target)
                
                if not source_exists and not target_exists:
                    missing_both += 1
                    if missing_both <= 3:  # Only log first 3 examples
                        self.logger.warning(f"Relationship missing both source and target: {source} -> {target}")
                    continue
                elif not source_exists:
                    missing_source += 1
                    if missing_source <= 3:  # Only log first 3 examples
                        self.logger.warning(f"Relationship missing source: {source}")
                    continue
                elif not target_exists:
                    missing_target += 1
                    if missing_target <= 3:  # Only log first 3 examples
                        self.logger.warning(f"Relationship missing target: {target}")
                    continue
            
            # Build edge attributes
            edge_attrs = {}
            for key, value in relationship.items():
                if key not in ['source', 'target', 'from', 'to', 'sourceComponentId', 'targetComponentId', 'source_id', 'target_id']:
                    edge_attrs[key] = value
            
            # Set default relationship type if not specified
            if 'relationship_type' not in edge_attrs and 'type' not in edge_attrs and 'relationshipType' not in edge_attrs:
                edge_attrs['relationship_type'] = 'depends_on'
            
            graph.add_edge(source, target, **edge_attrs)
            relationship_count += 1
        
        # Log relationship loading summary
        total_relationships = len(relationships)
        self.logger.info(f"Relationship loading summary:")
        self.logger.info(f"  Total relationships: {total_relationships}")
        self.logger.info(f"  Valid relationships: {relationship_count}")
        self.logger.info(f"  Missing source only: {missing_source}")
        self.logger.info(f"  Missing target only: {missing_target}")
        self.logger.info(f"  Missing both: {missing_both}")
        if total_relationships > 0:
            success_rate = relationship_count / total_relationships * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
        else:
            self.logger.info(f"  Success rate: 0%")
        return graph
    
    def _extract_components(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract components from various JSON structures"""
        components = []
        
        # Try different possible structures
        if 'components' in data:
            # Standard structure: {"components": [...]}
            components = data['components']
        elif 'nodes' in data:
            # Graph structure: {"nodes": [...]}
            components = data['nodes']
        elif isinstance(data, list):
            # Direct list of components
            components = data
        elif 'architecture' in data and 'components' in data['architecture']:
            # Nested structure: {"architecture": {"components": [...]}}
            components = data['architecture']['components']
        else:
            # Look for any list that contains component-like objects
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    if any(field in value[0] for field in ['id', 'component_id', 'name']):
                        components = value
                        break
        
        if not components:
            self.logger.warning("No components found in architecture data")
        
        return components if isinstance(components, list) else []
    
    def _extract_relationships(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relationships from various JSON structures"""
        relationships = []
        
        # Try different possible structures
        if 'relationships' in data:
            relationships = data['relationships']
        elif 'edges' in data:
            relationships = data['edges']
        elif 'dependencies' in data:
            relationships = data['dependencies']
        elif 'connections' in data:
            relationships = data['connections']
        elif 'architecture' in data and 'relationships' in data['architecture']:
            relationships = data['architecture']['relationships']
        
        return relationships if isinstance(relationships, list) else []
    
    def get_dataset_description(self) -> str:
        """Get description of the architecture dataset"""
        return (
            "Cloud Architecture Network containing infrastructure components "
            "and their dependencies. Includes services, databases, load balancers, "
            "and other infrastructure elements with their relationships and configurations."
        )
    
    def get_sample_queries(self) -> List[str]:
        """Get sample queries appropriate for architecture data"""
        return [
            "What storage components do we have?",
            "Find components in the us-west-2 region",
            "Show all database dependencies",
            "Which components are most critical?",
            "Find isolated components with no connections",
            "What components are in the production environment?",
            "Show the dependency chain for the API gateway",
            "Find components that depend on the database",
            "What are the most connected components?",
            "Identify potential single points of failure"
        ]
    
    def get_data_sources(self) -> List[str]:
        """Get list of data source files"""
        return [self.architecture_file]
    
    def get_dataset_name(self) -> str:
        """Get the dataset name"""
        return "Cloud Architecture"
    
    def validate_graph(self, graph: nx.MultiDiGraph) -> Dict[str, Any]:
        """Enhanced validation for architecture graphs"""
        # Get base validation
        result = super().validate_graph(graph)
        
        # Add architecture-specific validation
        warnings = result.get("warnings", [])
        errors = result.get("errors", [])
        
        # Check for common architecture issues
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            warnings.append(f"Found {len(isolated_nodes)} isolated components: {isolated_nodes[:5]}")
        
        # Check for cycles (might indicate circular dependencies)
        if graph.is_directed():
            try:
                cycles = list(nx.simple_cycles(graph))
                if cycles:
                    warnings.append(f"Found {len(cycles)} potential circular dependencies")
            except:
                pass  # Skip if cycle detection fails
        
        # Check for missing critical attributes
        components_without_type = []
        for node_id, attrs in graph.nodes(data=True):
            if 'type' not in attrs and 'component_type' not in attrs:
                components_without_type.append(node_id)
        
        if components_without_type:
            warnings.append(f"{len(components_without_type)} components missing type information")
        
        result["warnings"] = warnings
        result["errors"] = errors
        result["valid"] = len(errors) == 0
        
        return result 