"""
Schema search engine combining embedding, processing, and context management.

This module provides the main QuerySchemaSearcher class that orchestrates
all components to provide intelligent, context-aware schema discovery.
"""

import json
import logging
from typing import Dict, List, Any

from .embedder import SchemaEmbedder
from .query_processor import QueryProcessor
from .context_manager import SmartSampleTruncator, CompactSchemaCreator

logger = logging.getLogger(__name__)


class QuerySchemaSearcher:
    """Main schema search engine combining all components"""
    
    def __init__(self, embedder: SchemaEmbedder):
        """
        Initialize search engine with embedding model.
        
        Args:
            embedder: Initialized SchemaEmbedder with schema embeddings
        """
        self.embedder = embedder
        self.query_processor = QueryProcessor()
        self.sample_truncator = SmartSampleTruncator()
        self.schema_creator = CompactSchemaCreator()
    
    def search_relevant_schema(self, 
                             query: str, 
                             top_k: int = 5, 
                             max_context_chars: int = 800) -> Dict[str, Any]:
        """
        Main search function: natural language query → relevant schema attributes
        
        Returns context-optimized response that won't overflow LLM.
        
        Args:
            query: User's natural language query
            top_k: Maximum number of attributes to return
            max_context_chars: Maximum character budget for response
            
        Returns:
            Dictionary with relevant attributes and usage guidance
        """
        
        # Step 1: Validate and clean query
        validation = self.query_processor.validate_query(query)
        if not validation["valid"]:
            return {
                "error": validation["reason"],
                "suggestion": validation["suggestion"],
                "cleaned_query": validation.get("cleaned_query", "")
            }
        
        clean_query = self.query_processor.clean_query_text(query)
        
        # Step 2: Search for relevant schema items via embedding similarity
        try:
            top_matches = self.embedder.search_relevant_attributes(clean_query, top_k)
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return {
                "error": f"Search failed: {str(e)}",
                "fallback": "Use get_schema_overview() for basic schema information",
                "suggestion": "Check if sentence-transformers is properly installed"
            }
        
        if not top_matches:
            return {
                "error": "No relevant attributes found",
                "suggestion": "Try a different query or use get_schema_overview() to see all available attributes",
                "query_analysis": validation.get("analysis", {})
            }
        
        # Step 3: Build context-aware response within budget
        relevant_items = []
        total_chars = 0
        
        for item_idx, similarity_score in top_matches:
            if total_chars >= max_context_chars:
                break
                
            if item_idx >= len(self.embedder.schema_items):
                logger.warning(f"Invalid item index: {item_idx}")
                continue
                
            item = self.embedder.schema_items[item_idx]
            
            # Create compact representation
            compact_item = self._create_compact_item(item, similarity_score)
            
            # Estimate size and check budget
            item_size = self.schema_creator.estimate_context_size(compact_item)
            
            if total_chars + item_size <= max_context_chars:
                relevant_items.append(compact_item)
                total_chars += item_size
            else:
                # If we have room for at least a minimal item, try to fit it
                if total_chars < max_context_chars * 0.8:  # 80% budget used
                    minimal_item = self._create_minimal_item(item, similarity_score)
                    minimal_size = self.schema_creator.estimate_context_size(minimal_item)
                    
                    if total_chars + minimal_size <= max_context_chars:
                        relevant_items.append(minimal_item)
                        total_chars += minimal_size
                break
        
        # Step 4: Generate usage examples and response
        usage_examples = self._generate_usage_examples(relevant_items)
        
        return {
            "query": query,
            "cleaned_query": clean_query,
            "relevant_attributes": relevant_items,
            "found_count": len(relevant_items),
            "context_size_chars": total_chars,
            "search_method": "embedding_similarity",
            "usage_examples": usage_examples,
            "performance_info": {
                "total_matches": len(top_matches),
                "returned_matches": len(relevant_items),
                "context_budget_used": f"{(total_chars/max_context_chars)*100:.1f}%"
            }
        }
    
    def _create_compact_item(self, item: Dict[str, Any], similarity_score: float) -> Dict[str, Any]:
        """Create LLM-friendly compact representation"""
        
        # Smart sample truncation
        samples = item["metadata"].get("samples", [])
        truncated_samples = self.sample_truncator.truncate_samples_intelligently(
            samples, max_total_chars=150, max_sample_chars=40
        )
        
        compact = {
            "name": item["name"],
            "type": item["type"],  # "node_attribute" or "edge_attribute"
            "relevance_score": f"{similarity_score:.2f}",
            "attribute_type": item["metadata"].get("attribute_type", "unknown"),
            "coverage": f"{item['metadata'].get('coverage', 0):.1%}",
        }
        
        # Only add samples if they're not empty after truncation
        if truncated_samples:
            compact["samples"] = truncated_samples
        
        return compact
    
    def _create_minimal_item(self, item: Dict[str, Any], similarity_score: float) -> Dict[str, Any]:
        """Create minimal representation when context budget is tight"""
        
        return {
            "name": item["name"],
            "type": item["type"],
            "relevance_score": f"{similarity_score:.2f}",
            "attribute_type": item["metadata"].get("attribute_type", "unknown")
        }
    
    def _generate_usage_examples(self, relevant_items: List[Dict]) -> List[str]:
        """Generate usage examples based on discovered attributes"""
        
        if not relevant_items:
            return []
        
        examples = []
        
        for item in relevant_items[:3]:  # Max 3 examples to keep response compact
            attr_name = item["name"]
            attr_type = item.get("attribute_type", "unknown")
            samples = item.get("samples", [])
            
            if attr_type == "categorical" and samples:
                example_value = samples[0]
                examples.append(f'find_by_attribute(attribute="{attr_name}", value="{example_value}")')
            elif attr_type == "numeric":
                examples.append(f'range_search(attribute="{attr_name}", min=0, max=100)')
            elif attr_type == "text" and samples:
                # Use part of sample for text search example
                sample = str(samples[0])
                search_term = sample.split()[0] if sample.split() else sample[:10]
                examples.append(f'text_search(query="{search_term}", fields=["{attr_name}"])')
            else:
                examples.append(f'get_attribute_details("{attr_name}")')
        
        return examples
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search system"""
        
        if self.embedder.schema_embeddings is None or self.embedder.schema_items is None:
            return {"error": "Schema embeddings not initialized"}
        
        node_attrs = sum(1 for item in self.embedder.schema_items if item["type"] == "node_attribute")
        edge_attrs = sum(1 for item in self.embedder.schema_items if item["type"] == "edge_attribute")
        
        return {
            "total_searchable_attributes": len(self.embedder.schema_items),
            "node_attributes": node_attrs,
            "edge_attributes": edge_attrs,
            "embedding_dimensions": self.embedder.schema_embeddings.shape[1],
            "model_name": self.embedder.model_name,
            "memory_usage_mb": (self.embedder.schema_embeddings.nbytes / (1024 * 1024)),
            "system_ready": True
        } 