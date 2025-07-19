"""
Core embedding system for schema attribute search.

This module provides the SchemaEmbedder and SchemaItemCreator classes that enable
domain-agnostic similarity search between user queries and graph schema attributes.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..loaders.base_loader import SchemaInfo, AttributeInfo

logger = logging.getLogger(__name__)


class SchemaEmbedder:
    """Lightweight embedding model for schema attribute search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with lightweight, fast embedding model.
        
        Model characteristics:
        - Size: ~90MB
        - Speed: ~1000 sentences/second on CPU
        - Dimensions: 384
        - No GPU required
        
        Args:
            model_name: SentenceTransformer model name
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for schema embedding. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = None
        self.schema_embeddings: Optional[np.ndarray] = None
        self.schema_items: Optional[List[Dict]] = None
        
        self._load_model()
        
    def _load_model(self):
        """Load the embedding model with error handling"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
    def initialize_schema_embeddings(self, schema: SchemaInfo) -> None:
        """Create and store embeddings for all schema items (one-time cost)"""
        
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        logger.info("Creating schema embeddings...")
        
        # Create schema items with factual descriptions
        item_creator = SchemaItemCreator()
        self.schema_items = item_creator.create_schema_items(schema)
        
        logger.info(f"Created {len(self.schema_items)} schema items")
        
        # Extract text descriptions
        schema_texts = [item["text"] for item in self.schema_items]
        
        # Generate embeddings (one-time computation)
        try:
            self.schema_embeddings = self.model.encode(
                schema_texts, 
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=32  # Process in batches for memory efficiency
            )
            logger.info(f"✅ Generated embeddings: {self.schema_embeddings.shape}")
        except Exception as e:
            logger.error(f"Failed to generate schema embeddings: {e}")
            raise
        
    def search_relevant_attributes(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Find schema attributes most relevant to user query.
        
        Args:
            query: User's natural language query
            top_k: Number of top matches to return
            
        Returns:
            List of (item_index, similarity_score) tuples
        """
        if self.schema_embeddings is None:
            raise ValueError("Schema embeddings not initialized. Call initialize_schema_embeddings() first.")
            
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        # Embed the query
        try:
            query_embedding = self.model.encode([query], convert_to_tensor=False)[0]
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
        
        # Calculate cosine similarity (normalized dot product)
        # Normalize embeddings for cosine similarity
        schema_norms = np.linalg.norm(self.schema_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        if query_norm == 0:
            logger.warning("Query embedding has zero norm")
            return []
        
        # Compute cosine similarities
        similarities = np.dot(self.schema_embeddings, query_embedding) / (schema_norms * query_norm)
        
        # Get top-k most similar items
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


class SchemaItemCreator:
    """Creates embedable text representations of schema attributes"""
    
    def create_schema_items(self, schema: SchemaInfo) -> List[Dict[str, Any]]:
        """Convert schema to list of embedable items"""
        
        items = []
        
        # Process node attributes
        for attr_name, attr_info in schema.node_attributes.items():
            text_repr = self._create_attribute_text(attr_name, attr_info, "node")
            
            items.append({
                "type": "node_attribute",
                "name": attr_name,
                "text": text_repr,
                "metadata": {
                    "attribute_type": str(attr_info.attribute_type),
                    "coverage": attr_info.coverage,
                    "unique_count": attr_info.unique_count,
                    "samples": attr_info.sample_values[:3] if hasattr(attr_info, 'sample_values') else []
                }
            })
        
        # Process edge attributes
        for attr_name, attr_info in schema.edge_attributes.items():
            text_repr = self._create_attribute_text(attr_name, attr_info, "edge")
            
            items.append({
                "type": "edge_attribute",
                "name": attr_name,
                "text": text_repr,
                "metadata": {
                    "attribute_type": str(attr_info.attribute_type),
                    "coverage": attr_info.coverage,
                    "unique_count": attr_info.unique_count,
                    "samples": attr_info.sample_values[:3] if hasattr(attr_info, 'sample_values') else []
                }
            })
        
        logger.info(f"Created {len(items)} schema items ({len(schema.node_attributes)} node + {len(schema.edge_attributes)} edge attributes)")
        return items
    
    def _create_attribute_text(self, name: str, info: AttributeInfo, context: str) -> str:
        """
        Create factual text description for embedding.
        
        CRITICAL: No semantic enhancement or domain knowledge.
        Only factual information about the attribute.
        
        Args:
            name: Attribute name
            info: Attribute information
            context: "node" or "edge"
            
        Returns:
            Factual text description suitable for embedding
        """
        text_parts = [
            f"{context} attribute named {name}",
            f"type {info.attribute_type}",
            f"coverage {info.coverage:.0%}"
        ]
        
        # Add sample values for context (let embedding model understand semantics)
        if hasattr(info, 'sample_values') and info.sample_values:
            # Take first 3 samples, convert to strings, filter None values
            samples = [str(v) for v in info.sample_values[:3] if v is not None]
            if samples:
                # Limit total sample text length
                sample_text = ' '.join(samples)
                if len(sample_text) > 100:  # Truncate if too long
                    sample_text = sample_text[:97] + "..."
                text_parts.append(f"examples {sample_text}")
        
        return " ".join(text_parts) 