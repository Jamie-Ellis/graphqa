"""
Schema discovery and embedding-based search system.

This package provides intelligent schema discovery capabilities that enable
query-driven attribute discovery without any hardcoded domain knowledge.
"""

from .embedder import SchemaEmbedder, SchemaItemCreator
from .query_processor import QueryProcessor  
from .context_manager import SmartSampleTruncator, CompactSchemaCreator
from .search_engine import QuerySchemaSearcher

__all__ = [
    "SchemaEmbedder",
    "SchemaItemCreator", 
    "QueryProcessor",
    "SmartSampleTruncator",
    "CompactSchemaCreator",
    "QuerySchemaSearcher"
] 