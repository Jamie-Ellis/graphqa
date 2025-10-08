"""
Universal Graph Retriever Loaders Package

This package contains dataset loaders that implement the BaseGraphLoader interface,
enabling easy integration of diverse graph datasets with minimal code.

Available Loaders:
- AmazonProductLoader: Product catalog and recommendation networks
- BaseGraphLoader: Abstract interface for custom loaders
"""

from .base_loader import BaseGraphLoader, LoaderError
from .amazon_loader import AmazonProductLoader

__all__ = [
    "BaseGraphLoader",
    "LoaderError", 
    "AmazonProductLoader",
] 