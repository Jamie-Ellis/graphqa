"""
Universal Graph Retriever Loaders Package

This package contains dataset loaders that implement the BaseGraphLoader interface,
enabling easy integration of diverse graph datasets with minimal code.

Available Loaders:
- AmazonProductLoader: Product catalog and recommendation networks
- ArchitectureLoader: Cloud infrastructure topology
- BaseGraphLoader: Abstract interface for custom loaders
"""

from .base_loader import BaseGraphLoader, LoaderError
from .amazon_loader import AmazonProductLoader
from .architecture_loader import ArchitectureLoader

__all__ = [
    "BaseGraphLoader",
    "LoaderError", 
    "AmazonProductLoader",
    "ArchitectureLoader"
] 