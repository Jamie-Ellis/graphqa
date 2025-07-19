"""
Amazon Product Loader for Universal Graph Retriever

Loads Amazon product catalog and recommendation networks with automatic schema discovery.
Handles product metadata, relationships (also-bought, also-viewed), and reviews.
"""

import json
import ast
import time
import gzip
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import networkx as nx

from .base_loader import BaseGraphLoader, LoaderError

logger = logging.getLogger(__name__)


class AmazonProductLoader(BaseGraphLoader):
    """
    Universal loader for Amazon product datasets.
    
    Supports both product metadata and relationship networks with automatic
    schema discovery and validation. Handles various Amazon data formats
    including compressed JSON files.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Amazon product loader.
        
        Config options:
        - metadata_file: Path to product metadata file
        - reviews_file: Path to reviews file  
        - test_mode: If True, load only a subset for testing
        - max_products: Maximum number of products to load
        - max_reviews: Maximum number of reviews to process
        """
        super().__init__(config)
        
        # Set default paths
        self.metadata_file = self.config.get(
            'metadata_file', 
            'data/datasets/amazon/meta_electronics_2018.json.gz'
        )
        self.reviews_file = self.config.get(
            'reviews_file',
            'data/datasets/amazon/reviews_electronics_2018.json.gz'
        )
        
        # Configuration
        self.test_mode = self.config.get('test_mode', False)
        self.max_products = self.config.get('max_products', 5000 if self.test_mode else None)
        self.max_reviews = self.config.get('max_reviews', 10000 if self.test_mode else None)
        
        self.logger.info(f"Amazon loader initialized - Test mode: {self.test_mode}")
    
    def load_graph(self) -> nx.MultiDiGraph:
        """Load Amazon product data into NetworkX graph"""
        graph = nx.MultiDiGraph()
        
        try:
            # Load product metadata first
            self._load_product_metadata(graph)
            
            # Load reviews and relationships
            self._load_reviews_and_relationships(graph)
            
            self.logger.info(
                f"Amazon graph loaded: {graph.number_of_nodes()} products, "
                f"{graph.number_of_edges()} relationships"
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Failed to load Amazon data: {e}")
            raise LoaderError(f"Amazon data loading failed: {e}")
    
    def _load_product_metadata(self, graph: nx.MultiDiGraph):
        """Load product metadata from Amazon metadata file"""
        metadata_path = Path(self.metadata_file)
        
        if not metadata_path.exists():
            self.logger.warning(f"Metadata file not found: {metadata_path}")
            return
        
        self.logger.info(f"Loading product metadata from {metadata_path}")
        
        # Handle both .gz and regular files
        if metadata_path.suffix == '.gz':
            file_opener = gzip.open
            mode = 'rt'
        else:
            file_opener = open
            mode = 'r'
        
        products_loaded = 0
        
        with file_opener(metadata_path, mode, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if self.max_products and products_loaded >= self.max_products:
                    break
                
                try:
                    # Parse JSON line
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Handle both json.loads and ast.literal_eval for Python dict format
                    try:
                        product = json.loads(line)
                    except json.JSONDecodeError:
                        try:
                            product = ast.literal_eval(line)
                        except (ValueError, SyntaxError):
                            self.logger.warning(f"Could not parse line {line_num}: {line[:100]}...")
                            continue
                    
                    # Extract product information
                    asin = product.get('asin')
                    if not asin:
                        continue
                    
                    # Build node attributes with safe field access
                    node_attrs = {
                        'asin': asin,
                        'product_title': product.get('title', product.get('product_title', '')),
                        'brand': product.get('brand', ''),
                        'price': self._safe_float(product.get('price')),
                        'average_rating': self._safe_float(product.get('average_rating')),
                        'review_count': self._safe_int(product.get('review_count')),
                        'description': product.get('description', ''),
                        'features': product.get('feature', []) if isinstance(product.get('feature'), list) else [],
                        'image_url': product.get('image', [''])[0] if isinstance(product.get('image'), list) else product.get('image', ''),
                        'sales_rank': self._safe_int(product.get('salesRank', {}).get('Electronics')) if isinstance(product.get('salesRank'), dict) else None
                    }
                    
                    # Handle categories
                    categories = product.get('category', [])
                    if isinstance(categories, list) and categories:
                        node_attrs['main_category'] = categories[0]
                        node_attrs['subcategory'] = categories[1] if len(categories) > 1 else ''
                        node_attrs['all_categories'] = categories
                    else:
                        node_attrs['main_category'] = 'Electronics'  # Default
                        node_attrs['subcategory'] = ''
                        node_attrs['all_categories'] = []
                    
                    graph.add_node(asin, **node_attrs)
                    
                    # Add relationships from 'related' field
                    related = product.get('related', {})
                    if isinstance(related, dict):
                        self._add_product_relationships(graph, asin, related)
                    
                    products_loaded += 1
                    
                    if products_loaded % 1000 == 0:
                        self.logger.info(f"Loaded {products_loaded} products...")
                
                except Exception as e:
                    self.logger.warning(f"Error processing product at line {line_num}: {e}")
                    continue
        
        self.logger.info(f"Loaded {products_loaded} products from metadata")
    
    def _add_product_relationships(self, graph: nx.MultiDiGraph, source_asin: str, related: Dict[str, List[str]]):
        """Add product relationships from related field"""
        relationship_types = {
            'also_bought': 'also_bought',
            'also_viewed': 'also_viewed', 
            'bought_together': 'bought_together',
            'buy_after_viewing': 'buy_after_viewing'
        }
        
        for rel_type, edge_type in relationship_types.items():
            related_products = related.get(rel_type, [])
            if isinstance(related_products, list):
                for target_asin in related_products:
                    if target_asin and target_asin != source_asin:
                        # Add edge even if target node doesn't exist yet
                        graph.add_edge(
                            source_asin, 
                            target_asin,
                            relationship_type=edge_type,
                            weight=1.0
                        )
    
    def _load_reviews_and_relationships(self, graph: nx.MultiDiGraph):
        """Load reviews to enhance product data and create user-based relationships"""
        reviews_path = Path(self.reviews_file)
        
        if not reviews_path.exists():
            self.logger.warning(f"Reviews file not found: {reviews_path}")
            return
        
        self.logger.info(f"Loading reviews from {reviews_path}")
        
        # Handle both .gz and regular files
        if reviews_path.suffix == '.gz':
            file_opener = gzip.open
            mode = 'rt'
        else:
            file_opener = open
            mode = 'r'
        
        reviews_processed = 0
        user_products = {}  # Track what products each user reviewed
        product_ratings = {}  # Track ratings for each product
        
        with file_opener(reviews_path, mode, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if self.max_reviews and reviews_processed >= self.max_reviews:
                    break
                
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse review data
                    try:
                        review = json.loads(line)
                    except json.JSONDecodeError:
                        try:
                            review = ast.literal_eval(line)
                        except (ValueError, SyntaxError):
                            continue
                    
                    asin = review.get('asin')
                    user_id = review.get('reviewerID')
                    rating = self._safe_float(review.get('overall'))
                    
                    if not asin or not user_id:
                        continue
                    
                    # Update product rating information if product exists
                    if graph.has_node(asin):
                        # Aggregate ratings for average calculation
                        if asin not in product_ratings:
                            product_ratings[asin] = []
                        if rating > 0:  # Only include valid ratings
                            product_ratings[asin].append(rating)
                    
                    # Track user product interactions for co-purchase analysis
                    if user_id not in user_products:
                        user_products[user_id] = []
                    user_products[user_id].append(asin)
                    
                    reviews_processed += 1
                    
                    if reviews_processed % 5000 == 0:
                        self.logger.info(f"Processed {reviews_processed} reviews...")
                
                except Exception as e:
                    self.logger.warning(f"Error processing review at line {line_num}: {e}")
                    continue
        
        # Update nodes with calculated average ratings
        self._update_product_ratings(graph, product_ratings)
        
        # Create co-purchase relationships based on users who reviewed multiple products
        self._create_user_based_relationships(graph, user_products)
        
        self.logger.info(f"Processed {reviews_processed} reviews")
    
    def _update_product_ratings(self, graph: nx.MultiDiGraph, product_ratings: Dict[str, List[float]]):
        """Update product nodes with calculated average ratings and review counts"""
        updated_count = 0
        
        for asin, ratings in product_ratings.items():
            if graph.has_node(asin) and ratings:
                # Calculate average rating
                avg_rating = sum(ratings) / len(ratings)
                review_count = len(ratings)
                
                # Update node attributes
                graph.nodes[asin]['average_rating'] = round(avg_rating, 2)
                graph.nodes[asin]['review_count'] = review_count
                updated_count += 1
        
        self.logger.info(f"Updated {updated_count} products with calculated average ratings")
    
    def _create_user_based_relationships(self, graph: nx.MultiDiGraph, user_products: Dict[str, List[str]]):
        """Create relationships between products based on user behavior"""
        co_purchase_count = {}
        
        for user_id, products in user_products.items():
            if len(products) > 1:  # User reviewed multiple products
                # Create co-purchase relationships
                for i, product1 in enumerate(products):
                    for product2 in products[i+1:]:
                        # Only create relationships between products that have metadata
                        if (graph.has_node(product1) and graph.has_node(product2) and
                            'product_title' in graph.nodes[product1] and 'product_title' in graph.nodes[product2]):
                            # Track co-purchase frequency
                            pair = tuple(sorted([product1, product2]))
                            co_purchase_count[pair] = co_purchase_count.get(pair, 0) + 1
        
        # Add edges for frequently co-purchased products
        for (product1, product2), count in co_purchase_count.items():
            if count >= 2:  # At least 2 users purchased both
                # Add bidirectional relationships
                graph.add_edge(
                    product1, product2,
                    relationship_type='co_purchased',
                    weight=count,
                    frequency=count
                )
                graph.add_edge(
                    product2, product1,
                    relationship_type='co_purchased', 
                    weight=count,
                    frequency=count
                )
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float"""
        if value is None or value == '':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_int(self, value) -> int:
        """Safely convert value to int"""
        if value is None or value == '':
            return 0
        try:
            return int(float(value))  # Handle string floats like "5.0"
        except (ValueError, TypeError):
            return 0
    
    def get_dataset_description(self) -> str:
        """Get description of the Amazon dataset"""
        return (
            "Amazon Product Network containing product metadata, customer reviews, "
            "and recommendation relationships. Includes product features, ratings, "
            "categories, and various relationship types like also-bought and also-viewed."
        )
    
    def get_sample_queries(self) -> List[str]:
        """Get sample queries appropriate for Amazon product data"""
        return [
            "What are the highest rated products?",
            "Find products under $50 with good reviews",
            "Show me popular items in the Electronics category", 
            "Which products are most often bought together?",
            "Find products with the most customer reviews",
            "What brands have the highest average ratings?",
            "Show products frequently viewed together",
            "Find the most connected products in the network",
            "What are the top-selling products by category?",
            "Identify products with strong recommendation patterns"
        ]
    
    def get_data_sources(self) -> List[str]:
        """Get list of data source files"""
        return [self.metadata_file, self.reviews_file]
    
    def get_dataset_name(self) -> str:
        """Get the dataset name"""
        return "Amazon Products" 