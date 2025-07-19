"""
Query processing system for schema search.

This module provides domain-agnostic query cleaning and processing using
sklearn's stop words without any hardcoded domain knowledge.
"""

import re
import logging
from typing import List, Dict, Any

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback minimal stop words if sklearn not available
    ENGLISH_STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the'
    }

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Process user queries for schema search without domain assumptions"""
    
    def __init__(self):
        """Initialize with sklearn stop words (no hardcoded domain knowledge)"""
        
        if SKLEARN_AVAILABLE:
            # Use sklearn's professionally maintained stop words
            self.stop_words = set(ENGLISH_STOP_WORDS)
            logger.info("Using sklearn stop words for query processing")
        else:
            # Fallback to minimal stop words
            self.stop_words = ENGLISH_STOP_WORDS
            logger.warning("sklearn not available, using minimal stop words")
        
        # Add common query words that don't add semantic value
        self.stop_words.update({
            'what', 'how', 'where', 'when', 'why', 'which', 'who',
            'show', 'find', 'get', 'list', 'tell', 'me', 'about',
            'can', 'could', 'would', 'should', 'do', 'does', 'did',
            'please', 'help', 'want', 'need', 'like', 'look'
        })
    
    def clean_query_text(self, query: str) -> str:
        """
        Clean query text without any domain-specific assumptions.
        
        Uses sklearn stop words - no hardcoded domain knowledge.
        
        Args:
            query: Raw user query
            
        Returns:
            Cleaned query text suitable for embedding
        """
        if not query or not query.strip():
            return ""
        
        # Basic cleaning
        text = query.lower().strip()
        
        # Remove punctuation but preserve meaningful characters like hyphens
        # Keep: letters, numbers, spaces, hyphens, underscores
        text = re.sub(r'[^\w\s\-_]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize and remove stop words
        words = text.split()
        meaningful_words = [
            word for word in words 
            if (word not in self.stop_words and 
                len(word) > 2 and  # Filter very short words
                not word.isdigit())  # Filter standalone numbers
        ]
        
        cleaned = " ".join(meaningful_words)
        
        logger.debug(f"Query cleaning: '{query}' → '{cleaned}'")
        return cleaned
    
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract search intent without domain knowledge.
        
        Returns basic query analysis for logging/debugging purposes.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query analysis metadata
        """
        clean_text = self.clean_query_text(query)
        
        # Basic pattern detection (no domain assumptions)
        analysis = {
            "original_query": query,
            "cleaned_query": clean_text,
            "word_count": len(clean_text.split()) if clean_text else 0,
            "character_count": len(clean_text),
            "has_numeric": bool(re.search(r'\d+', query)),
            "has_comparison": bool(re.search(r'(high|low|top|best|worst|>|<|=|greater|less|more|most)', query.lower())),
            "has_aggregation": bool(re.search(r'(count|sum|average|total|distribution|stats)', query.lower())),
            "is_question": query.strip().endswith('?'),
            "too_short": len(clean_text.split()) < 2 if clean_text else True
        }
        
        return analysis
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate query for schema search suitability.
        
        Args:
            query: User query
            
        Returns:
            Validation result with suggestions
        """
        if not query or not query.strip():
            return {
                "valid": False,
                "reason": "Empty query",
                "suggestion": "Please provide a question about the data"
            }
        
        analysis = self.extract_query_intent(query)
        
        if analysis["too_short"]:
            return {
                "valid": False,
                "reason": "Query too short after cleaning",
                "suggestion": "Try a more specific question with descriptive words",
                "cleaned_query": analysis["cleaned_query"]
            }
        
        if analysis["word_count"] > 20:
            return {
                "valid": True,
                "warning": "Very long query may not match well",
                "suggestion": "Consider simplifying to key concepts",
                "analysis": analysis
            }
        
        return {
            "valid": True,
            "analysis": analysis
        } 