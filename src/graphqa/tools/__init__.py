"""
Universal Retriever Tools Package

Complete set of universal graph analysis tools that work with any graph dataset.
These tools replace architecture-specific tools with universal equivalents.
"""

from .graph_explorer import UniversalGraphExplorer
from .universal_query import UniversalGraphQuery
from .universal_stats import UniversalGraphStats
from .universal_analyzer import UniversalNodeAnalyzer
from .universal_algorithms import UniversalAlgorithmSelector

__all__ = [
    # Core universal tools
    "UniversalGraphExplorer",
    "UniversalGraphQuery", 
    "UniversalGraphStats",
    "UniversalNodeAnalyzer",
    "UniversalAlgorithmSelector",
]

# Tool categories for easy organization
CORE_TOOLS = [
    "UniversalGraphExplorer",
    "UniversalGraphQuery"
]

ANALYSIS_TOOLS = [
    "UniversalGraphStats",
    "UniversalNodeAnalyzer", 
    "UniversalAlgorithmSelector"
]

# Mapping for backward compatibility with architecture-specific tools
COMPATIBILITY_MAPPING = {
    # Old architecture-specific tool name -> New universal tool
    "GraphQueryTool": "UniversalGraphQuery",
    "ComponentAnalyzer": "UniversalNodeAnalyzer", 
    "GraphStatsTool": "UniversalGraphStats",
    "IntelligentAlgorithmSelector": "UniversalAlgorithmSelector",
} 