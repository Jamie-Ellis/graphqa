"""
Smart context management for schema responses.

This module provides intelligent sample truncation and schema summarization
to prevent LLM context overflow while preserving meaningful information.
"""

import json
import logging
from typing import List, Dict, Any, Tuple
import re

from ..loaders.base_loader import SchemaInfo, AttributeInfo

logger = logging.getLogger(__name__)


class SmartSampleTruncator:
    """Intelligent sample truncation to prevent LLM context overflow"""
    
    def truncate_samples_intelligently(self, 
                                     values: List[Any], 
                                     max_total_chars: int = 500,
                                     max_sample_chars: int = 80) -> List[str]:
        """
        Truncate samples while preserving meaning and staying within budget.
        
        Strategy:
        1. Individual samples max 80 characters
        2. Total samples max 500 characters  
        3. Content-aware truncation based on data patterns
        
        Args:
            values: List of sample values
            max_total_chars: Maximum total characters for all samples
            max_sample_chars: Maximum characters per individual sample
            
        Returns:
            List of intelligently truncated sample strings
        """
        if not values:
            return []
        
        smart_samples = []
        total_chars = 0
        
        for value in values[:10]:  # Never process more than 10 samples
            if total_chars >= max_total_chars:
                break
                
            # Convert to string and apply intelligent truncation
            sample = self._truncate_single_sample(value, max_sample_chars)
            
            if total_chars + len(sample) <= max_total_chars:
                smart_samples.append(sample)
                total_chars += len(sample)
            else:
                # Try to fit a final truncated sample if there's room
                remaining = max_total_chars - total_chars
                if remaining > 20:  # Minimum useful sample size
                    truncated_final = sample[:remaining-3] + "..."
                    smart_samples.append(truncated_final)
                break
        
        logger.debug(f"Truncated {len(values)} samples to {len(smart_samples)} ({total_chars} chars)")
        return smart_samples
    
    def _truncate_single_sample(self, value: Any, max_chars: int) -> str:
        """Apply content-aware truncation to individual sample"""
        
        str_val = str(value)
        
        if len(str_val) <= max_chars:
            return str_val
        
        # Content-aware truncation strategies
        if self._looks_like_url(str_val):
            return self._truncate_url(str_val, max_chars)
        elif self._looks_like_id_or_arn(str_val):
            return self._truncate_id(str_val, max_chars)
        elif self._looks_like_json(str_val):
            return self._truncate_json(str_val, max_chars)
        elif self._looks_like_path(str_val):
            return self._truncate_path(str_val, max_chars)
        else:
            # Default: simple truncation
            return str_val[:max_chars-3] + "..."
    
    def _looks_like_url(self, text: str) -> bool:
        """Detect URL patterns"""
        return (text.startswith(('http://', 'https://', 'ftp://', 'www.')) or 
                '.com' in text or '.org' in text or '.net' in text)
    
    def _looks_like_id_or_arn(self, text: str) -> bool:
        """Detect ID or ARN patterns"""
        return (':' in text or 
                text.startswith(('arn:', 'aws:', 'k8s:', 'gcp:')) or
                (len(text) > 10 and '-' in text and text.replace('-', '').replace('_', '').isalnum()))
    
    def _looks_like_json(self, text: str) -> bool:
        """Detect JSON patterns"""
        return text.strip().startswith(('{', '['))
    
    def _looks_like_path(self, text: str) -> bool:
        """Detect file/directory path patterns"""
        return ('/' in text and (text.startswith('/') or '.' in text)) or '\\' in text
    
    def _truncate_url(self, url: str, max_chars: int) -> str:
        """Truncate URL showing domain and path structure"""
        if max_chars < 20:
            return url[:max_chars-3] + "..."
        
        # Try to preserve domain and show path structure
        if '://' in url:
            parts = url.split('://', 1)
            if len(parts) == 2:
                protocol, rest = parts
                if '/' in rest:
                    domain_path = rest.split('/', 1)
                    domain = domain_path[0]
                    path = domain_path[1] if len(domain_path) > 1 else ""
                    
                    preserved = f"{protocol}://{domain}"
                    remaining = max_chars - len(preserved) - 6  # Leave room for ".../"
                    
                    if remaining > 0 and path:
                        preserved += "/..." + path[-remaining:]
                    
                    return preserved[:max_chars-3] + "..." if len(preserved) > max_chars else preserved
        
        # Fallback
        return url[:max_chars-3] + "..."
    
    def _truncate_id(self, id_str: str, max_chars: int) -> str:
        """Truncate ID/ARN showing prefix and suffix"""
        if max_chars < 15:
            return id_str[:max_chars-3] + "..."
        
        # For IDs with colons (ARNs, etc.), show prefix and suffix
        if ':' in id_str:
            parts = id_str.split(':')
            if len(parts) >= 3:
                prefix = ':'.join(parts[:2])  # First two parts
                suffix = parts[-1]  # Last part
                
                if len(prefix) + len(suffix) + 5 <= max_chars:  # 5 for ":...:"
                    return f"{prefix}:...:{suffix}"
        
        # For other ID patterns, show beginning and end
        if len(id_str) > max_chars:
            prefix_len = (max_chars - 3) // 2
            suffix_len = max_chars - prefix_len - 3
            return f"{id_str[:prefix_len]}...{id_str[-suffix_len:]}"
        
        return id_str
    
    def _truncate_json(self, json_str: str, max_chars: int) -> str:
        """Truncate JSON showing structure"""
        if max_chars < 10:
            return json_str[:max_chars-3] + "..."
        
        try:
            # Try to parse and show structure
            data = json.loads(json_str)
            if isinstance(data, dict):
                keys = list(data.keys())[:3]  # First 3 keys
                structure = "{" + ", ".join(f'"{k}": ...' for k in keys)
                if len(keys) < len(data):
                    structure += ", ..."
                structure += "}"
                return structure if len(structure) <= max_chars else json_str[:max_chars-3] + "..."
            elif isinstance(data, list):
                return f"[...{len(data)} items...]"
        except:
            pass
        
        # Fallback
        return json_str[:max_chars-3] + "..."
    
    def _truncate_path(self, path: str, max_chars: int) -> str:
        """Truncate path showing directory structure"""
        if len(path) <= max_chars:
            return path
        
        # Show beginning and end of path
        if '/' in path:
            parts = path.split('/')
            if len(parts) > 2:
                start = parts[0] + '/' + (parts[1] if len(parts) > 2 else '')
                end = '/' + parts[-1]
                
                if len(start) + len(end) + 4 <= max_chars:  # 4 for "/.../"
                    return f"{start}/...{end}"
        
        # Fallback
        return path[:max_chars-3] + "..."


class CompactSchemaCreator:
    """Create compact schema summaries that always fit in LLM context"""
    
    def create_schema_summary(self, schema: SchemaInfo) -> Dict[str, Any]:
        """
        Create high-level schema overview guaranteed to fit in context.
        
        Target: <500 tokens total
        
        Args:
            schema: Full schema information
            
        Returns:
            Compact schema summary
        """
        summary = {
            "dataset_name": schema.dataset_name,
            "scale": f"{schema.node_count:,} nodes, {schema.edge_count:,} edges",
            "node_attributes_count": len(schema.node_attributes),
            "edge_attributes_count": len(schema.edge_attributes),
            "search_instruction": "Use search_schema_by_query('your question') to find relevant attributes"
        }
        
        # Add top attributes if space allows
        if len(schema.node_attributes) <= 20:  # Only for smaller schemas
            top_node_attrs = self.prioritize_attributes_mathematically(schema.node_attributes)[:5]
            summary["sample_node_attributes"] = [name for name, score in top_node_attrs]
        
        if len(schema.edge_attributes) <= 10:  # Only for smaller schemas
            top_edge_attrs = self.prioritize_attributes_mathematically(schema.edge_attributes)[:3]
            summary["sample_edge_attributes"] = [name for name, score in top_edge_attrs]
        
        return summary
    
    def prioritize_attributes_mathematically(self, 
                                           attributes: Dict[str, AttributeInfo]) -> List[Tuple[str, float]]:
        """
        Rank attributes by mathematical importance (no domain knowledge).
        
        Scoring factors:
        1. Coverage (higher = more important)
        2. Uniqueness ratio (moderate = most useful)
        3. Name patterns (generic patterns only)
        
        Args:
            attributes: Dictionary of attribute name to AttributeInfo
            
        Returns:
            List of (attribute_name, score) tuples sorted by score
        """
        scored_attributes = []
        
        for name, info in attributes.items():
            score = 0.0
            
            # Coverage score (0-40 points)
            score += info.coverage * 40
            
            # Uniqueness score (0-30 points) - moderate uniqueness is best
            if hasattr(info, 'unique_count') and info.coverage > 0:
                # Estimate total items from coverage
                estimated_total = max(info.unique_count / max(info.coverage, 0.01), 100)
                uniqueness_ratio = info.unique_count / estimated_total
                
                if 0.1 <= uniqueness_ratio <= 0.8:  # Sweet spot for analysis
                    score += 30
                elif uniqueness_ratio < 0.1:  # Too few unique values (not interesting)
                    score += 15
                else:  # Too many unique values (likely IDs)
                    score += 10
            
            # Name pattern score (0-20 points) - generic patterns only
            name_lower = name.lower()
            if any(pattern in name_lower for pattern in ['name', 'title', 'label']):
                score += 20  # Identification fields
            elif any(pattern in name_lower for pattern in ['type', 'category', 'kind', 'class']):
                score += 18  # Classification fields
            elif any(pattern in name_lower for pattern in ['status', 'state', 'condition']):
                score += 15  # Status fields
            elif any(pattern in name_lower for pattern in ['count', 'size', 'number', 'amount']):
                score += 12  # Numeric fields
            elif any(pattern in name_lower for pattern in ['id', 'identifier', 'key']):
                score += 8   # ID fields (less useful for analysis)
            elif any(pattern in name_lower for pattern in ['created', 'modified', 'updated', 'time']):
                score += 10  # Temporal fields
            
            # Penalty for very long names (likely auto-generated)
            if len(name) > 30:
                score -= 5
            
            scored_attributes.append((name, score))
        
        return sorted(scored_attributes, key=lambda x: x[1], reverse=True)
    
    def estimate_context_size(self, data: Any) -> int:
        """
        Estimate context size in characters for any data structure.
        
        Args:
            data: Data structure to estimate
            
        Returns:
            Estimated character count
        """
        try:
            return len(json.dumps(data, default=str))
        except:
            return len(str(data)) 