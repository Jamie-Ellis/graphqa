"""
Simple Langfuse observability integration for Universal Retrieval Agent.

This module provides the easiest way to add LLM observability to the agent.
"""

import os
import logging
import warnings
from typing import Optional

# Try to import Langfuse - if not available, observability will be disabled
try:
    from langfuse import Langfuse, get_client
    from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
    # Suppress OpenTelemetry span warnings from Langfuse integration
    warnings.filterwarnings("ignore", "Calling end() on an ended span", module="opentelemetry.sdk.trace")
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    get_client = None
    CallbackHandler = None

logger = logging.getLogger(__name__)

class UniversalObservability:
    """Simple observability wrapper for Universal Retrieval Agent."""
    
    def __init__(self, 
                 public_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 host: Optional[str] = None):
        """
        Initialize Langfuse observability.
        
        Args:
            public_key: Langfuse public key (or set LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (or set LANGFUSE_SECRET_KEY env var) 
            host: Langfuse host (or set LANGFUSE_HOST env var, defaults to cloud)
        """
        self.enabled = False
        self.handler = None
        self.client = None
        
        if not LANGFUSE_AVAILABLE:
            logger.info("📊 Langfuse not available - observability disabled (install with: pip install langfuse)")
            return
        
        try:
            # Setup Langfuse client
            if public_key and secret_key:
                Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host or "http://localhost:3000"  # Default to local
                )
            
            # Get client instance
            self.client = get_client()
            
            # Test connection
            if self.client.auth_check():
                self.handler = CallbackHandler()
                self.enabled = True
                logger.info("✅ Langfuse observability enabled")
            else:
                logger.warning("❌ Langfuse auth failed - observability disabled")
                
        except Exception as e:
            logger.warning(f"⚠️ Langfuse setup failed: {e} - observability disabled")
    
    def get_langchain_handler(self):
        """Get LangChain callback handler for tracing."""
        return self.handler if self.enabled else None
    
    def trace_agent_run(self, name: str = "universal-agent-query"):
        """Create a trace context for agent runs."""
        # Not used - LangChain CallbackHandler handles all tracing automatically
        return DummyContext()
    
    def is_enabled(self) -> bool:
        """Check if observability is enabled."""
        return self.enabled
    
    def flush(self):
        """Flush any pending traces."""
        if self.enabled and self.client:
            self.client.flush()


class DummyContext:
    """Dummy context manager when observability is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def update_trace(self, **kwargs):
        pass


# Global instance - configure once, use everywhere
_global_observability = None

def get_observability() -> UniversalObservability:
    """Get global observability instance."""
    global _global_observability
    if _global_observability is None:
        _global_observability = UniversalObservability()
    return _global_observability

def configure_observability(public_key: str = None, 
                          secret_key: str = None, 
                          host: str = None):
    """Configure global observability settings."""
    global _global_observability
    _global_observability = UniversalObservability(
        public_key=public_key,
        secret_key=secret_key, 
        host=host
    ) 