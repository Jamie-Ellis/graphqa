"""
Configuration system for Universal Graph Retriever

Provides centralized configuration management for datasets, tools, and system settings.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env file in current directory and parent directories
    load_dotenv(override=False)  # Don't override existing env vars
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset"""
    name: str
    description: str
    loader_class: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)


@dataclass
class ToolConfig:
    """Configuration for analysis tools"""
    enabled: bool = True
    max_results: int = 50
    cache_enabled: bool = True
    performance_mode: str = "balanced"  # fast, balanced, thorough


@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout_seconds: int = 30
    max_iterations: int = 4  # Agent reasoning iterations before stopping


@dataclass
class PerformanceConfig:
    """Performance and resource configuration"""
    max_graph_size: int = 100000
    max_memory_mb: int = 2048
    parallel_processing: bool = True
    cache_directory: str = "data/cache"
    log_level: str = "INFO"


class UniversalRetrieverConfig:
    """
    Main configuration class for Universal Graph Retriever.
    
    Handles loading from YAML files and provides access to all
    configuration sections with sensible defaults.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        self.datasets: Dict[str, DatasetConfig] = {}
        self.tools = ToolConfig()
        self.llm = LLMConfig()
        self.performance = PerformanceConfig()
        
        # Load configuration if file provided
        if config_file:
            self.load_from_file(config_file)
        else:
            self._set_defaults()
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_file}, using defaults")
            self._set_defaults()
            return
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Load dataset configurations
            datasets_config = config_data.get('datasets', {})
            for name, dataset_data in datasets_config.items():
                self.datasets[name] = DatasetConfig(
                    name=dataset_data.get('name', name),
                    description=dataset_data.get('description', ''),
                    loader_class=dataset_data.get('loader_class', ''),
                    enabled=dataset_data.get('enabled', True),
                    config=dataset_data.get('config', {}),
                    data_sources=dataset_data.get('data_sources', [])
                )
            
            # Load tool configuration
            tools_config = config_data.get('tools', {})
            self.tools = ToolConfig(
                enabled=tools_config.get('enabled', True),
                max_results=tools_config.get('max_results', 50),
                cache_enabled=tools_config.get('cache_enabled', True),
                performance_mode=tools_config.get('performance_mode', 'balanced')
            )
            
            # Load LLM configuration
            llm_config = config_data.get('llm', {})
            self.llm = LLMConfig(
                model=llm_config.get('model', 'o3-mini'),
                temperature=llm_config.get('temperature', 0.1),
                max_tokens=llm_config.get('max_tokens', 2000),
                timeout_seconds=llm_config.get('timeout_seconds', 30),
                max_iterations=llm_config.get('max_iterations', 4)
            )
            
            # Load performance configuration
            perf_config = config_data.get('performance', {})
            self.performance = PerformanceConfig(
                max_graph_size=perf_config.get('max_graph_size', 100000),
                max_memory_mb=perf_config.get('max_memory_mb', 2048),
                parallel_processing=perf_config.get('parallel_processing', True),
                cache_directory=perf_config.get('cache_directory', 'data/cache'),
                log_level=perf_config.get('log_level', 'INFO')
            )
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values"""
        # Default dataset configurations
        self.datasets = {
            'amazon_products': DatasetConfig(
                name='Amazon Products',
                description='Amazon product catalog and recommendation networks',
                loader_class='AmazonProductLoader',
                enabled=True,
                config={
                    'metadata_file': 'data/datasets/amazon/meta_electronics_2018.json.gz',
                    'reviews_file': 'data/datasets/amazon/reviews_electronics_2018.json.gz',
                    'test_mode': False,
                    'max_products': None,
                    'max_reviews': None
                },
                data_sources=[
                    'data/datasets/amazon/meta_electronics_2018.json.gz',
                    'data/datasets/amazon/reviews_electronics_2018.json.gz'
                ]
            ),
            'architecture': DatasetConfig(
                name='Cloud Architecture',
                description='Infrastructure components and dependencies',
                loader_class='ArchitectureLoader',
                enabled=True,
                config={
                    'architecture_file': 'data/input/architecture_states/tenant-org_9cTqhmQdCFGo8tzO-202506051140.json',  # Must be set at runtime
                    'validate_references': True
                },
                data_sources=[]  # Set at runtime
            )
        }
        
        # Default configurations already set in dataclass definitions
        logger.info("Using default configuration")
    
    def get_dataset_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """Get configuration for a specific dataset"""
        return self.datasets.get(dataset_name)
    
    def get_enabled_datasets(self) -> List[str]:
        """Get list of enabled dataset names"""
        return [name for name, config in self.datasets.items() if config.enabled]
    
    def add_dataset(self, name: str, config: DatasetConfig):
        """Add or update a dataset configuration"""
        self.datasets[name] = config
    
    def save_to_file(self, config_file: str):
        """Save current configuration to YAML file"""
        config_data = {
            'datasets': {
                name: {
                    'name': config.name,
                    'description': config.description,
                    'loader_class': config.loader_class,
                    'enabled': config.enabled,
                    'config': config.config,
                    'data_sources': config.data_sources
                }
                for name, config in self.datasets.items()
            },
            'tools': {
                'enabled': self.tools.enabled,
                'max_results': self.tools.max_results,
                'cache_enabled': self.tools.cache_enabled,
                'performance_mode': self.tools.performance_mode
            },
            'llm': {
                'model': self.llm.model,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens,
                'timeout_seconds': self.llm.timeout_seconds
            },
            'performance': {
                'max_graph_size': self.performance.max_graph_size,
                'max_memory_mb': self.performance.max_memory_mb,
                'parallel_processing': self.performance.parallel_processing,
                'cache_directory': self.performance.cache_directory,
                'log_level': self.performance.log_level
            }
        }
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_file}: {e}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check dataset configurations
        for name, dataset in self.datasets.items():
            if not dataset.name:
                issues.append(f"Dataset '{name}' missing name")
            if not dataset.loader_class:
                issues.append(f"Dataset '{name}' missing loader_class")
        
        # Check performance limits
        if self.performance.max_graph_size <= 0:
            issues.append("max_graph_size must be positive")
        
        if self.performance.max_memory_mb <= 0:
            issues.append("max_memory_mb must be positive")
        
        # Check LLM configuration
        if self.llm.temperature < 0 or self.llm.temperature > 2:
            issues.append("temperature should be between 0 and 2")
        
        if self.llm.max_tokens <= 0:
            issues.append("max_tokens must be positive")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'datasets': {name: vars(config) for name, config in self.datasets.items()},
            'tools': vars(self.tools),
            'llm': vars(self.llm),
            'performance': vars(self.performance)
        } 