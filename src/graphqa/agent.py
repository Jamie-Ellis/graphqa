"""
Universal Graph Retrieval Agent

Production-ready agent that can analyze any graph dataset using natural language.
Uses schema discovery and universal tools to provide intelligent analysis
across different domains (Amazon products, social networks, etc.).
"""

import json
import logging
import time
import warnings
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import networkx as nx

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from .observability import get_observability
from langchain_core.messages import SystemMessage
from langchain import hub

# Suppress OpenTelemetry warnings from Langfuse
warnings.filterwarnings("ignore", message="Calling end() on an ended span")

from .config import UniversalRetrieverConfig
from .loaders import AmazonProductLoader
from .tools import (
    UniversalGraphExplorer,
    UniversalGraphQuery, 
    UniversalGraphStats,
    UniversalNodeAnalyzer,
    UniversalAlgorithmSelector
)
from .data_structures import SchemaInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalRetrievalAgent:
    """
    Universal graph analysis agent that works with any dataset.
    
    Features:
    - Multi-dataset support (Amazon and any custom graph data)
    - Schema discovery and adaptive analysis
    - Natural language interface with intelligent tool selection
    - Universal tools that work across domains
    - Production logging and error handling
    """
    
    def __init__(self, 
                 dataset_name: str = "amazon",
                 config: Optional[UniversalRetrieverConfig] = None,
                 config_file: Optional[str] = None,
                 llm_model: Optional[str] = None,  # If None, read from config
                 temperature: Optional[float] = None,  # If None, read from config
                 verbose: bool = False):
        """
        Initialize the Universal Retrieval Agent.
        
        Args:
            dataset_name: Name of dataset to load ("amazon", etc.)
            config: Configuration object (takes precedence over config_file)
            config_file: Path to YAML config file (default: look for config.yaml)
            llm_model: LLM model to use (default: read from config, fallback to gpt-4o)
            temperature: Temperature for LLM responses (default: read from config, fallback to 0.1)
            verbose: Enable verbose tool output display
        """
        self.dataset_name = dataset_name
        
        # Load configuration with priority: config > config_file > auto-detect > defaults
        if config is not None:
            self.config = config
        elif config_file is not None:
            self.config = UniversalRetrieverConfig(config_file)
        else:
            # Auto-detect config.yaml in current directory
            from pathlib import Path
            default_config = Path("config.yaml")
            if default_config.exists():
                logger.info(f"📋 Loading configuration from {default_config}")
                self.config = UniversalRetrieverConfig(str(default_config))
            else:
                logger.info("📋 Using built-in default configuration")
                self.config = UniversalRetrieverConfig()
        
        # Read LLM settings from config if not explicitly provided
        if llm_model is None:
            llm_model = self.config.llm.model
        if temperature is None:
            temperature = self.config.llm.temperature
        
        self.verbose = verbose  # Add verbose mode for clean tool output
        self.graph = None
        self.schema = None
        self.tools = []
        self.agent = None
        self.agent_executor = None
        
        # Initialize LLM based on model type
        # Support both OpenAI (gpt-*) and Google (gemini-*) models
        if llm_model.startswith("gemini"):
            # Google Gemini models
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found in environment variables. "
                    "Please set it in your .env file or environment."
                )
            self.llm = ChatGoogleGenerativeAI(
                model=llm_model,
                temperature=temperature,
                max_output_tokens=8000,  # Gemini's max output tokens
                google_api_key=api_key
            )
            logger.info(f"Using Google Gemini model: {llm_model}")
        else:
            # OpenAI models (default)
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                max_tokens=10000,  # Increased output limit
                timeout=60        # Longer timeout for complex queries
            )
            logger.info(f"Using OpenAI model: {llm_model}")
        
        # Memory for conversation context (with size limit to prevent context overflow)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=8000  # Limit memory to prevent context overflow
        )
        
        logger.info(f"Universal Retrieval Agent initialized for dataset: {dataset_name}")
    
    def load_dataset(self, dataset_name: Optional[str] = None) -> bool:
        """
        Load a specific dataset and discover its schema.
        
        Args:
            dataset_name: Dataset to load (uses instance default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_name:
            self.dataset_name = dataset_name
        
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            
            # Get dataset configuration
            dataset_config = self.config.get_dataset_config(self.dataset_name)
            if dataset_config is None:
                # Use default configurations - preserved for backward compatibility
                if self.dataset_name == "amazon":
                    config_dict = {"test_mode": True}  # Safe default for unknown setups
                    loader = AmazonProductLoader(config_dict)
                elif self.dataset_name == "conversation_analysis":
                    config_dict = {}
                    from .loaders import ConversationAnalysisLoader
                    loader = ConversationAnalysisLoader(config_dict)
                else:
                    logger.error(f"Unknown dataset: {self.dataset_name}")
                    return False
            else:
                # Use configuration from config file - respects user settings
                config_dict = dataset_config.config.copy()
                if self.dataset_name == "amazon":
                    # No longer force test_mode - respect user's config.yaml settings
                    loader = AmazonProductLoader(config_dict)
                elif self.dataset_name == "conversation_analysis":
                    from .loaders import ConversationAnalysisLoader
                    loader = ConversationAnalysisLoader(config_dict)
                else:
                    logger.error(f"Unsupported dataset: {self.dataset_name}")
                    return False
            
            # Load the graph
            self.graph = loader.load_graph()
            
            if self.graph is None or self.graph.number_of_nodes() == 0:
                logger.error("Failed to load graph data")
                return False
            
            # Discover schema
            self.schema = loader.discover_schema(self.graph)
            
            logger.info(f"✅ Dataset loaded successfully:")
            logger.info(f"   📊 {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            logger.info(f"   🏷️  {len(self.schema.node_attributes)} node attributes discovered")
            logger.info(f"   🔗 {len(self.schema.edge_attributes)} edge attributes discovered")
            
            # Initialize tools with the loaded dataset
            self._initialize_tools()
            
            # Create the agent
            self._create_agent()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {str(e)}")
            return False
    
    def switch_dataset(self, new_dataset: str) -> bool:
        """
        Switch to a different dataset during the session.
        
        Args:
            new_dataset: Name of the new dataset to load
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Switching from {self.dataset_name} to {new_dataset}")
        
        # Clear current memory to avoid confusion
        self.memory.clear()
        
        # Load new dataset
        return self.load_dataset(new_dataset)
    
    def _initialize_tools(self):
        """Initialize universal tools with the loaded dataset"""
        if self.graph is None:
            raise RuntimeError("Graph must be loaded before initializing tools")
        
        # Create universal tools
        graph_explorer = UniversalGraphExplorer(
            graph=self.graph,
            dataset_name=self.dataset_name
        )
        
        # Set the schema for the graph explorer (needed for embedding search)
        object.__setattr__(graph_explorer, 'graph_schema', self.schema)
        
        # Reinitialize schema search now that we have the schema
        graph_explorer._init_schema_search()
        
        graph_query = UniversalGraphQuery(
            graph=self.graph,
            graph_schema=self.schema,
            dataset_name=self.dataset_name
        )
        
        graph_stats = UniversalGraphStats(
            graph=self.graph,
            graph_schema=self.schema,
            dataset_name=self.dataset_name
        )
        
        node_analyzer = UniversalNodeAnalyzer(
            graph=self.graph,
            graph_schema=self.schema,
            dataset_name=self.dataset_name
        )
        
        algorithm_selector = UniversalAlgorithmSelector(
            graph=self.graph,
            graph_schema=self.schema,
            dataset_name=self.dataset_name
        )
        
        # Create wrapper functions that handle JSON parameter parsing with clean display
        def make_tool_wrapper(tool_instance):
            """Create a wrapper that handles JSON parameter parsing with organized output display"""
            def wrapper(input_str: str) -> str:
                # Display tool execution start (only if verbose mode)
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"\n🔧 EXECUTING TOOL: {tool_instance.name}")
                    print(f"📝 Input: {input_str}")
                    print()  # Empty line after input
                
                try:
                    # Try to parse as JSON first
                    import json
                    params = json.loads(input_str)
                    
                    # Call the tool with parsed parameters
                    if hasattr(tool_instance, '_run'):
                        result = tool_instance._run(**params)
                    else:
                        result = tool_instance.run(**params)
                    
                    # Display organized result (only if verbose mode) 
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"📋 TOOL RESULT:")
                        # Try to format JSON nicely
                        try:
                            if isinstance(result, str) and (result.startswith('{') or result.startswith('[')):
                                import json
                                parsed = json.loads(result)
                                print(json.dumps(parsed, indent=2))
                            else:
                                print(result)
                        except:
                            print(result)
                        print("-" * 50)
                        print()  # Empty line after result
                    
                    return result
                        
                except json.JSONDecodeError:
                    # If not valid JSON, try treating as a simple query string
                    if hasattr(tool_instance, '_run'):
                        result = tool_instance._run(input_str)
                    else:
                        result = tool_instance.run(input_str)
                    
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"📋 TOOL RESULT:")
                        print(result)
                        print("-" * 50)
                        print()  # Empty line after result
                    
                    return result
                except Exception as e:
                    error_msg = f"Error executing tool: {str(e)}"
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"❌ TOOL ERROR: {error_msg}")
                        print("-" * 50)
                        print()  # Empty line after error
                    return error_msg
            
            return wrapper
        
        # Store references to original tool instances for direct access
        self.original_tools = {
            'graph_explorer': graph_explorer,
            'graph_query': graph_query, 
            'graph_stats': graph_stats,
            'node_analyzer': node_analyzer,
            'algorithm_selector': algorithm_selector
        }
        
        # Convert to LangChain Tools with proper parameter handling
        self.tools = [
            Tool(
                name=graph_explorer.name,
                description=graph_explorer.description,
                func=make_tool_wrapper(graph_explorer)
            ),
            Tool(
                name=graph_query.name,
                description=graph_query.description,
                func=make_tool_wrapper(graph_query)
            ),
            Tool(
                name=graph_stats.name,
                description=graph_stats.description,
                func=make_tool_wrapper(graph_stats)
            ),
            Tool(
                name=node_analyzer.name,
                description=node_analyzer.description,
                func=make_tool_wrapper(node_analyzer)
            ),
            Tool(
                name=algorithm_selector.name,
                description=algorithm_selector.description,
                func=make_tool_wrapper(algorithm_selector)
            )
        ]
        
        logger.info(f"✅ Initialized {len(self.tools)} universal tools")
    
    def _create_agent(self):
        """Create the ReAct agent with universal tools"""
        
        # Create a custom prompt that includes schema information
        schema_info = self._get_schema_summary()
        
        prompt_template = f"""You are a Universal Graph Analysis Assistant for {self.dataset_name} dataset.

CURRENT SCHEMA: {schema_info}

ANALYSIS WORKFLOW:
1. **Discover**: Use graph_explorer for schema discovery (unless question is very specific)
2. **Search**: Use universal_graph_query with exact JSON format for targeted searches  
3. **Analyze**: Apply stats/algorithms for insights
4. **Conclude**: Provide Final Answer after max 4 tool uses

CRITICAL: Use exact JSON format from tool descriptions. Start with schema discovery for new questions.

You have access to the following tools:

{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )
        
        # Create ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor with verbose output to show thinking process
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,  # Keep verbose to show agent thinking
            handle_parsing_errors=True,
            max_iterations=self.config.llm.max_iterations  # Configurable from config.yaml
        )
        
        logger.info("✅ Universal agent created successfully")
    
    def _get_schema_summary(self) -> str:
        """Get a concise summary of the dataset schema"""
        if not self.schema:
            return "Schema not yet discovered"
        
        # Node attributes summary
        node_attrs = []
        for attr_name, attr_info in list(self.schema.node_attributes.items())[:5]:  # Top 5
            node_attrs.append(f"{attr_name} ({attr_info.attribute_type.value})")
        
        # Edge attributes summary  
        edge_attrs = []
        for attr_name, attr_info in list(self.schema.edge_attributes.items())[:3]:  # Top 3
            edge_attrs.append(f"{attr_name} ({attr_info.attribute_type.value})")
        
        summary = f"""
        Nodes: {self.schema.node_count:,} | Edges: {self.schema.edge_count:,} | Directed: {self.schema.is_directed}
        Key Node Attributes: {', '.join(node_attrs)}
        Key Edge Attributes: {', '.join(edge_attrs) if edge_attrs else 'None'}
        """
        
        return summary.strip()
    
    def ask(self, question: str) -> str:
        """
        Ask a natural language question about the dataset.
        
        Args:
            question: Natural language question
            
        Returns:
            AI-generated response with analysis results
        """
        if self.agent_executor is None:
            return "❌ No dataset loaded. Please load a dataset first using load_dataset()."
        
        # Get observability instance
        obs = get_observability()
        
        try:
            start_time = time.time()
            logger.info(f"Processing question: {question}")
            
            # Prepare callbacks for LangChain (handles all tracing automatically)
            callbacks = []
            langchain_handler = obs.get_langchain_handler()
            if langchain_handler:
                callbacks.append(langchain_handler)
            
            # Execute the agent with observability
            response = self.agent_executor.invoke(
                {"input": question},
                config={"callbacks": callbacks} if callbacks else {}
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Question processed in {execution_time:.2f} seconds")
            
            # Check if we got a meaningful response
            output = response.get("output", "")
            
            # More precise failure detection - avoid false positives
            if (not output or 
                len(output.strip()) < 10 or 
                (output.startswith("Agent stopped") and len(output.strip()) < 50)):
                return f"""❌ I encountered some difficulties processing your question.

🔧 **Possible solutions:**
• Try rephrasing your question more specifically
• Use simpler terms or break complex questions into parts  
• Ask about dataset structure first: 'What does this dataset contain?'
• Try example questions: type 'examples' for suggestions

💡 **Quick diagnostic:** type 'status' to check system health"""
            
            # Check for format errors specifically
            if "Invalid Format" in output or "Missing 'Action:'" in output:
                return f"""❌ **Agent format error detected.**

🔧 **What happened:** The AI agent broke its reasoning format while analyzing your question.

💡 **Try these alternatives:**
• Rephrase your question more simply: "What products are most popular?"
• Ask for basic info first: "What does this dataset contain?"
• Use the 'schema' command to understand available data
• Try 'examples' for working question formats

🛠️ **Quick fix:** type 'reset' to clear memory and try again"""
            
            return output
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"""❌ **Error processing your question:** {str(e)}

🔧 **Troubleshooting steps:**
1. Check if your question is clear and specific
2. Try asking about the dataset structure: 'What does this data contain?'  
3. Use 'examples' command for sample questions
4. Type 'help' for comprehensive guidance"""
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded dataset"""
        if self.graph is None:
            return {"error": "No dataset loaded"}
        
        return {
            "dataset_name": self.dataset_name,
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "schema_summary": self._get_schema_summary(),
            "available_tools": [tool.name for tool in self.tools]
        }
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions appropriate for the current dataset"""
        if self.dataset_name == "amazon":
            return [
                "What are the most popular product categories?",
                "Find products with the highest ratings",
                "Show me products similar to a specific item",
                "What are the main product communities or clusters?",
                "Which products have the most connections?",
                "Analyze the relationship patterns in the data"
            ]
        else:
            return [
                "What does this dataset contain?",
                "Show me the overall structure",
                "Find the most important nodes",
                "What are the main communities or groups?",
                "Analyze the connectivity patterns",
                "What interesting patterns can you find?"
            ]
    
    def reset_conversation(self):
        """Reset the conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        obs = get_observability()
        return {
            "dataset_loaded": self.graph is not None,
            "dataset_name": self.dataset_name,
            "agent_ready": self.agent_executor is not None,
            "tools_count": len(self.tools),
            "memory_messages": len(self.memory.chat_memory.messages) if self.memory else 0,
            "observability_enabled": obs.is_enabled()
        }
    
    def shutdown(self):
        """Shutdown the agent and flush any pending traces"""
        obs = get_observability()
        if obs.is_enabled():
            logger.info("Flushing observability traces...")
            obs.flush()
        logger.info("Universal Retrieval Agent shutdown complete")


# LangGraph Platform Integration
# Export a LangGraph-compatible graph for deployment
def _create_langgraph_graph():
    """Create and return a LangGraph graph for the conversation analysis dataset."""
    from langgraph.prebuilt import create_react_agent
    from graphqa.agent import UniversalRetrievalAgent
    
    # Initialize GraphQA agent with conversation analysis dataset
    print("🔄 Initializing GraphQA for LangGraph Platform...")
    graphqa_agent = UniversalRetrievalAgent(
        dataset_name="conversation_analysis",
        verbose=False
    )
    graphqa_agent.load_dataset()
    
    # Convert GraphQA's LangChain tools to LangGraph-compatible format
    # GraphQA already uses LangChain Tool format, which works with LangGraph
    tools = graphqa_agent.tools
    llm = graphqa_agent.llm
    
    # Create a ReAct agent graph using LangGraph's prebuilt function
    graph = create_react_agent(llm, tools)
    
    print(f"✅ LangGraph agent created with {len(tools)} GraphQA tools")
    return graph

# Export for LangGraph Platform
graph = _create_langgraph_graph() 