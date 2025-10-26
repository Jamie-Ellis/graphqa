"""
LangGraph deployment file for GraphQA

This file creates a LangGraph-compatible graph from GraphQA's existing agent.
It's separate from agent.py to avoid relative import issues when LangGraph loads it directly.
"""

from langgraph.prebuilt import create_react_agent
from graphqa.agent import UniversalRetrievalAgent

# Initialize GraphQA agent with conversation analysis dataset
print("🔄 Initializing GraphQA for LangGraph Platform...")
graphqa_agent = UniversalRetrievalAgent(
    dataset_name="conversation_analysis",
    verbose=False
)
graphqa_agent.load_dataset()

# Get GraphQA's tools and LLM
tools = graphqa_agent.tools
llm = graphqa_agent.llm

# Create LangGraph ReAct agent using GraphQA's tools
graph = create_react_agent(llm, tools)

print(f"✅ LangGraph agent created with {len(tools)} GraphQA tools")
