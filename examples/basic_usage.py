"""
GraphQA Basic Usage Example
"""

import os

def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return
    
    from graphqa import GraphQA
    
    print("🎯 GraphQA Basic Usage Example")
    print("=" * 40)
    
    # Initialize GraphQA
    agent = GraphQA(dataset_name="amazon", verbose=True)
    
    # Load dataset
    print("\n📊 Loading dataset...")
    success = agent.load_dataset()
    
    if not success:
        print("❌ Dataset loading failed")
        return
    
    # Example questions
    questions = [
        "What types of nodes exist?",
        "How many products are there?", 
        "What's the average rating?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🤔 Question {i}: {question}")
        try:
            response = agent.ask(question)
            print(f"💡 Answer: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    agent.shutdown()
    print("\n✅ Example completed!")

if __name__ == "__main__":
    main()
