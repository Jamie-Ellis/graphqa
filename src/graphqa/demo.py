"""
GraphQA Interactive Demo

Run with: python -m graphqa.demo
"""

import sys
import warnings

# Suppress warnings for clean demo experience
warnings.filterwarnings("ignore", message="Calling end() on an ended span")

def run_interactive_demo(dataset_name: str = "amazon"):
    """Run the interactive GraphQA demo"""
    
    print("🎯 " + "="*60)
    print("   GraphQA: Natural Language Graph Analysis Framework")
    print("="*64)
    print()
    print("💡 Ask questions about your graph data in plain English!")
    print("   Examples:")
    print("   • 'What are the most connected nodes?'")
    print("   • 'Find communities in the graph'")
    print("   • 'What's the average degree?'")
    print("   • 'Show me nodes with highest centrality'")
    print()
    
    try:
        from .agent import UniversalRetrievalAgent
        
        # Initialize agent
        print(f"🔧 Initializing GraphQA with {dataset_name} dataset...")
        agent = UniversalRetrievalAgent(dataset_name=dataset_name, verbose=True)
        
        # Load dataset
        print("📊 Loading graph dataset...")
        success = agent.load_dataset()
        
        if not success:
            print("❌ Failed to load dataset. Please check your data files.")
            return
            
        print("✅ GraphQA ready! Type 'help' for commands or 'quit' to exit.")
        print("-" * 60)
        
        # Interactive loop
        while True:
            try:
                query = input("\n🤔 Ask GraphQA: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'help':
                    print_help()
                    continue
                elif query.lower() == 'examples':
                    print_examples()
                    continue
                
                # Process the query
                print(f"\n🔍 Processing: {query}")
                response = agent.ask(query)
                print(f"\n📋 Answer:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                
        agent.shutdown()
        print("\n✅ GraphQA session ended.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Failed to start GraphQA: {e}")

def print_help():
    """Print help information"""
    print("\n📚 GraphQA Commands:")
    print("  help      - Show this help message")
    print("  examples  - Show example questions")
    print("  quit      - Exit GraphQA")

def print_examples():
    """Print example questions"""
    print("\n🎯 Example Questions:")
    print("  • What types of nodes and relationships exist?")
    print("  • How many nodes and edges are in the graph?")
    print("  • What are the most connected nodes?")
    print("  • Find communities or clusters")
    print("  • What's the average degree distribution?")

def main():
    """Main entry point for the demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphQA Interactive Demo")
    parser.add_argument("--dataset", default="amazon", help="Dataset to load")
    
    args = parser.parse_args()
    run_interactive_demo(args.dataset)

if __name__ == "__main__":
    main()
