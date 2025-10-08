#!/usr/bin/env python3
"""
GraphQA Quick Start Script

This script helps new users get started with GraphQA quickly.
It checks dependencies, guides through setup, and runs a simple test.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Try to load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def print_banner():
    """Print welcome banner"""
    print("🚀 GraphQA Quick Start")
    print("=" * 30)
    print("Welcome to GraphQA - Natural Language Graph Analysis!")
    print("")

def check_python():
    """Check Python version"""
    print("🐍 Checking Python...")
    version = sys.version_info
    if version < (3, 8):
        print(f"❌ Python {version.major}.{version.minor} found, but 3.8+ required")
        print("   Please upgrade Python: https://python.org")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} (Good!)")
    return True

def check_virtual_env(auto_mode=False):
    """Check if we're in a virtual environment"""
    print("\n📦 Checking virtual environment...")
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("⚠️  Not in a virtual environment")
        print("   Recommendation: Create one with 'python3 -m venv venv && source venv/bin/activate'")
        if auto_mode:
            print("   Continuing in automated mode...")
            return True
        else:
            response = input("   Continue anyway? (y/N): ").lower()
            if response != 'y':
                return False
    else:
        print("✅ Virtual environment active")
    return True

def check_graphqa_installed():
    """Check if GraphQA is installed"""
    print("\n🔧 Checking GraphQA installation...")
    try:
        import graphqa
        print("✅ GraphQA is installed")
        return True
    except ImportError:
        print("❌ GraphQA not found")
        print("   Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
            print("✅ GraphQA installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install GraphQA")
            print("   Try manually: pip install -e .")
            return False

def check_api_key(auto_mode=False):
    """Check for OpenAI API key"""
    print("\n🔑 Checking OpenAI API key...")
    
    # Check if .env file exists
    env_file = Path(".env")
    env_example_file = Path("env.example")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        print("\n📝 To set up your API key:")
        print("   1. Go to https://platform.openai.com/api-keys")
        print("   2. Create a new API key")
        print("   3. Choose one of these options:")
        print("")
        
        if env_example_file.exists() and not env_file.exists():
            print("   Option A (Recommended): Use .env file")
            print("      cp env.example .env")
            print("      # Then edit .env and add your API key")
        elif env_file.exists():
            print("   Option A: Update your .env file")
            print("      # Edit .env and set OPENAI_API_KEY=your-key-here")
        else:
            print("   Option A: Create a .env file")
            print("      echo 'OPENAI_API_KEY=your-key-here' > .env")
        
        print("")
        print("   Option B: Set environment variable")
        print("      export OPENAI_API_KEY='your-key-here'")
        print("   Option C: Add to shell profile")
        print("      echo 'export OPENAI_API_KEY=\"your-key-here\"' >> ~/.bashrc")
        
        # Offer multiple options
        if auto_mode:
            print("   In automated mode - skipping interactive API key setup")
            print("   Please set up your API key manually:")
            if env_example_file.exists():
                print("      cp env.example .env && edit .env")
            else:
                print("      echo 'OPENAI_API_KEY=your-key-here' > .env")
            return False
        
        print("")
        print("   What would you like to do?")
        print("   1. Create/update .env file now")
        print("   2. I already have it in .env (reload)")
        print("   3. Set key for this session only") 
        print("   4. Skip (I'll set it up later)")
        
        while True:
            choice = input("\n   Choose option (1-4): ").strip()
            
            if choice == "1":
                key = input("   Enter your API key: ").strip()
                if key:
                    try:
                        with open(".env", "w") as f:
                            f.write(f"# GraphQA Environment Configuration\n")
                            f.write(f"OPENAI_API_KEY={key}\n")
                            f.write(f"\n# Optional: Add other configuration here\n")
                            f.write(f"# See env.example for more options\n")
                        print("✅ .env file created successfully")
                        # Reload environment variables
                        try:
                            from dotenv import load_dotenv
                            load_dotenv(override=True)
                        except ImportError:
                            pass
                        os.environ["OPENAI_API_KEY"] = key
                        return True
                    except Exception as e:
                        print(f"❌ Failed to create .env file: {e}")
                        continue
                else:
                    print("   No key entered, try again")
                    continue
                    
            elif choice == "2":
                print("   Reloading .env file...")
                try:
                    from dotenv import load_dotenv
                    load_dotenv(override=True)
                    reloaded_key = os.getenv("OPENAI_API_KEY")
                    if reloaded_key:
                        print("✅ API key loaded from .env file")
                        return True
                    else:
                        print("❌ No OPENAI_API_KEY found in .env file")
                        print("   Please check your .env file contains: OPENAI_API_KEY=your-key-here")
                        continue
                except ImportError:
                    print("❌ python-dotenv not available, cannot reload .env")
                    continue
                except Exception as e:
                    print(f"❌ Error reloading .env: {e}")
                    continue
                    
            elif choice == "3":
                key = input("   Enter your API key for this session: ").strip()
                if key:
                    os.environ["OPENAI_API_KEY"] = key
                    print("✅ API key set for this session")
                    return True
                else:
                    print("   No key entered, try again")
                    continue
                    
            elif choice == "4":
                print("   Skipping API key setup")
                return False
                
            else:
                print("   Invalid choice, please enter 1, 2, 3, or 4")
                continue
    else:
        # Mask the key for security
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        
        # Check where the key came from
        if env_file.exists():
            print(f"✅ API key found in .env file ({masked_key})")
        else:
            print(f"✅ API key found in environment ({masked_key})")
        return True

def run_basic_test():
    """Run a basic functionality test"""
    print("\n🧪 Running basic test...")
    try:
        from graphqa import GraphQA
        print("✅ GraphQA import successful")
        
        # Test initialization (this may fail with dataset loading, which is expected)
        agent = GraphQA(dataset_name="amazon")
        print("✅ Agent initialization successful")
        
        # Note: We don't test dataset loading since sample data may not be available
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_next_steps():
    """Show what to do next"""
    print("\n🎉 Quick start completed!")
    print("\n📖 What's next?")
    print("   1. Run the basic example:")
    print("      python examples/basic_usage.py")
    print("")
    print("   2. Check your .env file setup:")
    print("      cat .env  # Should contain your OPENAI_API_KEY")
    print("")
    print("   3. Customize performance settings:")
    print("      cat config.yaml  # Edit to change dataset loading limits")
    print("      # Templates available:")
    print("      #   cp config-templates/demo.yaml config.yaml (5K products, fast)")
    print("      #   cp config-templates/full-analysis.yaml config.yaml (200K+, slow)")
    print("")
    print("   4. Check documentation:")
    print("      docs/user-guide.md")
    print("")
    print("   5. Try with your own data:")
    print("      - See existing loaders in src/graphqa/loaders/")
    print("      - Create your own data loader")
    print("")
    print("   6. Join the community:")
    print("      - GitHub: https://github.com/catio-tech/graphqa")
    print("      - Issues: https://github.com/catio-tech/graphqa/issues")
    print("")
    print("💡 Tips:")
    print("   - Keep your .env file private and never commit it to git!")
    print("   - Edit config.yaml to customize dataset loading behavior")
    print("   - We ship in test mode by default to prevent memory overload")

def main():
    """Main quickstart function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphQA Setup and Quick Start")
    parser.add_argument("--auto", action="store_true", 
                       help="Run in automated mode (non-interactive)")
    parser.add_argument("--venv", action="store_true",
                       help="Create virtual environment if needed")
    args = parser.parse_args()
    
    # Set global automation mode
    global AUTOMATED_MODE
    AUTOMATED_MODE = args.auto
    
    print_banner()
    
    # Handle venv creation in automated mode
    if args.auto and args.venv:
        create_venv_if_needed()
    
    # Check all requirements
    checks = [
        ("Python version", check_python),
        ("Virtual environment", lambda: check_virtual_env(auto_mode=args.auto)),
        ("GraphQA installation", check_graphqa_installed),
        ("OpenAI API key", lambda: check_api_key(auto_mode=args.auto)),
        ("Basic functionality", run_basic_test)
    ]
    
    failed_checks = []
    for name, check_func in checks:
        if not check_func():
            failed_checks.append(name)
    
    if failed_checks:
        print(f"\n❌ Setup incomplete. Failed checks: {', '.join(failed_checks)}")
        if not args.auto:
            print("   Please address the issues above and run this script again.")
        sys.exit(1)
    
    show_next_steps()

def create_venv_if_needed():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("🐍 Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "venv"])
            print("✅ Virtual environment created")
            print("   Please activate it with: source venv/bin/activate")
            print("   Then run this script again.")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)

if __name__ == "__main__":
    # Global flag for automation mode
    AUTOMATED_MODE = False
    main() 