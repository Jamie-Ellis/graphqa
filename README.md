# GraphQA: Natural Language Graph Analysis Framework

> **Ask questions about any graph in natural language - no SQL, no graph query languages, just plain English**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-green.svg)](https://langchain.com/)

GraphQA transforms any graph dataset into a natural language interface. Whether you're analyzing social networks, product catalogs, cloud architectures, or citation networks, simply ask questions in plain English and get intelligent answers.

## 🚀 Quick Start

```bash
pip install graphqa
```

```python
from graphqa import GraphQA

# Load any graph dataset
agent = GraphQA(dataset_name="my_graph")
agent.load_dataset()

# Ask questions in natural language  
response = agent.ask("What are the most connected nodes?")
print(response)

agent.shutdown()
```

## 🎯 What Makes GraphQA Special

- **🧠 Zero Configuration**: No domain knowledge required
- **🔍 Embedding-Based Discovery**: AI understands your data structure  
- **🤖 ReAct Agent**: Powered by LangChain reasoning framework
- **📊 Universal Tools**: Works across any graph dataset type
- **⚡ Fast**: Sub-50ms schema discovery with 90MB model
- **🔧 Production Ready**: Built-in observability and error handling

## 📚 Documentation

- [User Guide](docs/user-guide.md) - Complete usage documentation
- [Examples](examples/) - Real-world usage examples

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
