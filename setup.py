"""
GraphQA: Natural Language Graph Analysis Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Dependencies defined here for backward compatibility
# Primary dependency definition is in pyproject.toml
requirements = [
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0", 
    "langchain-community>=0.3.0",
    "networkx>=3.0",
    "numpy>=1.24.0",
    "pandas>=1.5.0",
    "scikit-learn>=1.3.0",
    "sentence-transformers>=3.0.0",
    "faiss-cpu>=1.7.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
]

setup(
    name="graphqa",
    version="1.0.0",
    author="GraphQA Team", 
    author_email="contact@graphqa.dev",
    description="Natural Language Graph Analysis Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/catio-tech/graphqa",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License", 
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "graphqa=graphqa.cli:main",
        ],
    },
)
