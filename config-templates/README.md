# GraphQA Configuration Templates

This directory contains pre-configured templates for different use cases. Copy the appropriate template to `config.yaml` in the project root.

## Available Templates

### 🚀 demo.yaml (Recommended for New Users)
```bash
cp config-templates/demo.yaml config.yaml
```

**Best for:**
- First-time users and demos
- Quick testing and experimentation  
- Presentations and tutorials
- Development and debugging

**Performance:**
- **Products**: 5,000 (vs 200,000+ full)
- **Load time**: 10-30 seconds
- **Memory**: ~200MB
- **CPU**: Light usage

### 🔥 full-analysis.yaml (For Research & Production)
```bash
cp config-templates/full-analysis.yaml config.yaml  
```

**Best for:**
- Research and academic analysis
- Production deployments
- Complete dataset exploration
- Advanced analytics projects

**Performance:**
- **Products**: 200,000+ (complete dataset)
- **Load time**: 2-5 minutes  
- **Memory**: ~2GB
- **CPU**: Heavy usage during loading

## Custom Configuration

You can also edit `config.yaml` directly or create your own template:

```yaml
datasets:
  amazon_products:
    config:
      test_mode: false      # true = limited, false = full dataset
      max_products: 25000   # Custom limit (or null for unlimited)
      max_reviews: 50000    # Custom limit (or null for unlimited)
```

## Performance Comparison

| Setting | Products | Reviews | Load Time | Memory | Best For |
|---------|----------|---------|-----------|---------|----------|
| Demo | 5K | 10K | 30s | 200MB | Learning, demos |
| Custom | 25K | 50K | 1-2min | 500MB | Balanced analysis |
| Full | 200K+ | 1M+ | 5min | 2GB | Research, production |

## Switching Configurations

```bash
# Switch to demo mode (fast)
cp config-templates/demo.yaml config.yaml

# Switch to full analysis (slow but complete) 
cp config-templates/full-analysis.yaml config.yaml

# Verify your current settings
grep -A 3 "test_mode\|max_products" config.yaml
```

Need help? Check the main [README.md](../README.md) for more details. 