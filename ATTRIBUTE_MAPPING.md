# GraphQA Graph Loader Analysis - FINDINGS

## ✅ What's Working Correctly

### Text Handling
- ✅ Using `utt.text` (correct) - this is the standard ConvoKit text field
- ✅ NOT using `utt.meta['clean_text']` - clean_text is a lazy-loaded auxiliary field
- ✅ Truncating to 200 chars for graph storage
- ✅ Storing full_text_length separately

### Node Type Attributes
- ✅ Using `node_type='person'` for speakers
- ✅ Using `node_type='message'` for utterances
- ✅ All metadata correctly pulled from `utt.meta.get()`

### Edge Attributes  
- ✅ TALKS_TO edges using `frequency` (not message_count)
- ✅ TALKS_TO edges using `relationship_class` for planning/supportive/collaborative
- ✅ All other edge types correctly structured

## ❌ What's Missing

### Community Information
**CRITICAL ISSUE**: Person nodes don't have community assignments!

**Why it's missing:**
1. Communities were calculated in notebook 09 using NetworkX Louvain
2. Results saved to `data/analysis_results/community_assignments.csv`
3. Loader doesn't read this CSV file
4. ConvoKit corpus doesn't have community metadata stored in it

**CSV Structure:**
```
person,community
Akib,0
Fahim Khan,0
Muhhamed,0
Emi,1
Kevin,1
...
```

**Impact:**
- ❌ Query "what are the main communities?" fails
- ❌ Cannot filter people by community
- ❌ Community-based analysis impossible

### Centrality Metrics
**ISSUE**: Person nodes don't have centrality scores!

**Why they're missing:**
1. Centralities calculated in notebook 09 using NetworkX
2. Results saved to `data/analysis_results/centrality_rankings.csv`
3. Loader doesn't read this CSV file
4. ConvoKit corpus doesn't have centrality metadata

**CSV Structure:**
```
person,pagerank_centrality,betweenness_centrality,degree_centrality,...
Fahim Khan,0.0523,0.1234,0.0876,...
```

**Impact:**
- ❌ Cannot query "most influential people" directly from node attributes
- ❌ Tool descriptions reference centrality attributes that don't exist
- ✅ Can calculate on-the-fly, but slower

## 🔧 Required Fixes

### Priority 1: Add Community Data to Loader

**File**: `conversation_loader.py`
**Method**: `_add_speaker_nodes()`

**Changes needed:**
```python
def __init__(self, config: Dict[str, Any] = None):
    super().__init__(config)
    self.corpus_path = self.config.get('corpus_path', '...')
    
    # NEW: Load community assignments
    self.analysis_results_path = Path(self.corpus_path).parent.parent / 'analysis_results'
    self.communities = self._load_communities()

def _load_communities(self) -> Dict[str, int]:
    """Load community assignments from CSV"""
    import pandas as pd
    csv_path = self.analysis_results_path / 'community_assignments.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return dict(zip(df['person'], df['community']))
    return {}

def _add_speaker_nodes(self, corpus, graph: nx.MultiDiGraph):
    for speaker in corpus.iter_speakers():
        utts = list(speaker.iter_utterances())
        
        graph.add_node(
            speaker.id,
            node_type='person',
            name=speaker.id,
            total_messages=len(utts),
            # ... existing attributes ...
            community=self.communities.get(speaker.id, None)  # NEW
        )
```

### Priority 2: Update Tool Descriptions

**Files to update:**
1. `graph_explorer.py`
2. `universal_query.py`
3. `universal_stats.py`
4. `universal_analyzer.py`
5. `universal_algorithms.py`

**Changes:**
- Replace `type='person'` → `node_type='person'`
- Replace `community_id` → `community`
- Remove references to `pagerank_centrality`, `betweenness_centrality`, `degree_centrality` on nodes
- Note that centrality needs to be computed on-demand

### Priority 3 (Optional): Add Centrality Data

If we want centrality scores on nodes (faster queries):
- Load `centrality_rankings.csv` in loader
- Add pagerank_centrality, betweenness_centrality, degree_centrality to person nodes
- Update tool descriptions to reflect availability

## Testing After Fixes

```bash
# 1. Update loader
# 2. Restart LangGraph
langgraph dev --allow-blocking

# 3. Test queries
"what are the main communities?"
→ Should return community distribution

"find people in community 0"
→ Should list Akib, Fahim Khan, Muhhamed

"who has planning relationships?"
→ Should find TALKS_TO edges with relationship_class='planning'
```

## Summary

**Text Handling**: ✅ Perfect (using utt.text correctly)
**Node Structure**: ✅ Perfect (using node_type)
**Edge Structure**: ✅ Perfect (using frequency, relationship_class)
**Communities**: ❌ Missing (need to load CSV)
**Centrality**: ⚠️ Not on nodes (compute on-demand or load from CSV)
