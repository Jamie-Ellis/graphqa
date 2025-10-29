# GraphQA Usage Guide - Conversation Network Analysis

**System Status**: ✅ Working (Community & Centrality data integrated)

## Quick Reference

### Dataset Overview
- **Total Messages**: 31,249 (REALTALK: 8,944 English + WhatsApp: 22,305 Dutch)
- **People**: 26 speakers with full network metrics
- **Communities**: 9 detected communities (Louvain algorithm)
- **Relationships**: 40 TALKS_TO edges with classification
- **Graph Structure**: 31,275 nodes, 85,479 edges

### Node Types
1. **Person nodes** (26): `node_type='person'`
2. **Message nodes** (31,249): `node_type='message'`

## Successful Query Patterns

### 1. Dataset Overview Queries

**"Tell me about the dataset"**
- ✅ Works: Returns comprehensive overview
- Response includes: node count, edge count, schema summary
- Best for: Initial exploration

**"Who are the most active people?"**
- ✅ Works: Sorts by `total_messages` attribute
- Returns: Top 10 people with message counts
- Example response: "[user73]: 9407 messages"

### 2. Community Queries

**"What are the main communities?"**
- ✅ Works: Uses `community` attribute (integer 0-8)
- Returns: Community distribution with member counts
- Note: Communities are **integers** (0, 1, 2...), not strings ("Community 0")

**Query format:**
```json
{"operation": "count_by_attribute", "attribute": "community"}
```

**"Find people in community 0"**
```json
{"operation": "filter_nodes", "filters": {"node_type": "person", "community": 0}}
```

### 3. Relationship Queries

**"Who has planning relationships?"**
- ✅ Works: Filters TALKS_TO edges by `relationship_class='planning'`
- Other classes: `supportive`, `collaborative`, `peer`, `hierarchical`, `casual`, `help_seeking`

**"Who does Fahim Khan talk to?"**
```json
{"operation": "get_neighbors", "node_id": "Fahim Khan"}
```

### 4. Centrality/Influence Queries

**"Who are the most influential people?"**
- ✅ Works: Uses `pagerank_centrality` attribute
- Also available: `betweenness_centrality`, `degree_centrality`
- All centrality metrics are **precomputed** and stored on person nodes

**Query format:**
```json
{"operation": "range_search", "attribute": "pagerank_centrality", "min_value": 0}
```

### 5. Message Content Queries

**"Find messages about meetings"**
```json
{"operation": "search_text", "query": "meeting", "fields": ["text"]}
```

**Important**: Always use `"query"` parameter, not `"text"`!

## Critical Attribute Names

### Person Node Attributes ✅
```python
{
  'node_type': 'person',           # NOT 'type'!
  'name': 'Fahim Khan',
  'total_messages': 844,
  'community': 0,                  # NOT 'community_id'! (integer)
  'pagerank_centrality': 0.0385,
  'betweenness_centrality': 0.0,
  'degree_centrality': 0.16,
  'commitment_count': 731,
  'question_ratio': 0.09,
  'answer_ratio': 0.143,
  'avg_msg_length': 95.1,
  'dataset': 'REALTALK',
  'unique_topics': 25
}
```

### Message Node Attributes ✅
```python
{
  'node_type': 'message',          # NOT 'type'!
  'text': 'Hey good afternoon...',
  'speaker': 'Fahim Khan',
  'is_question': False,
  'is_answer': False,
  'message_type': 'neither',
  'topic_id': 5,
  'topic_label': 'Planning',
  'has_commitment': True,
  'commitment_type': 'Meeting',
  'commitment_confidence': 0.85
}
```

### TALKS_TO Edge Attributes ✅
```python
{
  'relationship_type': 'TALKS_TO',
  'frequency': 247,                # NOT 'message_count'!
  'questions_asked': 14,
  'answers_given': 33,
  'commitments_made': 192,
  'shared_topics': 16,
  'avg_response_time': 18055.88,
  'reciprocity': True,
  'relationship_class': 'planning' # planning/supportive/collaborative/etc.
}
```

## Common Pitfalls & Fixes

### ❌ Wrong: `"type": "person"`
### ✅ Right: `"node_type": "person"`

### ❌ Wrong: `"community_id": 0`
### ✅ Right: `"community": 0`

### ❌ Wrong: `"message_count": 100`
### ✅ Right: `"frequency": 100`

### ❌ Wrong: Community as string "Community 0"
### ✅ Right: Community as integer `0`

## Tool Selection Guide

### Use `graph_explorer` for:
- Schema discovery (`discover_schema`)
- Finding people/messages (`find_by_attribute`)
- Counting by category (`count_by_attribute`)
- Text search in messages (`search_text`)
- Network navigation (`get_neighbors`)

### Use `universal_graph_stats` for:
- Centrality rankings (`stat_type="centrality"`)
- Community distributions (`stat_type="distribution", "groupby": ["community"]`)
- Dataset summary (`stat_type="summary"`)
- Network topology (`stat_type="topology"`)

### Use `universal_graph_query` for:
- Complex multi-attribute searches (`find_by_pattern`)
- Similarity queries (`find_similar`)
- Neighborhood exploration (`explore_neighborhood`)
- Aggregations (`aggregate_by_attribute`)

### Use `universal_node_analyzer` for:
- Deep dive on specific person (`analyze_node`)
- Centrality metrics for one node (`node_metrics`)
- Relationship patterns (`node_relationships`)
- Finding similar people (`node_similarity`)

### Use `universal_algorithm_selector` for:
- Path finding (`shortest_paths`)
- Network topology analysis (`network_topology`)
- Anomaly detection (`anomaly_detection`)

## Response Format Examples

### Successful Community Query
```
Community distribution:
- Community 0: 3 people
- Community 1: 4 people
- Community 2: 3 people
...
```

### Successful Centrality Query
```
Most influential people (PageRank):
1. Fahim Khan (0.0385)
2. Muhhamed (0.0312)
3. Akib (0.0289)
...
```

### Successful Relationship Query
```
Planning relationships:
- Fahim Khan → Muhhamed (247 messages, 192 commitments)
- Akib → Emi (156 messages, 89 commitments)
...
```

## Best Practices

### 1. Start with Schema Discovery
Always begin with `discover_schema` to understand available attributes:
```json
{"operation": "discover_schema"}
```

### 2. Use Specific Attribute Names
Refer to the "Critical Attribute Names" section above. The system is **case-sensitive** and **exact-match**.

### 3. Filter by Node Type First
When searching for people or messages, always filter by `node_type`:
```json
{"filters": {"node_type": "person", "community": 0}}
```

### 4. Leverage Precomputed Metrics
- Centrality scores are **already computed** - no need to recalculate
- Community assignments are **already assigned** - just filter
- Relationship classifications are **already done** - just query

### 5. Use Range Search for Numeric Queries
For "most active", "most influential", etc.:
```json
{"operation": "range_search", "attribute": "total_messages", "min_value": 0}
```
This returns results sorted by value (descending).

### 6. Combine Multiple Filters
```json
{
  "operation": "filter_nodes",
  "filters": {
    "node_type": "person",
    "community": 0,
    "total_messages": {"$gt": 100}
  }
}
```

## Troubleshooting

### Query Fails: "Could not find attribute"
- ✅ Check spelling: `node_type` not `type`
- ✅ Check case: attributes are lowercase
- ✅ Run `discover_schema` first to see available attributes

### Query Fails: "No results found"
- ✅ Check value type: community is `0` (int) not `"Community 0"` (string)
- ✅ Check node_type filter is included
- ✅ Verify the attribute exists on that node type

### Query Returns Partial Results
- ✅ Increase `limit` parameter (default: 50)
- ✅ Use more specific filters
- ✅ Check if sorting is applied

## Advanced Use Cases

### Find All People in a Specific Community
```json
{
  "operation": "filter_nodes",
  "filters": {"node_type": "person", "community": 0},
  "limit": 100
}
```

### Find Most Committed People (by commitment count)
```json
{
  "operation": "range_search",
  "attribute": "commitment_count",
  "min_value": 100,
  "limit": 10
}
```

### Find Planning Relationships with High Frequency
```json
{
  "query_type": "find_by_relationship",
  "parameters": {
    "relationship_type": "TALKS_TO",
    "min_connections": 50
  }
}
```
Then filter by `relationship_class='planning'`.

### Explore Someone's Network
```json
{
  "operation": "get_neighbors",
  "node_id": "Fahim Khan",
  "depth": 2,
  "limit": 50
}
```

## System Architecture

```
ConvoKit Corpus → NetworkX MultiDiGraph → GraphQA Tools → LangGraph Agent → Chat API
     ↑                      ↑                    ↑
  Combined_bert      Community/Centrality    5 Universal Tools
  (31,249 msgs)      CSVs (notebook 09)      (Schema-aware)
```

## Data Lineage

1. **Source**: `conversation-analysis/data/processed/combined_bert/`
2. **Enrichment**: `conversation-analysis/data/analysis_results/`
   - `community_assignments.csv` (26 people → 9 communities)
   - `centrality_rankings.csv` (26 people → PageRank/Betweenness/Degree)
3. **Loading**: `graphqa/src/graphqa/loaders/conversation_loader.py`
4. **Graph**: NetworkX MultiDiGraph with 31,275 nodes, 85,479 edges
5. **Tools**: 5 schema-aware tools with automatic attribute discovery
6. **Agent**: LangGraph ReAct agent with Gemini 2.5 Flash
7. **API**: `/runs/stream` endpoint with SSE streaming

## Performance Notes

- **Graph loading**: ~10 seconds (includes community/centrality CSV loading)
- **Schema discovery**: < 1 second (cached after first load)
- **Simple queries**: < 1 second
- **Complex aggregations**: 1-3 seconds
- **Text search**: 2-5 seconds (searches 31K messages)

## Update Log

### 2025-10-29: Community & Centrality Integration ✅
- Added CSV loading for community assignments (26 people → 9 communities)
- Added CSV loading for centrality metrics (PageRank, Betweenness, Degree)
- Updated loader to attach metadata to person nodes
- Verified all 26 people have both community and centrality data
- Increased node attributes from 24 → 28 (added 4 new fields)

### Known Working Queries ✅
- "Tell me about the dataset" → Full overview
- "Who are the most active people?" → Top 10 by total_messages
- "What are the main communities?" → 9 communities with counts
- "Who has planning relationships?" → TALKS_TO edges filtered
- "Who does Fahim Khan talk to?" → Neighbor exploration

## Support & Debugging

### Enable Verbose Logging
```python
agent = UniversalRetrievalAgent('conversation_analysis', verbose=True)
```

### Check Node Attributes
```python
agent.load_dataset()
print(agent.graph.nodes['Fahim Khan'])
```

### Inspect Tool Responses
Check LangGraph server logs for tool invocation details.

### Common Error Messages

**"Could not retrieve the main communities"**
- ✅ Fixed: Use `community` attribute, not `community_id`

**"No attribute 'type' found"**
- ✅ Fixed: Use `node_type` attribute, not `type`

**"Expected string, got integer"**
- ✅ Fixed: Community is integer (0-8), not string ("Community 0")

---

**Last Updated**: 2025-10-29  
**System Version**: GraphQA with Community/Centrality Integration  
**Status**: ✅ Production Ready
