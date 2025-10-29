from graphqa.agent import UniversalRetrievalAgent

print("Testing updated loader with community and centrality data...\n")

agent = UniversalRetrievalAgent('conversation_analysis', verbose=False)
agent.load_dataset()

# Test person node attributes
fahim = agent.graph.nodes['Fahim Khan']
print("Fahim Khan attributes:")
print(f"  node_type: {fahim.get('node_type')}")
print(f"  community: {fahim.get('community')}")
print(f"  pagerank_centrality: {fahim.get('pagerank_centrality')}")
print(f"  betweenness_centrality: {fahim.get('betweenness_centrality')}")
print(f"  degree_centrality: {fahim.get('degree_centrality')}")
print(f"  total_messages: {fahim.get('total_messages')}")

# Check how many people have community data
people = [(n, d) for n, d in agent.graph.nodes(data=True) if d.get('node_type') == 'person']
people_with_community = [p for p in people if p[1].get('community') is not None]
people_with_centrality = [p for p in people if p[1].get('pagerank_centrality') is not None]

print(f"\n✅ People with community data: {len(people_with_community)}/{len(people)}")
print(f"✅ People with centrality data: {len(people_with_centrality)}/{len(people)}")

# Show community distribution
from collections import Counter
communities = [d.get('community') for n, d in people if d.get('community') is not None]
print(f"\nCommunity distribution:")
for comm, count in sorted(Counter(communities).items()):
    print(f"  Community {comm}: {count} people")

print("\n✅ Loader test complete!")
