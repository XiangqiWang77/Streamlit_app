from py2neo import Graph

# Connect to Neo4j
graph_db = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

# Load nodes with their IDs
nodes_query = "MATCH (n) RETURN n.name AS name, ID(n) AS id"
nodes = graph_db.run(nodes_query).data()

# Load edges with source and target node IDs
edges_query = "MATCH (a)-[r]->(b) RETURN a.name AS source_name, b.name AS target_name"
edges = graph_db.run(edges_query).data()

# Example format:
# nodes: [{'name': 'Node1', 'id': 1}, {'name': 'Node2', 'id': 2}]
# edges: [{'source_name': 1, 'target_name': 2}, {'source_name': 3, 'target_name': 4}]


import networkx as nx
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec

# Create the graph (G)
G = nx.Graph()

# Add your nodes and edges (from Neo4j as earlier)
for node in nodes:
    G.add_node(node['id'], name=node['name'])  # Assuming `node['id']` is an integer

for edge in edges:
    G.add_edge(edge['source_name'], edge['target_name'])

# Create a node mapping: map integer node IDs to strings
node_mapping = {node_id: str(node_id) for node_id in G.nodes()}
nx.relabel_nodes(G, node_mapping, copy=False)

# Convert the networkx graph to StellarGraph
stellar_graph = StellarGraph.from_networkx(G)

# Check the StellarGraph node IDs
#print(stellar_graph.node_ids())


# Generate random walks for Node2Vec
rw = BiasedRandomWalk(stellar_graph)
# Convert node IDs to strings for the walks
walks = rw.run(
    nodes=list(stellar_graph.nodes()),  # Convert node IDs to strings
    length=100,
    n=10,
    p=0.5,
    q=2.0
)

# Train Word2Vec to get node embeddings
model = Word2Vec(walks, vector_size=128, window=10, min_count=1, sg=1, workers=4)

# Create a dictionary for node embeddings
node_embeddings = {node: model.wv[node] for node in stellar_graph.nodes()}

import numpy as np

def edge_embedding(source_node, target_node, embeddings, method="average"):
    source_emb = embeddings[source_node]
    target_emb = embeddings[target_node]
    
    if method == "concatenate":
        return np.concatenate([source_emb, target_emb])
    elif method == "average":
        return np.mean([source_emb, target_emb], axis=0)

# Generate embeddings for each edge
edge_embeddings = {}
for edge in edges:
    source_name = edge['source_name']
    target_name = edge['target_name']
    edge_embeddings[(source_name, target_name)] = edge_embedding(source_name, target_name, node_embeddings)


for (source_name, target_name), embedding in edge_embeddings.items():
    query = """
    MATCH (a)-[r]->(b)
    WHERE a.name = $source_name AND b.name = $target_name
    SET r.embedding = $embedding
    """
    graph_db.run(query, source_name=source_name, target_name=target_name, embedding=embedding.tolist())

