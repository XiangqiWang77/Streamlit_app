import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from py2neo import Graph

# 加载预训练的 NLP 模型进行实体提取
nlp = spacy.load("en_core_web_sm")
graph_db = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
from py2neo import Graph

# Connect to Neo4j
graph_db = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

# Load nodes with their IDs
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
#node_embeddings = {node: model.wv[node] for node in stellar_graph.nodes()}

edge_embeddings_query = """
    MATCH (a)-[r]->(b)
    RETURN a.name AS source_name, b.name AS target_name, r.embedding AS embedding
"""
edge_embeddings = graph_db.run(edge_embeddings_query).data()

# 示例数据：边的嵌入
#edge_embeddings = [
#    {'source_name': 'Hypanus americanus (Hildebrand & Schroeder, 1928)', 'target_name': 'Anna Maria island', 'embedding': [-0.17853116989135742, 0.4871777296066284, -0.36421728134155273, 0.6717486381530762, 0.3584253787994385, ...]},
#    {'source_name': 'Another Species', 'target_name': 'Another Location', 'embedding': [0.123, -0.234, 0.345, ...]},
#    # 添加更多边嵌入
#]

# 从问题中提取实体

def compute_entity_embedding(entity, model):
    # Tokenize the entity if necessary (e.g., split by spaces)
    tokens = entity.split()
    
    # Get embeddings for each token and average them
    embeddings = [model.wv[token] for token in tokens if token in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        # Handle out-of-vocabulary entities
        return np.random.rand(model.vector_size)
    

def extract_entities_from_question(question):
    doc = nlp(question)
    
    # Extract named entities
    named_entities = [ent.text for ent in doc.ents]
    
    # Initialize sets to store relevant entities
    animals = set()
    locations = set()
    
    # Define a set of entity labels that might correspond to locations
    location_labels = {'GPE', 'LOC'}  # Geopolitical Entity and Location
    
    # Iterate over named entities to categorize them
    for ent in doc.ents:
        if ent.label_ in location_labels:
            locations.add(ent.text)
        else:
            # Add named entities that could be animals if they are not categorized
            # In practice, further filtering or domain-specific adjustments might be needed
            animals.add(ent.text)
    
    # Extract noun phrases to include possible animal names or locations not captured by NER
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    # Combine and refine entities
    combined_entities = set(animals).union(set(locations), set(noun_phrases))
    
    # Filter out irrelevant entities based on context, if needed
    # For example, exclude common terms that are unlikely to be animals or locations
    relevant_entities = [entity for entity in combined_entities if len(entity.split()) > 1]
    
    return relevant_entities

# 计算与每个边嵌入的余弦相似度
def calculate_similarity(entity_pair, edge_embeddings, model):
    similarities = {}
    source_entity, target_entity = entity_pair
    
    # Compute or obtain embeddings for the source and target entities
    source_embedding = compute_entity_embedding(source_entity, model)
    target_embedding = compute_entity_embedding(target_entity, model)
    
    # Compute combined entity embedding (e.g., average or concatenation)
    combined_entity_embedding = (source_embedding + target_embedding) / 2
    
    for edge in edge_embeddings:
        edge_embedding = np.array(edge['embedding'])
        # Compute similarity between the combined entity embedding and edge embedding
        similarity = cosine_similarity([combined_entity_embedding], [edge_embedding])[0][0]
        similarities[(edge['source_name'], edge['target_name'])] = similarity
    
    return similarities

# 提取 top-k 相关边
def get_top_k_edges(similarities, k):
    sorted_edges = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_edges[:k]

# 主流程
def main(question, edge_embeddings, top_k=2):
    # 提取实体
    entities = extract_entities_from_question(question)
    
    print("entities",entities)
    # 生成实体对
    entity_pairs = [(entities[i], entities[j]) for i in range(len(entities)) for j in range(i+1, len(entities))]
    
    print("entity pairs", entity_pairs)
    # 计算每个实体对与边的相似度
    all_similarities = {}
    for entity_pair in entity_pairs:
        similarities = calculate_similarity(entity_pair, edge_embeddings, model)
        all_similarities.update(similarities)
    
    # 提取 top-k 相关边
    top_k_edges = get_top_k_edges(all_similarities, top_k)
    
    return top_k_edges

# 示例问题
question = "What are the interactions between Hypanus americanus and Anna Maria island?"

# 获取 top-k 相关边
top_k_edges = main(question, edge_embeddings, top_k=2)
print("Top-k Relevant Edges:")
for edge, similarity in top_k_edges:
    print(f"Edge: {edge}, Similarity: {similarity}")
