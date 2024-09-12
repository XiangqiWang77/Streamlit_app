from sentence_transformers import SentenceTransformer
from py2neo import Graph
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Connect to Neo4j
graph_db = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get question embedding
question = "What birds live in Alachua?"
question_embedding = model.encode(question)

question_embedding=question_embedding.reshape(1, -1)
question_embedding=np.tile(question_embedding, (128, 1))
print(f"Original question embedding shape: {question_embedding.shape}")
pca = PCA(n_components=128)
pca.fit(question_embedding)
question_embedding_reduced = pca.transform(question_embedding)

# Check the shape after dimensionality reduction
print(f"Reduced question embedding shape: {question_embedding_reduced.shape}")

# Retrieve edge embeddings from Neo4j
edge_embeddings_query = """
    MATCH (a)-[r]->(b)
    RETURN a.name AS source_name, b.name AS target_name, r.embedding AS embedding
"""
edge_embeddings = graph_db.run(edge_embeddings_query).data()

# Extract edge embeddings and names
edge_names = [f"{record['source_name']}->{record['target_name']}" for record in edge_embeddings]
edge_embeddings_array = np.array([record['embedding'] for record in edge_embeddings])
print(f"Reduced edge embedding shape: {edge_embeddings_array.shape}")

similarities = []
for i, edge_emb in enumerate(edge_embeddings_array):
    edge_emb=np.tile(edge_emb, (1, 1))
    
    similarity = cosine_similarity(edge_emb, question_embedding_reduced)[0][0]
    similarities.append((edge_names[i], similarity))

# Sort edges by similarity in descending order and select top K
K = 5  # Number of top edges to select
top_k_edges = sorted(similarities, key=lambda x: x[1], reverse=True)[:K]

# Print top K edges
print(f"Top {K} edges:")
for edge, similarity in top_k_edges:
    print(f"Edge: {edge}, Similarity: {similarity}")
