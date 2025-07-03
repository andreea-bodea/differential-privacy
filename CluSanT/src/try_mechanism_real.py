import os
import sys
import json
import numpy as np
from clusant import CluSanT
from embedding_handler import EmbeddingHandler
import time
from tabulate import tabulate

# Paths
embedding_file = 'all-mpnet-base-v2'
embeddings_path = f'embeddings/{embedding_file}.txt'
loc_clusters_path = 'clusters/gpt-4/LOC.json'
org_clusters_path = 'clusters/gpt-4/ORG.json'

# Check if embeddings file exists
if not os.path.exists(embeddings_path):
    print(f"Embeddings file '{embeddings_path}' not found. Please generate it first.")
    sys.exit(1)

# Load clusters
with open(loc_clusters_path, 'r') as f:
    loc_clusters = json.load(f)
with open(org_clusters_path, 'r') as f:
    org_clusters = json.load(f)

# Merge clusters and lowercase all words
clusters = {}
cluster_id = 0
for cluster in [loc_clusters, org_clusters]:
    for words in cluster.values():
        clusters[cluster_id] = [w.lower() for w in words]
        cluster_id += 1

# Load embeddings
embedding_handler = EmbeddingHandler(model_name=embedding_file)
embeddings = embedding_handler.load_embeddings(embeddings_path)

# Instantiate CluSanT
clusant = CluSanT(
    embedding_file=embedding_file,
    embeddings=embeddings,
    epsilon=1.0,
    num_clusters=len(clusters),
    K=1,
    dp_type='metric',
)
# Patch clusters in the instance
del clusant.clusters
clusant.clusters = clusters

# Sample text
sample_text = "I visited Paris and met with CompanyX in Berlin."
sensitive_words = ['paris', 'companyx', 'berlin']

# Time the CluSanT mechanism
start = time.time()
anonymized_text = sample_text
for word in sensitive_words:
    replacement = clusant.replace_word(word)
    anonymized_text = anonymized_text.replace(word.capitalize(), replacement.capitalize() if replacement else 'XXXXX')
    anonymized_text = anonymized_text.replace(word, replacement if replacement else 'XXXXX')
elapsed = time.time() - start

print("Original text:", sample_text)
print("Anonymized text:", anonymized_text)

# Print timing table
performance_table = [["CluSanT", f"{elapsed:.6f}"]]
print("\nPerformance Table:")
print(tabulate(performance_table, headers=["Method", "Total Time (s)"], tablefmt="grid")) 