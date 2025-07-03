import os
from embedding_handler import EmbeddingHandler

# Paths
embedding_file = 'all-mpnet-base-v2'
output_dir = 'embeddings'
json_filepaths = [
    'clusters/gpt-4/LOC.json',
    'clusters/gpt-4/ORG.json',
]

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate and save embeddings
handler = EmbeddingHandler(model_name=embedding_file)
handler.generate_and_save_embeddings(json_filepaths, output_dir)

print(f"Embeddings generated and saved to {output_dir}/{embedding_file}.txt") 