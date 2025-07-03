#!/bin/bash

# Create directories if they do not exist
mkdir -p centroids
mkdir -p embeddings
mkdir -p inter
mkdir -p intra

# Activate the virtual environment
source env/bin/activate

# Run the generate_and_save_embeddings method
python -c "
from src.embedding_handler import EmbeddingHandler
handler = EmbeddingHandler()
handler.generate_and_save_embeddings(['clusters/gpt-4/LOC.json', 'clusters/gpt-4/ORG.json'], 'embeddings')
"

# Run the main test script
python src/test.py
