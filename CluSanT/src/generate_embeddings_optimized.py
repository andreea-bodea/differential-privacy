# Optimized for single-process, batched GPU embedding. Ray removed for speed.
import os
import json
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2"
CACHE_PATH = "embeddings/all-mpnet-base-v2_cache.json"
OUTPUT_PATH = "embeddings/all-mpnet-base-v2.txt"
CLUSTER_FILES = ["clusters/gpt-4/LOC.json", "clusters/gpt-4/ORG.json"]

# (Optional) For higher Hugging Face rate limits, login with `huggingface-cli login` or set HF_TOKEN in your environment.
# Example: export HF_TOKEN=your_token_here

def load_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Cache file {cache_path} is corrupted or unreadable. Recreating it.")
            return {}
    return {}

def save_cache(cache, cache_path):
    with open(cache_path, 'w') as f:
        json.dump(cache, f)

def main():
    print("Checking if model is cached (downloading if necessary)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("Model is cached and ready.")

    # 1. Load all unique words from cluster files
    words = set()
    for cluster_file in CLUSTER_FILES:
        with open(cluster_file, 'r') as f:
            cluster = json.load(f)
            for word_list in cluster.values():
                words.update([w.lower() for w in word_list])
    words = list(words)

    # 2. Load cache and filter out already-embedded words
    cache = load_cache(CACHE_PATH)
    words_to_embed = [w for w in words if w not in cache]

    # 3. Batch embedding calls
    batch_size = 128
    for i in range(0, len(words_to_embed), batch_size):
        batch = words_to_embed[i:i+batch_size]
        embeddings = model.encode(batch, batch_size=32, show_progress_bar=True)
        for word, emb in zip(batch, embeddings):
            cache[word] = emb.tolist()
        save_cache(cache, CACHE_PATH)  # Save progress incrementally

    # 4. Save all embeddings to output file (word + space-separated floats per line)
    with open(OUTPUT_PATH, 'w') as f:
        for word, emb in cache.items():
            f.write(f"{word} {' '.join(map(str, emb))}\n")

if __name__ == "__main__":
    main() 