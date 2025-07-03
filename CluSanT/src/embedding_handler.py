import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingHandler:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _load_json_data(self, filepath):
        """Load words from a JSON file where values are lists of words."""
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        words = set()
        for key in data.keys():
            # Convert all words to lowercase before adding to the set
            words.update(word.lower() for word in data[key])
        return words

    def _is_float(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _save_embeddings(self, data, filepath):
        """Save embeddings to a file."""
        with open(filepath, "a", encoding="utf-8") as file:
            for word, embedding in data.items():
                if embedding is not None:
                    embedding_str = " ".join(map(str, embedding))
                    file.write(f"{word} {embedding_str}\n")

    def get_embedding(self, word):
        """Generate embedding for a word."""
        return self.model.encode(word)

    def load_embeddings(self, filepath):
        """Load embeddings from a file."""
        embeddings = {}
        with open(filepath, "r", encoding="utf-8") as f:
            # Read the first line to determine the embedding start index and dimension
            first_line = f.readline().strip()
            tokens = first_line.split()
            embedding_start_index = next(
                i for i, token in enumerate(tokens) if self._is_float(token)
            )
            embedding_dim = len(tokens) - embedding_start_index
            print(f"Embedding dimension: {embedding_dim}")

            # Process all lines including the first one
            f.seek(0)
            for line in f:
                values = line.strip().split()
                word = " ".join(values[:-embedding_dim])
                vector = list(map(float, values[-embedding_dim:]))
                assert len(vector) == embedding_dim, "Embedding dimension mismatch"
                embeddings[word] = vector

        return embeddings

    def generate_and_save_embeddings(self, json_filepaths, output_dir):
        """Process JSON files, generate embeddings, and save them."""
        words = set()
        for filepath in json_filepaths:
            words.update(self._load_json_data(filepath))

        embeddings = {}
        for word in tqdm(words, desc="Generating embeddings"):
            if word not in embeddings:
                embedding = self.get_embedding(word)
                if embedding is not None:
                    embeddings[word] = embedding

        embeddings_filepath = f"{output_dir}/{self.model_name}.txt"
        self._save_embeddings(embeddings, embeddings_filepath)
