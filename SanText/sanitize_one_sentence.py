import numpy as np
from tqdm import tqdm
from spacy.lang.en import English
from collections import Counter
import unicodedata
import os
import pickle
# Import core logic from SanText.py
from SanText import cal_probability, SanText, SanText_plus, SanText_init, SanText_plus_init
from tabulate import tabulate
import time

# --- Utility functions (from utils.py) ---
def word_normalize(text):
    return unicodedata.normalize('NFD', text)

def get_vocab_SST2_cached(data_dir, tokenizer, tokenizer_type="word", cache_path="vocab_cache.pkl"):
    if os.path.exists(cache_path):
        print(f"Loading vocabulary from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            vocab, words, sensitive_words, sensitive_words2id = pickle.load(f)
        return vocab, words, sensitive_words, sensitive_words2id
    else:
        print("Building vocabulary from SST-2...")
        vocab = Counter()
        for split in ['train', 'dev']:
            data_file_path = os.path.join(data_dir, split + ".tsv")
            num_lines = sum(1 for _ in open(data_file_path))
            with open(data_file_path, 'r') as csvfile:
                next(csvfile)
                for line in tqdm(csvfile, total=num_lines - 1):
                    line = line.strip().split("\t")
                    text = line[0]
                    tokenized_text = [token.text for token in tokenizer(text)]
                    for token in tokenized_text:
                        vocab[token] += 1
        words = [key for key, _ in vocab.most_common()]
        sensitive_word_count = int(0.5 * len(vocab))
        sensitive_words = words[-sensitive_word_count - 1:]
        sensitive_words2id = {word: k for k, word in enumerate(sensitive_words)}
        with open(cache_path, "wb") as f:
            pickle.dump((vocab, words, sensitive_words, sensitive_words2id), f)
        return vocab, words, sensitive_words, sensitive_words2id

# --- Main script ---
if __name__ == "__main__":
    # Parameters
    glove_path = "data/glove.840B.300d.txt"
    filtered_glove_path = "data/glove.filtered.txt"
    data_dir = "data/SST-2/"
    epsilon = 15.0
    p = 0.2
    sensitive_word_percentage = 0.5
    example_sentence = "The movie was absolutely wonderful and inspiring."
    vocab_cache_path = "vocab_cache.pkl"

    # Tokenizer
    tokenizer = English()
    # Build or load vocab from SST-2
    vocab, words, sensitive_words, sensitive_words2id = get_vocab_SST2_cached(data_dir, tokenizer, tokenizer_type="word", cache_path=vocab_cache_path)
    vocab_set = set(vocab.keys())

    # Use filtered GloVe if available
    if os.path.exists(filtered_glove_path):
        used_glove_path = filtered_glove_path
        print(f"Using filtered GloVe file: {filtered_glove_path}")
    else:
        used_glove_path = glove_path
        print(f"Using full GloVe file: {glove_path}")

    # --- GloVe Embedding Caching ---
    glove_cache_path = "glove_filtered_cache.pkl"
    if os.path.exists(glove_cache_path):
        print(f"Loading filtered GloVe from cache: {glove_cache_path}")
        with open(glove_cache_path, "rb") as f:
            word2id, sword2id, id2sword, all_word_embed, sensitive_word_embed, all_words = pickle.load(f)
    else:
        # Load GloVe embeddings for vocab
        print("Loading GloVe embeddings...")
        word2id = {}
        sword2id = {}
        id2sword = {}
        all_word_embed = []
        sensitive_word_embed = []
        all_words = []
        sensitive_count = 0
        all_count = 0
        with open(used_glove_path) as f:
            for row in tqdm(f, desc="GloVe"):
                content = row.rstrip().split(' ')
                cur_word = word_normalize(content[0])
                if cur_word in vocab and cur_word not in word2id:
                    word2id[cur_word] = all_count
                    all_words.append(cur_word)
                    all_count += 1
                    emb = [float(i) for i in content[1:]]
                    all_word_embed.append(emb)
                    if cur_word in sensitive_words2id:
                        sword2id[cur_word] = sensitive_count
                        id2sword[sensitive_count] = cur_word
                        sensitive_count += 1
                        sensitive_word_embed.append(emb)
        all_word_embed = np.array(all_word_embed, dtype='f')
        sensitive_word_embed = np.array(sensitive_word_embed, dtype='f')
        # Save to cache
        with open(glove_cache_path, "wb") as f:
            pickle.dump((word2id, sword2id, id2sword, all_word_embed, sensitive_word_embed, all_words), f)

    # Probability matrix
    print("Calculating probability matrix...")
    prob_matrix = cal_probability(all_word_embed, sensitive_word_embed, epsilon=epsilon)

    # Tokenize example sentence
    doc = [token.text for token in tokenizer(example_sentence)]
    print(f"Original: {example_sentence}")
    print(f"Tokenized: {doc}")

    # Map tokens to indices (skip OOV for SanText)
    doc_indices = [word2id[token] for token in doc if token in word2id]

    # Initialize global prob_matrix for SanText
    SanText_init(prob_matrix)

    # SanText
    start_time = time.time()
    sanitized_indices = SanText(doc_indices)
    stext_time = time.time() - start_time
    sanitized_words = [all_words[idx] for idx in sanitized_indices]
    print("SanText output:", " ".join(sanitized_words))

    # Initialize global variables for SanText_plus
    SanText_plus_init(prob_matrix, word2id, sword2id, all_words, p, tokenizer)

    # SanText+
    start_time = time.time()
    sanitized_plus = SanText_plus(doc)
    stext_plus_time = time.time() - start_time
    print("SanText+ output:", sanitized_plus)

    # Print performance comparison table
    table = [
        ["SanText", f"{stext_time:.6f}"],
        ["SanText+", f"{stext_plus_time:.6f}"]
    ]
    headers = ["Method", "Total Time (s)"]
    print("\nPerformance Comparison Table:")
    print(tabulate(table, headers=headers, tablefmt="grid")) 