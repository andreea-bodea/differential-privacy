import pickle

glove_path = "data/glove.840B.300d.txt"
filtered_glove_path = "data/glove.filtered.txt"
vocab_cache_path = "vocab_cache.pkl"

# Load vocab set from cache
with open(vocab_cache_path, "rb") as f:
    vocab, words, sensitive_words, sensitive_words2id = pickle.load(f)
vocab_set = set(vocab.keys())

with open(glove_path) as fin, open(filtered_glove_path, "w") as fout:
    for line in fin:
        word = line.split(" ", 1)[0]
        if word in vocab_set:
            fout.write(line)
print(f"Filtered GloVe written to {filtered_glove_path} with {len(vocab_set)} words.") 