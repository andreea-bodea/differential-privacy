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

class SanTextBatchProcessor:
    def __init__(self,
                 glove_path="data/glove.840B.300d.txt",
                 filtered_glove_path="data/glove.filtered.txt",
                 data_dir="data/SST-2/",
                 epsilon=15.0,
                 p=0.2,
                 sensitive_word_percentage=0.5,
                 vocab_cache_path="vocab_cache.pkl",
                 glove_cache_path="glove_filtered_cache.pkl"):
        self.epsilon = epsilon
        self.p = p
        self.sensitive_word_percentage = sensitive_word_percentage
        self.tokenizer = English()
        self.vocab, self.words, self.sensitive_words, self.sensitive_words2id = get_vocab_SST2_cached(
            data_dir, self.tokenizer, tokenizer_type="word", cache_path=vocab_cache_path)
        if os.path.exists(filtered_glove_path):
            used_glove_path = filtered_glove_path
        else:
            used_glove_path = glove_path
        if os.path.exists(glove_cache_path):
            with open(glove_cache_path, "rb") as f:
                self.word2id, self.sword2id, self.id2sword, self.all_word_embed, self.sensitive_word_embed, self.all_words = pickle.load(f)
        else:
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
                    if cur_word in self.vocab and cur_word not in word2id:
                        word2id[cur_word] = all_count
                        all_words.append(cur_word)
                        all_count += 1
                        emb = [float(i) for i in content[1:]]
                        all_word_embed.append(emb)
                        if cur_word in self.sensitive_words2id:
                            sword2id[cur_word] = sensitive_count
                            id2sword[sensitive_count] = cur_word
                            sensitive_count += 1
                            sensitive_word_embed.append(emb)
            all_word_embed = np.array(all_word_embed, dtype='f')
            sensitive_word_embed = np.array(sensitive_word_embed, dtype='f')
            with open(glove_cache_path, "wb") as f:
                pickle.dump((word2id, sword2id, id2sword, all_word_embed, sensitive_word_embed, all_words), f)
            self.word2id = word2id
            self.sword2id = sword2id
            self.id2sword = id2sword
            self.all_word_embed = all_word_embed
            self.sensitive_word_embed = sensitive_word_embed
            self.all_words = all_words
        self.prob_matrix = cal_probability(self.all_word_embed, self.sensitive_word_embed, epsilon=self.epsilon)

    def sanitize(self, sentence, method="SanText", epsilons=None):
        doc = [token.text for token in self.tokenizer(sentence)]
        if method == "SanText":
            doc_indices = [self.word2id[token] for token in doc if token in self.word2id]
            # Per-word epsilon support
            if epsilons is not None:
                sanitized_words = []
                for idx, word_idx in enumerate(doc_indices):
                    epsilon = epsilons[idx] if idx < len(epsilons) else self.epsilon
                    prob_matrix = cal_probability(self.all_word_embed, self.sensitive_word_embed, epsilon=epsilon)
                    SanText_init(prob_matrix)
                    sanitized_index = SanText([word_idx])
                    sanitized_words.append(self.all_words[sanitized_index[0]])
                sanitized_sentence = " ".join(sanitized_words)
                return sanitized_sentence
            else:
                SanText_init(self.prob_matrix)
                sanitized_indices = SanText(doc_indices)
                sanitized_words = [self.all_words[idx] for idx in sanitized_indices]
                sanitized_sentence = " ".join(sanitized_words)
                return sanitized_sentence
        elif method == "SanText+":
            SanText_plus_init(self.prob_matrix, self.word2id, self.sword2id, self.all_words, self.p, self.tokenizer)
            sanitized_plus = SanText_plus(doc)
            return sanitized_plus
        else:
            raise ValueError("method must be either 'SanText' or 'SanText+'")

# --- Main script ---
if __name__ == "__main__":
    example_sentences = [
        "The movie was absolutely wonderful and inspiring.",
        #"I did not enjoy the film at all.",
        "The plot was predictable but the acting was great."
    ]
    processor = SanTextBatchProcessor()
    # Example: per-word epsilon for the first sentence
    per_word_epsilons = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # Should match number of tokens in the sentence
    print("\nSanText with per-word epsilon (first sentence) - epsilon: ", per_word_epsilons)
    sanitized = processor.sanitize(example_sentences[0], method="SanText", epsilons=per_word_epsilons)
    print(f"Original: {example_sentences[0]}")
    print(f"Sanitized: {sanitized}\n")
    # Standard batch processing for all sentences
    for method in ["SanText", "SanText+"]:
        print(f"\nMethod: {method} (Serial Processing)")
        start_time = time.time()
        if method == "SanText":
            # Example: per-word epsilons for each sentence
            epsilons_list = [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # for first sentence
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],         # for second sentence
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]     # for third sentence
            ]
            results = [processor.sanitize(sent, method=method, epsilons=eps) for sent, eps in zip(example_sentences, epsilons_list)]
        else:
            results = [processor.sanitize(sent, method=method) for sent in example_sentences]
        elapsed = time.time() - start_time
        for sent, sanitized in zip(example_sentences, results):
            print(f"Original: {sent}")
            print(f"Sanitized: {sanitized}\n")
        print(f"Total time for {len(example_sentences)} sentences (serial): {elapsed:.6f} seconds\n")

    # Parallel processing example with per-word epsilons for SanText
    import concurrent.futures
    for method in ["SanText", "SanText+"]:
        print(f"\nMethod: {method} (Parallel Processing)")
        start_time = time.time()
        if method == "SanText":
            epsilons_list = [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
            ]
            def sanitize_with_eps(args):
                sentence, epsilons = args
                return processor.sanitize(sentence, method="SanText", epsilons=epsilons)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(sanitize_with_eps, zip(example_sentences, epsilons_list)))
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda sent: processor.sanitize(sent, method=method), example_sentences))
        elapsed = time.time() - start_time
        for sent, sanitized in zip(example_sentences, results):
            print(f"Original: {sent}")
            print(f"Sanitized: {sanitized}\n")
        print(f"Total time for {len(example_sentences)} sentences (parallel): {elapsed:.6f} seconds\n") 