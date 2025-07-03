import os
import sys
import json
import numpy as np
from clusant import CluSanT
from embedding_handler import EmbeddingHandler
import time
from tabulate import tabulate

class CluSanTBatchProcessor:
    def __init__(self,
                 embedding_file='all-mpnet-base-v2',
                 embeddings_path=None,
                 loc_clusters_path='clusters/gpt-4/LOC.json',
                 org_clusters_path='clusters/gpt-4/ORG.json',
                 epsilon=1.0,
                 K=1,
                 dp_type='metric'):
        if embeddings_path is None:
            embeddings_path = f'embeddings/{embedding_file}.txt'
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file '{embeddings_path}' not found. Please generate it first.")
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
        self.clusant = CluSanT(
            embedding_file=embedding_file,
            embeddings=embeddings,
            epsilon=epsilon,
            num_clusters=len(clusters),
            K=K,
            dp_type=dp_type,
        )
        # Patch clusters in the instance
        del self.clusant.clusters
        self.clusant.clusters = clusters

    def sanitize(self, sentence, sensitive_words):
        anonymized_text = sentence
        for word in sensitive_words:
            replacement = self.clusant.replace_word(word)
            anonymized_text = anonymized_text.replace(word.capitalize(), replacement.capitalize() if replacement else 'XXXXX')
            anonymized_text = anonymized_text.replace(word, replacement if replacement else 'XXXXX')
        return anonymized_text

def _sanitize_helper(args):
    # Helper for multiprocessing: re-initialize processor in each process
    sentence, sensitive_words, processor_args = args
    processor = CluSanTBatchProcessor(**processor_args)
    return processor.sanitize(sentence, sensitive_words)

if __name__ == "__main__":
    example_sentences = [
        ("I visited Paris and met with CompanyX in Berlin.", ['paris', 'companyx', 'berlin']),
        ("London is a city where OrgY has an office.", ['london', 'orgy']),
        ("The headquarters of OrgZ are in Rome.", ['orgz', 'rome'])
    ]
    processor_args = {}
    processor = CluSanTBatchProcessor(**processor_args)

    # Serial processing
    print("Serial Processing: (for small datasets)")
    start = time.time()
    results = []
    for sent, sensitive_words in example_sentences:
        anonymized = processor.sanitize(sent, sensitive_words)
        results.append((sent, anonymized))
    elapsed = time.time() - start
    for orig, anon in results:
        print("Original text:", orig)
        print("Anonymized text:", anon)
        print()
    performance_table = [["CluSanT (serial)", f"{elapsed:.6f}"]]
    print("Performance Table:")
    print(tabulate(performance_table, headers=["Method", "Total Time (s)"], tablefmt="grid"))

    # Parallel processing with multiprocessing.Pool
    print("\nParallel Processing (multiprocessing.Pool): (for big datasets only)")
    import multiprocessing
    start = time.time()
    # Prepare arguments for each process
    pool_args = [(sent, sensitive_words, processor_args) for sent, sensitive_words in example_sentences]
    with multiprocessing.Pool() as pool:
        parallel_results = pool.map(_sanitize_helper, pool_args)
    elapsed_parallel = time.time() - start
    for (orig, _), anon in zip(example_sentences, parallel_results):
        print("Original text:", orig)
        print("Anonymized text:", anon)
        print()
    performance_table = [["CluSanT (parallel)", f"{elapsed_parallel:.6f}"]]
    print("Performance Table:")
    print(tabulate(performance_table, headers=["Method", "Total Time (s)"], tablefmt="grid")) 