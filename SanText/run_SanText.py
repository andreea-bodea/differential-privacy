import argparse
import torch
import random
import numpy as np
import logging
import os
import pickle
from tqdm import tqdm
from scipy.special import softmax
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from utils import get_vocab_SST2, get_vocab_CliniSTS, get_vocab_QNLI, word_normalize
from spacy.lang.en import English
from transformers import BertTokenizer, BertForMaskedLM
from SanText import SanText_plus, SanText_plus_init, SanText, SanText_init, cal_probability
from collections import Counter
import ray

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def cal_probability(word_embed_1, word_embed_2, epsilon=2.0):
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix


def get_vocab_SST2_cached(data_dir, tokenizer, tokenizer_type="word", cache_path="vocab_cache.pkl", sensitive_word_percentage=0.5):
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
        sensitive_word_count = int(sensitive_word_percentage * len(vocab))
        sensitive_words = words[-sensitive_word_count - 1:]
        sensitive_words2id = {word: k for k, word in enumerate(sensitive_words)}
        with open(cache_path, "wb") as f:
            pickle.dump((vocab, words, sensitive_words, sensitive_words2id), f)
        return vocab, words, sensitive_words, sensitive_words2id


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="./data/SST-2/",
        type=str,
        help="The input dir"
    )

    parser.add_argument(
        "--bert_model_path",
        default="bert-base-uncased",
        type=str,
        help="bert model name or path. leave it bank if you are using Glove"
    )

    parser.add_argument(
        "--output_dir",
        default="./output_SanText/QNLI/",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--word_embedding_path",
        default='./data/glove.840B.300d.txt',
        type=str,
        help="The pretrained word embedding path. leave it blank if you are using BERT",
    )

    parser.add_argument(
        "--word_embedding_size",
        default=300,
        type=int,
        help="The pretrained word embedding size. leave it blank if you are using BERT",
    )

    parser.add_argument(
        '--method',
        choices=['SanText', 'SanText_plus'],
        default='SanText_plus',
        help='Sanitized method'
    )

    parser.add_argument(
        '--embedding_type',
        choices=['glove', 'bert'],
        default='glove',
        help='embedding used for sanitization'
    )

    parser.add_argument('--task',
                        choices=['CliniSTS', "SST-2", "QNLI"],
                        default='SST-2',
                        help='NLP eval tasks')

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--epsilon", type=float, default=15, help="privacy parameter epsilon")
    parser.add_argument("--p", type=float, default=0.2, help="SanText+: probability of non-sensitive words to be sanitized")

    parser.add_argument("--sensitive_word_percentage", type=float, default=0.5,
                        help="SanText+: how many words are treated as sensitive")

    parser.add_argument("--threads", type=int, default=12, help="number of processors")

    parser.add_argument("--use_ray", action="store_true", help="Use Ray for parallel sanitization")

    args = parser.parse_args()

    set_seed(args)

    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("Running method: %s, task: %s,  epsilon = %s, random_seed: %d" % (
    args.method, args.task, args.epsilon, args.seed))

    if args.method == "SanText":
        args.sensitive_word_percentage = 1.0
        args.output_dir = os.path.join(args.output_dir, "eps_%.2f" % args.epsilon)
    else:
        args.output_dir = os.path.join(args.output_dir, "eps_%.2f" % args.epsilon, "sword_%.2f_p_%.2f"%(args.sensitive_word_percentage,args.p))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Building Vocabulary...")

    vocab_cache_path = "vocab_cache.pkl"
    filtered_glove_path = "data/glove.filtered.txt"
    if args.embedding_type=="glove":
        tokenizer = English()
        tokenizer_type="word"
        # Use cached vocab
        vocab, words, sensitive_words, sensitive_words2id = get_vocab_SST2_cached(
            args.data_dir, tokenizer, tokenizer_type=tokenizer_type, cache_path=vocab_cache_path, sensitive_word_percentage=args.sensitive_word_percentage)
        vocab_set = set(vocab.keys())
    else:
        tokenizer  = BertTokenizer.from_pretrained(args.bert_model_path)
        tokenizer_type = "subword"
        if args.task == "SST-2":
            vocab = get_vocab_SST2(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
        elif args.task == "CliniSTS":
            vocab = get_vocab_CliniSTS(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
        elif args.task == "QNLI":
            vocab = get_vocab_QNLI(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
        else:
            raise NotImplementedError
        words = [key for key, _ in vocab.most_common()]
        sensitive_word_count = int(args.sensitive_word_percentage * len(vocab))
        sensitive_words = words[-sensitive_word_count - 1:]
        sensitive_words2id = {word: k for k, word in enumerate(sensitive_words)}
        vocab_set = set(vocab.keys())
    logger.info("#Total Words: %d, #Sensitive Words: %d" % (len(words),len(sensitive_words2id)))

    sensitive_word_embed = []
    all_word_embed=[]

    word2id = {}
    sword2id = {}
    sensitive_count = 0
    all_count = 0
    # Use filtered GloVe if available
    if args.embedding_type == "glove":
        if os.path.exists(filtered_glove_path):
            used_glove_path = filtered_glove_path
            logger.info(f"Using filtered GloVe file: {filtered_glove_path}")
        else:
            used_glove_path = args.word_embedding_path
            logger.info(f"Using full GloVe file: {args.word_embedding_path}")
        num_lines = sum(1 for _ in open(used_glove_path))
        logger.info("Loading Word Embedding File: %s" % used_glove_path)
        with open(used_glove_path) as f:
            for row in tqdm(f, total=num_lines):
                content = row.rstrip().split(' ')
                cur_word=word_normalize(content[0])
                if cur_word in vocab_set and cur_word not in word2id:
                    word2id[cur_word] = all_count
                    all_count += 1
                    emb=[float(i) for i in content[1:]]
                    all_word_embed.append(emb)
                    if cur_word in sensitive_words2id:
                        sword2id[cur_word] = sensitive_count
                        sensitive_count += 1
                        sensitive_word_embed.append(emb)
                assert len(word2id)==len(all_word_embed)
                assert len(sword2id) == len(sensitive_word_embed)
            f.close()
    else:
        logger.info("Loading BERT Embedding File: %s" % args.bert_model_path)
        model=BertForMaskedLM.from_pretrained(args.bert_model_path)
        embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

        for cur_word in tokenizer.vocab:
            if cur_word in vocab and cur_word not in word2id:
                word2id[cur_word] = all_count
                emb = embedding_matrix[tokenizer.convert_tokens_to_ids(cur_word)]
                all_word_embed.append(emb)
                all_count += 1

                if cur_word in sensitive_words2id:
                    sword2id[cur_word] = sensitive_count
                    sensitive_count += 1
                    sensitive_word_embed.append(emb)
            assert len(word2id) == len(all_word_embed)
            assert len(sword2id) == len(sensitive_word_embed)

    all_word_embed=np.array(all_word_embed, dtype='f')
    sensitive_word_embed = np.array(sensitive_word_embed, dtype='f')

    logger.info("All Word Embedding Matrix: %s" % str(all_word_embed.shape))
    logger.info("Sensitive Word Embedding Matrix: %s" % str(sensitive_word_embed.shape))

    logger.info("Calculating Prob Matrix for Exponential Mechanism...")
    prob_matrix = cal_probability(all_word_embed,sensitive_word_embed, args.epsilon)

    threads = min(args.threads, cpu_count())

    # Ray initialization if needed
    if args.use_ray:
        ray.init(ignore_reinit_error=True)

    for file_name in ['train.tsv','dev.tsv']:
        data_file = os.path.join(args.data_dir, file_name)
        out_file = open(os.path.join(args.output_dir, file_name), 'w')
        logger.info("Processing file: %s. Will write to: %s" % (data_file,os.path.join(args.output_dir, file_name)))

        num_lines = sum(1 for _ in open(data_file))
        with open(data_file, 'r') as rf:
            # header
            header = next(rf)
            out_file.write(header)
            labels = []
            docs = []
            if args.task == "SST-2":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text = content[0]
                    label = int(content[1])
                    if args.embedding_type == "glove":
                        doc = [token.text for token in tokenizer(text)]
                    else:
                        doc = tokenizer.tokenize(text)
                    docs.append(doc)
                    labels.append(label)
            elif args.task == "CliniSTS":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text1 = content[7]
                    text2 = content[8]
                    label = content[-1]
                    if args.embedding_type == "glove":
                        doc1 = [token.text for token in tokenizer(text1)]
                        doc2 = [token.text for token in tokenizer(text2)]
                    else:
                        doc1 = tokenizer.tokenize(text1)
                        doc2 = tokenizer.tokenize(text2)
                    docs.append(doc1)
                    docs.append(doc2)
                    labels.append(label)
            elif args.task == "QNLI":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text1 = content[1]
                    text2 = content[2]
                    label = content[-1]
                    if args.embedding_type == "glove":
                        doc1 = [token.text for token in tokenizer(text1)]
                        doc2 = [token.text for token in tokenizer(text2)]
                    else:
                        doc1 = tokenizer.tokenize(text1)
                        doc2 = tokenizer.tokenize(text2)

                    docs.append(doc1)
                    docs.append(doc2)
                    labels.append(label)

            rf.close()

        # --- Ray-based parallelization ---
        if args.use_ray:
            if args.method == "SanText_plus":
                id2sword = {v: k for k, v in sword2id.items()}
                @ray.remote
                def sanitize_plus(doc, prob_matrix, word2id, sword2id, id2sword, all_words, p):
                    # Inline logic from SanText_plus
                    new_doc = []
                    for word in doc:
                        if word in word2id:
                            if word in sword2id:
                                index = word2id[word]
                                sampling_prob = prob_matrix[index]
                                sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                                new_doc.append(id2sword[sampling_index[0]])
                            else:
                                flip_p = np.random.random()
                                if flip_p <= p:
                                    index = word2id[word]
                                    sampling_prob = prob_matrix[index]
                                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                                    new_doc.append(id2sword[sampling_index[0]])
                                else:
                                    new_doc.append(word)
                        else:
                            sampling_prob = 1 / len(all_words) * np.ones(len(all_words), )
                            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                            new_doc.append(all_words[sampling_index[0]])
                    return " ".join(new_doc)
                futures = [
                    sanitize_plus.remote(doc, prob_matrix, word2id, sword2id, id2sword, words, args.p)
                    for doc in docs
                ]
                results = ray.get(futures)
            else:
                @ray.remote
                def sanitize(doc, prob_matrix):
                    # Inline logic from SanText
                    new_doc = []
                    for token in doc:
                        sampling_prob = prob_matrix[token]
                        sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                        new_doc.append(sampling_index[0])
                    return new_doc
                # Map tokens to indices for SanText
                doc_indices = [[word2id[token] for token in doc if token in word2id] for doc in docs]
                futures = [sanitize.remote(doc_idx, prob_matrix) for doc_idx in doc_indices]
                sanitized_indices_list = ray.get(futures)
                # Convert indices back to words
                results = [" ".join([words[idx] for idx in sanitized_indices]) for sanitized_indices in sanitized_indices_list]
        else:
            # --- Original multiprocessing code ---
            if args.method == "SanText_plus":
                with Pool(threads, initializer=SanText_plus_init, initargs=(prob_matrix, word2id, sword2id, words, args.p, tokenizer)) as p:
                    annotate_ = partial(
                        SanText_plus,
                    )
                    results = list(
                        tqdm(
                            p.imap(annotate_, docs, chunksize=32),
                            total=len(docs),
                            desc="Sanitize docs using SanText",
                        )
                    )
                    p.close()
            else:
                with Pool(threads, initializer=SanText_init, initargs=(prob_matrix,)) as p:
                    annotate_ = partial(
                        SanText,
                    )
                    results = list(
                        tqdm(
                            p.imap(annotate_, docs, chunksize=32),
                            total=len(docs),
                            desc="Sanitize docs using SanText",
                        )
                    )
                    p.close()

        logger.info("Saving ...")

        if args.task == "SST-2":
            for i, predicted_text in enumerate(results):
                write_content = predicted_text + "\t" + str(labels[i]) + "\n"
                out_file.write(write_content)
        elif args.task == "CliniSTS":
            assert len(results) / 2 == len(labels)
            for i in range(len(labels)):
                predicted_text1 = results[i*2]
                predicted_text2 = results[i*2+1]
                write_content = str(i) + "\t" + "none\t" * 6 + predicted_text1 + "\t" + predicted_text2 + "\t" + str(
                    labels[i]) + "\n"
                out_file.write(write_content)
        elif args.task == "QNLI":
            assert len(results) / 2 == len(labels)
            for i in range(len(labels)):
                predicted_text1 = results[i*2]
                predicted_text2 = results[i*2+1]
                write_content = str(i) + "\t" + predicted_text1 + "\t" + predicted_text2 + "\t" + str(
                    labels[i]) + "\n"
                out_file.write(write_content)

        out_file.close()



if __name__ == "__main__":
    main()
