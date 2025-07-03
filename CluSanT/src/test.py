import json
import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from embedding_handler import EmbeddingHandler
from clusant import CluSanT

INPUT_FILEPATH = "echr_dev.json"
# Load GPT-2 Model and Tokenizer
perplexity_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large")
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-large")
perplexity_model.eval()

# Move model to GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
perplexity_model.to(DEVICE)

similarity_model = SentenceTransformer("all-mpnet-base-v2")


# Load JSON data from files
def load_json_file(filepath):
    with open(filepath, "r") as file:
        return json.load(file)


def read_existing_parameters(results_file):
    existing_parameters = set()
    try:
        with open(results_file, "r") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header line
            for row in reader:
                if row:
                    parameters = tuple(row[:5])
                    existing_parameters.add(parameters)
    except FileNotFoundError:
        # If the file doesn't exist, it's the first run, so there are no existing parameters.
        pass
    return existing_parameters


def filter_new_parameters(all_parameters, existing_parameters):
    # Convert existing parameters set of tuples to a set of strings for easy comparison
    existing_parameters_strings = {
        "{}, {}, {}, {}, {}".format(*params) for params in existing_parameters
    }

    # Filter out parameters that have not been processed yet by comparing string versions
    new_parameters = [
        params
        for params in all_parameters
        if "{}, {}, {}, {}, {}".format(*params) not in existing_parameters_strings
    ]
    return new_parameters


def calculate_perplexity(comment):
    try:
        max_length = 1024  # Max token count for the model
        stride = 512
        nlls = []
        for i in range(0, len(comment), stride):
            chunk = comment[i : i + max_length]
            encodings = tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=max_length
            )
            # Move encodings to the same device as model
            encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
            with torch.no_grad():
                outputs = perplexity_model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            nlls.append(loss.item() * encodings["input_ids"].size(1))
        total_loss = sum(nlls)
        average_loss = total_loss / sum(
            min(len(comment[i : i + max_length]), max_length)
            for i in range(0, len(comment), stride)
        )
        perplexity = torch.exp(torch.tensor(average_loss))
        return perplexity.item()
    except Exception as e:
        print(f"Error calculating perplexity for comment: {e}")
        return np.nan


# Function to calculate cosine similarity
def calculate_similarity(text1, text2):
    embeddings1 = similarity_model.encode(text1)
    embeddings2 = similarity_model.encode(text2)
    return 1 - cosine(embeddings1, embeddings2)


def collect_unique_mentions(annotations, only_loc_and_org):
    # Collects all unique mentions from all annotators' entity mentions.
    unique_mentions = set()
    for _, annotator_value in annotations.items():
        for mention in annotator_value["entity_mentions"]:
            # Add only the unique 'span_text' to the set
            if only_loc_and_org:
                if mention["entity_type"] == "ORG" or mention["entity_type"] == "LOC":
                    unique_mentions.add(mention["span_text"])
            else:
                unique_mentions.add(mention["span_text"])

    return unique_mentions


# Uncomment the following line if you also want to replace entities outside "LOCS" and "ORGS"
# LOCS_AND_ORGS = [True, False]
LOCS_AND_ORGS = [True]

EPSILONS = [0.1, 0.5, 1, 2, 4, 8, 16]
NUM_CLUSTERS = [1, 40, 180, 360, 720]
KS = [1, 8, 16, 32, 64, 128]

# Uncomment the following line to use both "metric" and "standard" differential privacy types.
# In metric DP, sensitivity = 1; in standard DP, sensitivity = maximum possible Euclidean distance between words/clusters.
# DP_TYPES = ["metric", "standard"]
DP_TYPES = ["metric"]


output_filepath = "results.csv"
# Ensure the main results file has headers
if not os.path.exists(output_filepath):
    with open(output_filepath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "loc_and_org",
                "epsilon",
                "num_cluster",
                "K",
                "dp_type",
                "average_similarity",
                "average_perplexity",
            ]
        )

# Filepath for anonymized texts
anonymized_text_filepath = "anonymized_texts.csv"
# Ensure the anonymized text file has headers
if not os.path.exists(anonymized_text_filepath):
    with open(anonymized_text_filepath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "loc_and_org",
                "epsilon",
                "num_cluster",
                "K",
                "dp_type",
                "anonymized_text",
            ]
        )

embedding_handler = EmbeddingHandler()
embeddings = embedding_handler.load_embeddings("embeddings/all-mpnet-base-v2.txt")

# Load existing parameters to avoid recalculating
existing_parameters = read_existing_parameters("results.csv")

parameters = [
    (only_loc_and_org, epsilon, num_cluster, k, dp_type)
    for only_loc_and_org in LOCS_AND_ORGS
    for epsilon in EPSILONS
    for num_cluster in NUM_CLUSTERS
    for k in ([1] if num_cluster == 1 else KS)  # Adjust 'k' based on 'num_cluster'
    for dp_type in DP_TYPES
]

# Filter out the parameters that have already been processed
new_parameters = filter_new_parameters(parameters, existing_parameters)

with open(INPUT_FILEPATH, "r") as file:
    data = json.load(file)

for param in tqdm(new_parameters, desc="Processing Parameters:"):
    clusant = CluSanT(
        embedding_file="all-mpnet-base-v2",
        epsilon=param[1],
        num_clusters=param[2],
        K=param[3],
        dp_type=param[4],
        embeddings=embeddings,
    )

    similarities = []
    perplexities = []
    for annotation in data:
        unique_mentions = collect_unique_mentions(annotation["annotations"], param[0])
        original_text = annotation["text"]
        anonymized_text = original_text

        for word in sorted(unique_mentions, key=len, reverse=True):
            new_word = clusant.replace_word(word)
            anonymized_text = anonymized_text.replace(
                word, new_word if new_word else "XXXXX"
            )

        similarities.append(calculate_similarity(original_text, anonymized_text))
        perplexities.append(calculate_perplexity(anonymized_text))

        # Save each anonymized text with its parameters
        with open(anonymized_text_filepath, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    param[0],
                    param[1],
                    param[2],
                    param[3],
                    param[4],
                    anonymized_text,
                ]
            )

    # Append results for this configuration
    with open(output_filepath, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                param[0],
                param[1],
                param[2],
                param[3],
                param[4],
                np.mean(similarities),
                np.mean(perplexities),
            ]
        )
