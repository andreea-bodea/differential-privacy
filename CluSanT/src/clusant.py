import os
import json
import numpy as np
from random import choice
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist


class CluSanT:
    def __init__(
        self,
        embedding_file,
        embeddings,
        epsilon=1,
        num_clusters=1,
        mechanism="clusant",
        metric_to_create_cluster="euclidean",
        distance_metric_for_cluster="euclidean",
        distance_metric_for_words="euclidean",
        dp_type="metric",
        K=1,
    ):
        self.embedding_file = embedding_file
        self.epsilon = epsilon
        self.num_clusters = num_clusters
        self.mechanism = mechanism
        self.metric_to_create_cluster = metric_to_create_cluster
        self.distance_metric_for_cluster = distance_metric_for_cluster
        self.distance_metric_for_words = distance_metric_for_words
        self.dp_type = dp_type
        self.K = K
        self.cluster_path = (
            f"clusters/{embedding_file}_{num_clusters}_{metric_to_create_cluster}.json"
        )
        self.centroids_path = f"centroids/{embedding_file}_{num_clusters}_{K}_{metric_to_create_cluster}.json"
        self.intra_cluster_path = f"intra/{embedding_file}_{num_clusters}_{metric_to_create_cluster}_{dp_type}.json"
        self.inter_cluster_path = f"inter/{embedding_file}_{num_clusters}_{K}_{metric_to_create_cluster}_{dp_type}.json"
        self.embeddings = embeddings
        self.clusters = self.load_clusters()
        self.inter_distances, self.inter_cluster_sensitivity = (
            self.calculate_inter_cluster_distances()
        )
        self.intra_cluster_sensitivity = self.calculate_intra_cluster_distances()

    def create_clusters(self):
        clusters = {}
        words = list(self.embeddings.keys())
        words_per_cluster = len(words) // min(self.num_clusters, len(words))
        seen_words = set()

        pbar = tqdm(total=len(words), desc="Creating clusters")
        while words:
            w = choice(words)
            all_embeddings = [self.embeddings[word] for word in words]
            distances = cdist(
                [self.embeddings[w]],
                all_embeddings,
                metric=self.metric_to_create_cluster,
            )[0]
            nearest_indices = np.argsort(distances)[:words_per_cluster]

            cluster = [words[i] for i in nearest_indices]
            clusters[len(clusters)] = cluster

            for word in cluster:
                if word in seen_words:
                    pbar.close()
                    raise ValueError(f"Word '{word}' appears in multiple clusters.")
                seen_words.add(word)

            words = [word for i, word in enumerate(words) if i not in nearest_indices]
            pbar.update(len(cluster))

        pbar.close()

        with open(self.cluster_path, "w") as f:
            json.dump(clusters, f)

        return clusters

    def load_clusters(self):
        if os.path.exists(self.cluster_path):
            with open(self.cluster_path, "r") as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}

        print("Creating new clusters")
        return self.create_clusters()

    def find_word_cluster(self, word):
        for label, words in self.clusters.items():
            if word in words:
                return label
        return None

    def exponential_mechanism(self, utilities, sensitivity):
        probabilities = np.exp(self.epsilon * np.array(utilities) / (2 * sensitivity))
        probabilities /= np.sum(probabilities)
        return probabilities

    def calculate_inter_cluster_distances(self):
        centroids = {}

        if os.path.exists(self.centroids_path):
            with open(self.centroids_path, "r") as f:
                data = json.load(f)
            centroids = {int(label): np.array(vector) for label, vector in data.items()}
        else:
            # Calculate centroids of each cluster for inter-cluster distances
            for label, words in self.clusters.items():
                cluster_vectors = [self.embeddings[word] for word in words]
                centroids[label] = (self.K * np.mean(cluster_vectors, axis=0)).tolist()

            with open(self.centroids_path, "w") as f:
                json.dump(centroids, f)

        # Check if distances are already computed
        if os.path.exists(self.inter_cluster_path):
            with open(self.inter_cluster_path, "r") as f:
                inter_cluster_distances = np.array(json.load(f))
        else:
            # Compute pairwise distances between centroids
            centroid_vectors = [centroids[label] for label in centroids]
            inter_cluster_distances = cdist(
                centroid_vectors,
                centroid_vectors,
                metric=self.distance_metric_for_cluster,
            )

            # Save the computed distances
            with open(self.inter_cluster_path, "w") as f:
                json.dump(inter_cluster_distances.tolist(), f)

        inter_cluster_sensitivity = (
            np.max(inter_cluster_distances) if self.dp_type == "standard" else 1
        )

        return (
            inter_cluster_distances,
            inter_cluster_sensitivity,
        )

    def calculate_intra_cluster_distances(self):
        if os.path.exists(self.intra_cluster_path):
            with open(self.intra_cluster_path, "r") as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}

        # Calculate max intra-cluster distances for each cluster
        intra_cluster_sensitivity = {}
        for label, words in tqdm(
            self.clusters.items(), desc="Calculating intra-cluster distances"
        ):
            cluster_embeddings = [self.embeddings[word] for word in words]
            if len(cluster_embeddings) > 1:
                distances = pdist(
                    cluster_embeddings, metric=self.distance_metric_for_words
                )
                max_distance = max(distances)
            else:
                max_distance = 0
            intra_cluster_sensitivity[label] = max_distance

        with open(self.intra_cluster_path, "w") as f:
            json.dump(intra_cluster_sensitivity, f)

        return intra_cluster_sensitivity

    def replace_word(self, target_word):
        target_word = target_word.lower()
        target_cluster_label = self.find_word_cluster(target_word)

        if target_cluster_label is None:
            return None

        if self.num_clusters == 1 or self.mechanism != "clusant":
            selected_cluster_label = target_cluster_label
        else:
            distances_from_cluster = [
                -self.inter_distances[target_cluster_label][i]
                for i in range(len(self.clusters))
            ]
            probabilities = self.exponential_mechanism(
                distances_from_cluster, self.inter_cluster_sensitivity
            )
            selected_cluster_label = np.random.choice(
                list(self.clusters.keys()), p=probabilities
            )

        # Get embeddings for the target and selected cluster's words
        selected_cluster_words = self.clusters[selected_cluster_label]
        target_word_embedding = np.array(self.embeddings[target_word])
        selected_cluster_word_embeddings = [
            self.embeddings[word] for word in selected_cluster_words
        ]

        # Reshape the target word embedding to be 2-dimensional (1, number_of_features)
        target_word_embedding = target_word_embedding.reshape(1, -1)
        word_embeddings_array = np.array(selected_cluster_word_embeddings)

        # Compute distances using cdist in one go
        distances_from_word = cdist(
            target_word_embedding,
            word_embeddings_array,
            metric=self.distance_metric_for_words,
        ).flatten()

        # Apply the exponential mechanism using the (possibly normalized) distances
        probabilities = self.exponential_mechanism(
            -np.array(distances_from_word),
            self.intra_cluster_sensitivity[selected_cluster_label],
        )

        # Select a new word from the selected cluster based on calculated probabilities
        selected_word = np.random.choice(selected_cluster_words, p=probabilities)

        return selected_word
