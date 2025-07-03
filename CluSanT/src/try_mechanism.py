import numpy as np
from clusant import CluSanT

# Minimal in-memory embeddings for demo
embeddings = {
    'paris': np.array([1.0, 0.0]),
    'london': np.array([0.9, 0.1]),
    'berlin': np.array([0.8, 0.2]),
    'companyx': np.array([0.0, 1.0]),
    'companyy': np.array([0.1, 0.9]),
    'companyz': np.array([0.2, 0.8]),
}

# Minimal clusters: 2 clusters, one for locations, one for organizations
clusters = {
    0: ['paris', 'london', 'berlin'],
    1: ['companyx', 'companyy', 'companyz']
}

# Patch CluSanT to use our in-memory clusters and avoid file I/O
def dummy_inter_cluster_distances(self):
    # 2 clusters, so 2x2 distance matrix
    inter_cluster_distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    inter_cluster_sensitivity = 1.0
    return inter_cluster_distances, inter_cluster_sensitivity

def dummy_intra_cluster_distances(self):
    # Each cluster has a max intra distance of 1.0 for demo
    return {0: 1.0, 1: 1.0}

class DemoCluSanT(CluSanT):
    def load_clusters(self):
        return clusters
    def calculate_inter_cluster_distances(self):
        return dummy_inter_cluster_distances(self)
    def calculate_intra_cluster_distances(self):
        return dummy_intra_cluster_distances(self)

# Instantiate with minimal parameters
demo = DemoCluSanT(
    embedding_file='demo',
    embeddings=embeddings,
    epsilon=1.0,
    num_clusters=2,
    K=1,
    dp_type='metric',
)

# Sample text and sensitive word
test_text = "I visited Paris and met with CompanyX."
sensitive_words = ['paris', 'companyx']

anonymized_text = test_text
for word in sensitive_words:
    replacement = demo.replace_word(word)
    anonymized_text = anonymized_text.replace(word.capitalize(), replacement.capitalize() if replacement else 'XXXXX')
    anonymized_text = anonymized_text.replace(word, replacement if replacement else 'XXXXX')

print("Original text:", test_text)
print("Anonymized text:", anonymized_text) 