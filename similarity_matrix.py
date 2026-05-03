import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


def build_similarity_matrix(features, method='cosine'):
    if method == 'cosine':
        return cosine_similarity(features)
    elif method == 'pearson':
        return np.corrcoef(features)
    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_similarity_matrix(matrix, save_path, title="Similarity Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix[:100, :100], cmap='RdYlBu_r', center=0, square=True)
    plt.title(f"{title} (first 100 samples)")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_similarity_matrix(matrix, save_path):
    np.save(save_path, matrix)
