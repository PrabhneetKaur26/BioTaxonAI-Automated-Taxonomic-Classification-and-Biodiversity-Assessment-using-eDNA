"""2-D embeddings for visualisation (PCA and UMAP).

These coordinates are for PLOTTING ONLY. Clustering and anomaly detection always run on
the 20-dimensional PCA space Z, never on the 2-D coordinates, because squeezing data into
2-D distorts distances.
"""
import numpy as np
from sklearn.manifold import trustworthiness

from src.edna import config


def pca_embedding(Z):
    """First two principal components (Z's columns are already ordered by variance)."""
    return np.asarray(Z)[:, :2].copy()


def umap_embedding(Z, n_neighbors=15, min_dist=0.1, seed=config.RANDOM_SEED):
    """2-D UMAP embedding of Z. Returns (fitted_reducer, coordinates).

    A fixed seed makes the result reproducible (UMAP then runs single-threaded).
    compatibility_layout=True pins the classic UMAP 0.5.x algorithm, so the plots do not
    change if a future UMAP release changes its default.
    """
    import umap  # imported here: it is slow to import and only needed when embedding

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed, n_jobs=1, compatibility_layout=True).fit(Z)
    return reducer, reducer.embedding_


def embedding_quality(Z, coordinates, n_neighbors=10):
    """Trustworthiness (0-1): do points that are neighbours in 2-D really neighbour in Z?"""
    return float(trustworthiness(Z, coordinates, n_neighbors=n_neighbors))