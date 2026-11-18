"""Scale the k-mer features and compress them with PCA."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.edna import config


def scale_and_reduce(X, n_components=20, seed=config.RANDOM_SEED):
    """Standardise the features, then project onto the first `n_components` PCA axes.

    Returns (scaler, pca, Z): the fitted scaler, the fitted PCA, and the reduced
    matrix Z of shape (n_sequences, n_components). Keep the fitted objects so new
    sequences can be transformed the same way later.
    """
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=n_components, random_state=seed).fit(X_scaled)
    return scaler, pca, pca.transform(X_scaled)


def cumulative_variance(X, seed=config.RANDOM_SEED):
    """Cumulative share of variance explained by 1, 2, 3, ... components."""
    X_scaled = StandardScaler().fit_transform(X)
    full = PCA(random_state=seed).fit(X_scaled)
    return np.cumsum(full.explained_variance_ratio_)