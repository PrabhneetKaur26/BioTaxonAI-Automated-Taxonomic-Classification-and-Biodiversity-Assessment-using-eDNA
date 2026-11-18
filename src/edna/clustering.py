"""K-Means clustering and the data-driven choice of the number of clusters (k)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

from src.edna import config

DEFAULT_SEEDS = (42, 0, 1, 7, 123)


def fit_kmeans(Z, k, seed=config.RANDOM_SEED, n_init=20):
    """Fit K-Means with k clusters. n_init restarts guard against bad starting points."""
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z)


def _subsample_stability(Z, k, full_labels, seed, n_subsamples, fraction):
    """Average ARI between the full clustering and clusterings of random subsamples."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_subsamples):
        idx = rng.choice(len(Z), int(fraction * len(Z)), replace=False)
        sub_labels = KMeans(k, n_init=10, random_state=int(rng.integers(1_000_000))).fit(Z[idx]).labels_
        scores.append(adjusted_rand_score(full_labels[idx], sub_labels))
    return float(np.mean(scores))


def evaluate_k_range(Z, k_values=range(2, 16), seeds=DEFAULT_SEEDS,
                     n_subsamples=20, fraction=0.8):
    """Internal-metric table for each k, averaged over several random seeds.

    Columns: inertia, silhouette, davies_bouldin, calinski_harabasz,
    stability (mean ARI under 80% subsampling) and stability_sd (spread across seeds).
    No taxonomy is used here.
    """
    rows = []
    for k in k_values:
        per_seed = {"inertia": [], "silhouette": [], "davies_bouldin": [],
                    "calinski_harabasz": [], "stability": []}
        for seed in seeds:
            km = fit_kmeans(Z, k, seed=seed, n_init=10)
            per_seed["inertia"].append(km.inertia_)
            per_seed["silhouette"].append(silhouette_score(Z, km.labels_))
            per_seed["davies_bouldin"].append(davies_bouldin_score(Z, km.labels_))
            per_seed["calinski_harabasz"].append(calinski_harabasz_score(Z, km.labels_))
            per_seed["stability"].append(
                _subsample_stability(Z, k, km.labels_, seed, n_subsamples, fraction))
        row = {"k": k, **{name: float(np.mean(v)) for name, v in per_seed.items()}}
        row["stability_sd"] = float(np.std(per_seed["stability"]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")


def choose_k(table, k_min=3, k_max=12, min_stability=0.85):
    """Pick k: among k values whose stability >= min_stability, take the best silhouette.

    Silhouette alone keeps rising with k on this data, so stability acts as the gate.
    """
    candidates = table.loc[k_min:k_max]
    stable = candidates[candidates["stability"] >= min_stability]
    if stable.empty:
        raise ValueError(f"No k in {k_min}..{k_max} reaches stability {min_stability}.")
    return int(stable["silhouette"].idxmax())