"""Hierarchical clustering, DBSCAN and a fair internal comparison with K-Means."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

NOISE = -1  # DBSCAN's label for points that belong to no cluster


def fit_hierarchical(Z, k, method="ward"):
    """Agglomerative clustering cut into k clusters.

    Returns (labels, linkage_matrix). Ward linkage merges the pair of clusters that
    increases within-cluster variance the least, which gives balanced groups.
    """
    link = linkage(Z, method)
    labels = fcluster(link, k, criterion="maxclust") - 1  # make labels start at 0
    return labels, link


def dbscan_grid(Z, eps_values=(3, 4, 5, 6, 7, 8), min_samples_values=(3, 5, 8, 10)):
    """Try DBSCAN settings; report clusters found, noise share and silhouette (non-noise)."""
    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit(Z).labels_
            kept = labels != NOISE
            n_clusters = len(set(labels[kept]))
            silhouette = (silhouette_score(Z[kept], labels[kept])
                          if n_clusters > 1 else np.nan)
            rows.append({"eps": eps, "min_samples": min_samples, "clusters": n_clusters,
                         "noise_pct": round(100 * (~kept).mean(), 1),
                         "silhouette_non_noise": silhouette})
    return pd.DataFrame(rows)


def choose_dbscan_params(grid, min_clusters=5, max_clusters=20, max_noise_pct=25):
    """Best silhouette among settings with a sensible cluster count and capped noise.

    The noise cap matters: silhouette on non-noise points rises when more points are
    thrown away, so without a cap the rule would discard most of the data.
    """
    ok = grid[grid["clusters"].between(min_clusters, max_clusters)
              & (grid["noise_pct"] <= max_noise_pct)].dropna()
    if ok.empty:
        raise ValueError("No DBSCAN setting satisfies the cluster-count and noise limits.")
    best = ok.sort_values("silhouette_non_noise", ascending=False).iloc[0]
    return float(best["eps"]), int(best["min_samples"])


def fit_dbscan(Z, eps, min_samples):
    """Fit DBSCAN. Noise points get the label -1."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit(Z).labels_


def _internal_scores(Z, labels, mask):
    """Silhouette, Davies-Bouldin and Calinski-Harabasz on the points selected by mask."""
    z, lab = Z[mask], labels[mask]
    if len(set(lab)) < 2:
        return np.nan, np.nan, np.nan
    return (silhouette_score(z, lab), davies_bouldin_score(z, lab),
            calinski_harabasz_score(z, lab))


def compare_internal(Z, partitions):
    """Compare clusterings with internal metrics, on a fair footing.

    `partitions` maps a method name to its label array (DBSCAN noise = -1).
    Two views are reported:
      * own points    : every point the method itself assigned to a cluster
      * shared points : only points that NO method called noise, so every method is
                        scored on exactly the same data.
    """
    shared = np.ones(len(Z), dtype=bool)
    for labels in partitions.values():
        shared &= labels != NOISE
    rows = []
    for name, labels in partitions.items():
        own = labels != NOISE
        sil_o, db_o, ch_o = _internal_scores(Z, labels, own)
        sil_s, db_s, ch_s = _internal_scores(Z, labels, shared)
        rows.append({"method": name, "clusters": len(set(labels[own])),
                     "points_clustered_pct": round(100 * own.mean(), 1),
                     "silhouette_own": sil_o, "davies_bouldin_own": db_o,
                     "silhouette_shared": sil_s, "davies_bouldin_shared": db_s})
    return pd.DataFrame(rows).set_index("method"), int(shared.sum())