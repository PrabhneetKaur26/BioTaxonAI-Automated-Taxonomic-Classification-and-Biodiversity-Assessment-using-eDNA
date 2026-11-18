"""External validation and interpretation of clusters using the held-out taxonomy.

Internal metrics (silhouette, Davies-Bouldin) judge cluster shape using only the data.
External metrics (ARI, NMI, ...) compare clusters with an independent answer key, the
reference-database taxonomy. The taxonomy is never used to build the clusters.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, completeness_score, homogeneity_score,
                             normalized_mutual_info_score)

from src.edna.comparison import NOISE

LEVELS = ("major_group", "tax_class")  # coarse (5 groups) and finer (classes) answer keys


def external_scores(labels, truth, mask):
    """ARI, NMI, homogeneity and completeness on the points selected by mask."""
    y, c = np.asarray(truth)[mask], np.asarray(labels)[mask]
    return {"ARI": adjusted_rand_score(y, c),
            "NMI": normalized_mutual_info_score(y, c),
            "homogeneity": homogeneity_score(y, c),
            "completeness": completeness_score(y, c)}


def compare_external(partitions, tax):
    """External scores for every method, at two taxonomy levels and on two point sets.

    own points    : every ASV the method assigned to a cluster (and that has a label)
    shared points : ASVs that no method called noise (identical data for every method)
    ASVs without a major group (Unassigned) are always excluded.
    """
    usable = tax["has_group"].to_numpy()
    shared = usable.copy()
    for labels in partitions.values():
        shared &= np.asarray(labels) != NOISE
    rows = []
    for name, labels in partitions.items():
        labels = np.asarray(labels)
        for view, mask in (("own points", usable & (labels != NOISE)),
                           ("shared points", shared)):
            for level in LEVELS:
                rows.append({"method": name, "view": view, "level": level,
                             "n_points": int(mask.sum()),
                             **external_scores(labels, tax[level].to_numpy(), mask)})
    return pd.DataFrame(rows)


def random_baseline(labels, tax, level="tax_class", repeats=200, seed=0):
    """Scores of the SAME cluster sizes assigned at random: what 'no signal' looks like."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    mask = tax["has_group"].to_numpy() & (labels != NOISE)
    truth = tax[level].to_numpy()
    results = [external_scores(rng.permutation(labels), truth, mask) for _ in range(repeats)]
    return {metric: float(np.mean([r[metric] for r in results])) for metric in results[0]}


def cluster_profiles(labels, asv, tax, gc, Z):
    """One row per cluster describing its members (taxonomy is used for description only)."""
    labels = np.asarray(labels)
    rows = []
    for cluster in sorted(set(labels) - {NOISE}):
        members = labels == cluster
        sub = tax[members]
        known = sub[sub["has_group"]]
        groups = known["major_group"].value_counts()
        classes = known["tax_class"].value_counts()
        genera = sub["genus"].dropna().value_counts().head(3)
        centroid = Z[members].mean(axis=0)
        representative = sub.index[np.linalg.norm(Z[members] - centroid, axis=1).argmin()]
        rows.append({
            "cluster": int(cluster), "size": int(members.sum()),
            "dominant_group": groups.index[0] if len(groups) else "Unassigned",
            "group_purity": groups.iloc[0] / len(known) if len(known) else np.nan,
            "dominant_class": classes.index[0] if len(classes) else "Unassigned",
            "class_purity": classes.iloc[0] / len(known) if len(known) else np.nan,
            "top_genera": ", ".join(f"{g} ({n})" for g, n in genera.items()) or "-",
            "unassigned": int((~sub["has_group"]).sum()),
            "poorly_resolved_pct": 100 * sub["poorly_resolved"].mean(),
            "mean_gc": float(np.asarray(gc)[members].mean()),
            "median_stations": float(asv.loc[sub.index, "n_stations"].median()),
            "reads_share_pct": 100 * asv.loc[sub.index, "total_reads"].sum() / asv["total_reads"].sum(),
            "representative_asv": representative,
        })
    return pd.DataFrame(rows).set_index("cluster")