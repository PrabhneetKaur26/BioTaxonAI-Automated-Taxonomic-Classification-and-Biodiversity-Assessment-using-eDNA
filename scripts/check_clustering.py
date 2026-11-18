"""Choose k for K-Means and fit the final model."""
import time

import numpy as np

from src.edna.clustering import choose_k, evaluate_k_range, fit_kmeans
from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.dimensionality import scale_and_reduce
from src.edna.features import kmer_frequency_matrix
from src.edna.preprocessing import build_asv_table, clean_asvs
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

dna, occ = load_raw_tables()
linked, _ = link_tables(dna, occ)
asv, _ = clean_asvs(build_asv_table(linked))
X, _ = kmer_frequency_matrix(asv["DNA_sequence"].tolist(), k=4)
_, _, Z = scale_and_reduce(X, n_components=20)

start = time.time()
table = evaluate_k_range(Z)
print(f"(k evaluation took {time.time() - start:.0f} s)\n")
print(table.round(3).to_string())

print("\nChosen k by threshold on stability:")
for threshold in (0.80, 0.85, 0.88):
    print(f"  min_stability = {threshold:.2f} -> k = {choose_k(table, min_stability=threshold)}")

K = choose_k(table)  # default rule: min_stability = 0.85
km = fit_kmeans(Z, K)
labels = km.labels_
print(f"\nFinal model: k = {K}")
print("Cluster sizes:", np.bincount(labels).tolist())
print(f"Silhouette {silhouette_score(Z, labels):.3f} | Davies-Bouldin {davies_bouldin_score(Z, labels):.3f} | Calinski-Harabasz {calinski_harabasz_score(Z, labels):.1f}")
labels_again = fit_kmeans(Z, K).labels_
print(f"Same seed gives identical labels: {bool((labels == labels_again).all())}")