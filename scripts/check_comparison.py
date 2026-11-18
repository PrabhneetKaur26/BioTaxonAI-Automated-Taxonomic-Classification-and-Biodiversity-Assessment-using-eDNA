"""Fit hierarchical clustering and DBSCAN, then compare them with K-Means."""
import numpy as np

from src.edna.clustering import fit_kmeans
from src.edna.comparison import (NOISE, choose_dbscan_params, compare_internal,
                                 dbscan_grid, fit_dbscan, fit_hierarchical)
from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.dimensionality import scale_and_reduce
from src.edna.features import kmer_frequency_matrix
from src.edna.preprocessing import build_asv_table, clean_asvs

K = 9  # chosen in the previous step (the final pipeline will compute and save it)

dna, occ = load_raw_tables()
linked, _ = link_tables(dna, occ)
asv, _ = clean_asvs(build_asv_table(linked))
X, _ = kmer_frequency_matrix(asv["DNA_sequence"].tolist(), k=4)
_, _, Z = scale_and_reduce(X, n_components=20)

labels_km = fit_kmeans(Z, K).labels_
labels_ward, _ = fit_hierarchical(Z, K, "ward")
print("Ward cluster sizes   :", sorted(np.bincount(labels_ward).tolist(), reverse=True))
print("K-Means cluster sizes:", sorted(np.bincount(labels_km).tolist(), reverse=True))

grid = dbscan_grid(Z)
print("\nDBSCAN grid (clusters / noise % / silhouette on non-noise points):")
print(grid.round(3).to_string(index=False))
eps, min_samples = choose_dbscan_params(grid)
labels_db = fit_dbscan(Z, eps, min_samples)
print(f"\nChosen DBSCAN: eps={eps}, min_samples={min_samples} -> "
      f"{len(set(labels_db) - {NOISE})} clusters, {(labels_db == NOISE).sum()} noise points "
      f"({(labels_db == NOISE).mean():.1%})")

table, n_shared = compare_internal(Z, {"K-Means": labels_km, "Hierarchical (Ward)": labels_ward,
                                       "DBSCAN": labels_db})
print(f"\nInternal comparison (shared points = {n_shared} ASVs that no method called noise):")
print(table.round(3).to_string())