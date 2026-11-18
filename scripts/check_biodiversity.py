"""Biodiversity per station, with and without correcting for sequencing depth."""
import numpy as np
import pandas as pd

from src.edna.biodiversity import (cluster_read_share, depth_dependence, expected_richness,
                                   latitude_trends, shannon_index, station_biodiversity)
from src.edna.clustering import fit_kmeans
from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.dimensionality import scale_and_reduce
from src.edna.features import kmer_frequency_matrix
from src.edna.preprocessing import (build_abundance_matrix, build_asv_table, build_stations,
                                    clean_asvs)

# 1) Hand-checkable sanity tests
assert abs(shannon_index([1, 1, 1, 1]) - np.log(4)) < 1e-9
assert expected_richness([5, 5, 5], depth=15) == 3          # full depth sees everything
assert abs(expected_richness([1, 1], depth=1) - 1.0) < 1e-9  # one read can reveal only one ASV
print("Sanity tests passed: Shannon(1,1,1,1)=ln 4; full-depth richness; one-read richness")

# 2) Real data
dna, occ = load_raw_tables()
linked, _ = link_tables(dna, occ)
stations = build_stations(linked)
asv, _ = clean_asvs(build_asv_table(linked))
matrix = build_abundance_matrix(linked, asv, stations)

table = station_biodiversity(matrix, stations)
depth = table.attrs["rarefaction_depth"]
pd.set_option("display.width", 200)
print(f"\nPER-STATION DIVERSITY (rarefied to {depth} reads, the smallest library)")
print(table.round(2).to_string())

print("\nDoes sequencing depth drive the numbers? Spearman(depth, measure):")
for name, (rho, p) in depth_dependence(table).items():
    print(f"  {name:<18} rho = {rho:+.2f}  (p = {p:.3f})")

print("\nTrend with latitude (descriptive; only 20 stations):")
print(latitude_trends(table).round(3).to_string())

labels = fit_kmeans(scale_and_reduce(kmer_frequency_matrix(asv["DNA_sequence"].tolist(), 4)[0], 20)[2], 9).labels_
share = cluster_read_share(matrix, labels)
print(f"\nCluster read share: shape {share.shape} (clusters x stations), columns sum to 1: {bool(np.allclose(share.sum(axis=0), 1))}")