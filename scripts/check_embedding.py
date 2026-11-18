"""Build the 2-D PCA and UMAP embeddings and check their quality."""
import time

import numpy as np

from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.dimensionality import scale_and_reduce
from src.edna.embedding import embedding_quality, pca_embedding, umap_embedding
from src.edna.features import kmer_frequency_matrix
from src.edna.preprocessing import build_asv_table, clean_asvs

dna, occ = load_raw_tables()
linked, _ = link_tables(dna, occ)
asv, _ = clean_asvs(build_asv_table(linked))
X, _ = kmer_frequency_matrix(asv["DNA_sequence"].tolist(), k=4)
_, _, Z = scale_and_reduce(X, n_components=20)

pca2 = pca_embedding(Z)
start = time.time()
_, umap2 = umap_embedding(Z)
print(f"UMAP took {time.time() - start:.0f} s (the first run is slow while numba compiles)")
_, umap2_again = umap_embedding(Z)

print(f"\nPCA  coordinates: {pca2.shape}   UMAP coordinates: {umap2.shape}")
print(f"Any missing values: {bool(np.isnan(pca2).any() or np.isnan(umap2).any())}")
print(f"UMAP reproducible with a fixed seed: {bool(np.allclose(umap2, umap2_again))}")

print("\nTrustworthiness at 10 neighbours (1.0 = neighbourhoods perfectly preserved):")
print(f"  PCA  (2 components): {embedding_quality(Z, pca2):.3f}")
print(f"  UMAP               : {embedding_quality(Z, umap2):.3f}")