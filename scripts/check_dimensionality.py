"""Quick check of feature scaling and PCA."""
import numpy as np

from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.dimensionality import cumulative_variance, scale_and_reduce
from src.edna.features import gc_content, kmer_frequency_matrix
from src.edna.preprocessing import build_asv_table, clean_asvs

dna, occ = load_raw_tables()
linked, _ = link_tables(dna, occ)
asv, _ = clean_asvs(build_asv_table(linked))
sequences = asv["DNA_sequence"].tolist()
X, names = kmer_frequency_matrix(sequences, k=4)

scaler, pca, Z = scale_and_reduce(X, n_components=20)
X_scaled = scaler.transform(X)

print(f"Input X shape: {X.shape} -> reduced Z shape: {Z.shape}")
print(f"Constant (zero-variance) k-mer columns: {int((X.std(axis=0) == 0).sum())}")
print(f"After scaling: mean of column means = {X_scaled.mean(axis=0).mean():.4f}, mean of column std = {X_scaled.std(axis=0).mean():.4f}")

cum = cumulative_variance(X)
print("\nCumulative variance explained:")
for n in (2, 5, 10, 20, 30, 50):
    print(f"  {n:>2} components: {cum[n - 1]:.1%}")
print("Components needed for 70% / 80% / 90%:", [int(np.searchsorted(cum, q) + 1) for q in (0.7, 0.8, 0.9)])
print(f"Chosen 20 components keep {pca.explained_variance_ratio_.sum():.1%} of the variance")

gc = gc_content(sequences)
print("\nCorrelation of GC content with PC1 / PC2: %.2f / %.2f" % (np.corrcoef(gc, Z[:, 0])[0, 1], np.corrcoef(gc, Z[:, 1])[0, 1]))

_, _, Z2 = scale_and_reduce(X, n_components=20)
print(f"Reproducible (two runs identical): {np.allclose(np.abs(Z), np.abs(Z2))}")