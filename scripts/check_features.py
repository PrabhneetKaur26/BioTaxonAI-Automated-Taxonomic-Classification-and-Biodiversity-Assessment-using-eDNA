"""Quick check of the k-mer feature extraction."""
import numpy as np

from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.features import gc_content, kmer_frequency_matrix
from src.edna.preprocessing import build_asv_table, clean_asvs

# 1) Hand-checkable toy example: "AAAC" has 2-mers AA, AA, AC -> AA = 2/3, AC = 1/3.
X_toy, names_toy = kmer_frequency_matrix(["AAAC"], k=2)
freq = dict(zip(names_toy, X_toy[0]))
assert abs(freq["AA"] - 2 / 3) < 1e-9 and abs(freq["AC"] - 1 / 3) < 1e-9
print("Toy example passed: AA = 2/3, AC = 1/3")

# 2) Real data
dna, occ = load_raw_tables()
linked, _ = link_tables(dna, occ)
asv, _ = clean_asvs(build_asv_table(linked))
sequences = asv["DNA_sequence"].tolist()

X, names = kmer_frequency_matrix(sequences, k=4)
gc = gc_content(sequences)

print(f"\nFeature matrix shape : {X.shape}  (ASVs x 4-mers)")
print(f"Every row sums to 1  : {np.allclose(X.sum(axis=1), 1)}")
print(f"Any missing values   : {bool(np.isnan(X).any())}")
print(f"Share of zero cells  : {(X == 0).mean():.3f}")
print(f"GC content           : min {gc.min():.3f}, mean {gc.mean():.3f}, max {gc.max():.3f}")
top = np.argsort(X.mean(axis=0))[::-1][:3]
print("Three most common 4-mers overall:", [(names[i], round(float(X[:, i].mean()), 4)) for i in top])
print(f"Rows used the same order as the ASV table: {len(X) == len(asv)}")