"""External validation of the clusterings against the held-out taxonomy."""
import pandas as pd

from src.edna.clustering import fit_kmeans
from src.edna.comparison import choose_dbscan_params, dbscan_grid, fit_dbscan, fit_hierarchical
from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.dimensionality import scale_and_reduce
from src.edna.evaluation import cluster_profiles, compare_external, random_baseline
from src.edna.features import gc_content, kmer_frequency_matrix
from src.edna.preprocessing import build_asv_table, clean_asvs
from src.edna.taxonomy import build_taxonomy_table

K = 9  # chosen earlier by the stability-gated rule

dna, occ = load_raw_tables()
linked, _ = link_tables(dna, occ)
asv, _ = clean_asvs(build_asv_table(linked))
tax = build_taxonomy_table(linked, asv)
sequences = asv["DNA_sequence"].tolist()
X, _ = kmer_frequency_matrix(sequences, k=4)
_, _, Z = scale_and_reduce(X, n_components=20)
gc = gc_content(sequences)

labels_km = fit_kmeans(Z, K).labels_
labels_ward, _ = fit_hierarchical(Z, K, "ward")
eps, min_samples = choose_dbscan_params(dbscan_grid(Z))
labels_db = fit_dbscan(Z, eps, min_samples)
partitions = {"K-Means": labels_km, "Hierarchical (Ward)": labels_ward, "DBSCAN": labels_db}

result = compare_external(partitions, tax)
pd.set_option("display.width", 200)
for view in ("own points", "shared points"):
    print(f"\nEXTERNAL VALIDATION - {view}")
    sub = result[result["view"] == view].drop(columns="view")
    print(sub.round(3).to_string(index=False))

base = random_baseline(labels_km, tax)
print("\nRandom-label baseline (K-Means cluster sizes, class level): "
      + ", ".join(f"{k} {v:.3f}" for k, v in base.items()))

print("\nK-MEANS CLUSTER PROFILES (taxonomy used for description only)")
profile = cluster_profiles(labels_km, asv, tax, gc, Z)
show = ["size", "dominant_group", "group_purity", "dominant_class", "class_purity",
        "unassigned", "poorly_resolved_pct", "reads_share_pct", "representative_asv"]
print(profile[show].round(2).to_string())
print("\nTop genera per cluster:")
print(profile["top_genera"].to_string())