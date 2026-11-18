"""Check the data-access layer against the saved pipeline results."""
import numpy as np

from src.edna import store

assert store.results_available(), store.NOT_READY_MESSAGE
tables = store.load_tables()
master, Z = tables["master"], tables["Z"]
print(f"Loaded {len(tables)} items. Master table: {master.shape}, PCA space: {Z.shape}")
print(f"Run created: {tables['metadata']['created_utc']}, k = {tables['metadata']['settings']['k']}")

# Search
by_genus = store.search_asvs(master, "clausocalanus")
assert len(by_genus) > 0 and by_genus["genus"].eq("Clausocalanus").any()
assert list(store.search_asvs(master, "ASV_066").index) == ["ASV_066"]
fragment = master.loc["ASV_066", "DNA_sequence"][50:64]
assert "ASV_066" in store.search_asvs(master, fragment).index
assert len(store.search_asvs(master, "no such organism xyz")) == 0
assert len(store.search_asvs(master, "cat")) < len(master)  # short text must not match DNA
print(f"Search: 'clausocalanus' -> {len(by_genus)} ASVs; id, DNA-fragment and no-match searches OK")
print(f"Filter cluster 2 + min 1000 reads -> {len(store.search_asvs(master, clusters=[2], min_reads=1000))} ASVs")

# Sequence-level investigation
profile = store.asv_station_profile(tables["matrix"], tables["stations"], "ASV_066")
nn = store.nearest_neighbours(Z, master, "ASV_066", n=5)
assert len(profile) == 20 and profile["reads"].sum() == master.loc["ASV_066", "total_reads"]
assert "ASV_066" not in nn.index and nn["distance"].is_monotonic_increasing
print(f"\nASV_066 is seen at {int((profile['reads'] > 0).sum())} stations; nearest neighbours:")
print(nn.round(2).to_string())

# Shortlist is configurable without retraining
sizes = {}
for top_n in (10, 20, 40, 80):
    sizes[top_n] = (len(store.candidate_shortlist(master, top_n)),
                    len(store.candidate_shortlist(master, top_n, only_priority=True)))
    print(f"Shortlist top_n={top_n:>2}: {sizes[top_n][0]} rows, {sizes[top_n][1]} priority")
assert sizes[40] == (40, 25)                       # matches the pipeline's default 8% budget
assert sizes[80][1] > sizes[40][1]                 # a bigger budget can add more priority ASVs
assert all(sizes[a][1] <= sizes[b][1] for a, b in ((10, 20), (20, 40), (40, 80)))
assert len(store.search_asvs(master, max_if_rank=20)) == 20

# Consistency: saved models reproduce the saved cluster labels and anomaly scores
models = store.load_models()
X = np.load(store.config.PROCESSED_DIR / "kmer_features.npy")
Z_again = models["pca"].transform(models["scaler"].transform(X))
assert np.allclose(Z_again, Z)
assert (models["kmeans"].predict(Z_again) == master["cluster_kmeans"].to_numpy()).all()
assert np.allclose(-models["isolation_forest"].score_samples(Z_again), master["if_score"].to_numpy())
print("\nConsistency: saved scaler+PCA+K-Means+Isolation Forest reproduce the saved labels and scores")
print("All checks passed")