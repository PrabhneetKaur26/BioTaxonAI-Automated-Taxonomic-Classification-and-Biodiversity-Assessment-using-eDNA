"""Score unusual sequences and build the candidate review table."""
from src.edna.anomaly import (build_candidate_table, posthoc_enrichment, ranking_stability,
                              score_anomalies)
from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.dimensionality import scale_and_reduce
from src.edna.features import kmer_frequency_matrix
from src.edna.preprocessing import build_asv_table, clean_asvs
from src.edna.taxonomy import build_taxonomy_table

dna, occ = load_raw_tables()
linked, _ = link_tables(dna, occ)
asv, _ = clean_asvs(build_asv_table(linked))
tax = build_taxonomy_table(linked, asv)
X, _ = kmer_frequency_matrix(asv["DNA_sequence"].tolist(), k=4)
_, _, Z = scale_and_reduce(X, n_components=20)

scores, iso_model = score_anomalies(Z, asv.index)
table = build_candidate_table(scores, asv, tax, top_fraction=0.08)

rho, overlap = ranking_stability(Z, asv.index)
print(f"Isolation Forest ranking across seeds: mean Spearman {rho:.3f}, mean top-40 overlap {overlap:.0f}/40")

print("\nHow the two detectors agree (top 8% each):")
print(table["agreement"].value_counts().to_string())
print("\nReview status:")
print(table["review_status"].value_counts().to_string())

print("\nPOST-HOC check (taxonomy was NOT used for scoring):")
for name, flag in [("Isolation Forest top 8%", table["if_flag"]), ("LOF top 8%", table["lof_flag"])]:
    r = posthoc_enrichment(flag, table["poorly_resolved"])
    print(f"  {name}: {r['share_poorly_resolved_flagged']:.0%} poorly resolved vs "
          f"{r['share_poorly_resolved_overall']:.0%} overall, odds ratio {r['odds_ratio']:.2f}, p = {r['p_value']:.1e}")

cols = ["if_score", "if_rank", "lof_rank", "total_reads", "n_stations", "review_status", "major_group", "rank"]
print("\nTop 8 most unusual sequences:")
print(table[cols].head(8).round(3).to_string())