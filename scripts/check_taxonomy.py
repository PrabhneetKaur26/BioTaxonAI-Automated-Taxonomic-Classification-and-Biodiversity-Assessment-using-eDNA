"""Quick check of the held-out taxonomy table."""
from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.preprocessing import build_asv_table, clean_asvs
from src.edna.taxonomy import build_taxonomy_table, count_conflicting_labels

dna, occ = load_raw_tables()
linked, _ = link_tables(dna, occ)
asv, _ = clean_asvs(build_asv_table(linked))
tax = build_taxonomy_table(linked, asv)

print(f"Rows in taxonomy table: {len(tax)} (ASVs: {len(asv)}), same order: {list(tax.index) == list(asv.index)}")
print("\nTaxonomic rank of the best label:")
print(tax["rank"].value_counts().to_string())
print("\nMajor groups:")
print(tax["major_group"].value_counts().to_string())
print(f"\nASVs without a major group (excluded from ARI/NMI): {(~tax['has_group']).sum()}")
print(f"Poorly resolved ASVs (rank order or coarser): {tax['poorly_resolved'].sum()} ({tax['poorly_resolved'].mean():.1%})")
print(f"ASVs whose name differs between stations: {count_conflicting_labels(linked, asv)}")
print("\nExample rows:")
print(tax[["major_group", "tax_class", "genus", "rank"]].head(4).to_string())