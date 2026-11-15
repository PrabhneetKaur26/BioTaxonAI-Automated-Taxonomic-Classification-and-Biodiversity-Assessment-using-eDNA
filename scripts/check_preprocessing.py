"""Quick check of the preprocessing step."""
from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.preprocessing import (build_abundance_matrix, build_asv_table,
                                    build_stations, clean_asvs)

dna, occ = load_raw_tables()
linked, _ = link_tables(dna, occ)

stations = build_stations(linked)
asv_all = build_asv_table(linked)
asv, report = clean_asvs(asv_all)
matrix = build_abundance_matrix(linked, asv, stations)

print("Stations (first 3 rows):")
print(stations.head(3).to_string())
print(f"\nASVs before cleaning : {len(asv_all)}")
print(f"Cleaning report      : {report}")
print(f"ASVs kept            : {len(asv)}")
print(f"\nAbundance matrix shape        : {matrix.shape}  (ASVs x stations)")
print(f"Share of zero cells (sparsity): {(matrix.values == 0).mean():.3f}")
print(f"Reads per station: min {matrix.sum(axis=0).min()}, max {matrix.sum(axis=0).max()}")
print(f"Most abundant ASV: {asv.index[0]} with {asv.iloc[0]['total_reads']} reads at {asv.iloc[0]['n_stations']} stations")
print(f"Matrix total reads equals ASV table total reads: {int(matrix.values.sum()) == int(asv['total_reads'].sum())}")