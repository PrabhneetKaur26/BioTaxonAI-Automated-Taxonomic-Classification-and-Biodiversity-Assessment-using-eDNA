"""Quick check that the raw data loads and links correctly."""
from src.edna.data_loading import link_tables, load_raw_tables

dna, occ = load_raw_tables()
linked, n_unmatched = link_tables(dna, occ)

print(f"DNA table rows              : {len(dna)}")
print(f"Occurrence table rows       : {len(occ)}")
print(f"DNA rows with no occurrence : {n_unmatched}")
print(f"Linked records              : {len(linked)}")
print(f"Unique DNA sequences (ASVs) : {linked['DNA_sequence'].nunique()}")
print(f"Sampling stations           : {linked['eventID'].nunique()}")
print(f"Latitude range              : {linked['decimalLatitude'].min()} to {linked['decimalLatitude'].max()}")
print(f"Missing values in key columns:\n{linked[['DNA_sequence', 'eventID', 'organismQuantity']].isna().sum().to_string()}")