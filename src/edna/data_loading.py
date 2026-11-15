"""Load the two raw OBIS tables and link them into one table."""
import pandas as pd

from src.edna import config

# Only the columns we actually need (the raw files have 15 and 69 columns).
DNA_COLUMNS = ["id", "DNA_sequence"]
OCCURRENCE_COLUMNS = [
    "occurrenceID", "eventID", "eventDate",
    "decimalLatitude", "decimalLongitude",
    "organismQuantity",  # number of DNA reads for this sequence at this station
    "scientificName", "taxonRank",
    "phylum", "class", "order", "family", "genus",  # held-out taxonomy
]


def load_raw_tables():
    """Read both raw files. Returns (dna_table, occurrence_table)."""
    for path in (config.DNA_FILE, config.OCCURRENCE_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw data file: {path}\n"
                "Place the two OBIS files in data/raw/ and try again."
            )
    dna = pd.read_csv(config.DNA_FILE, sep="\t", usecols=DNA_COLUMNS)
    occ = pd.read_csv(
        config.OCCURRENCE_FILE, sep="\t", usecols=OCCURRENCE_COLUMNS, low_memory=False
    )
    return dna, occ


def link_tables(dna, occ):
    """Join DNA sequences to occurrence records.

    The DNA table's `id` equals the occurrence table's `occurrenceID`.
    Returns (linked_table, number_of_dna_rows_without_a_match).
    """
    dna = dna.rename(columns={"id": "occurrenceID"})
    n_unmatched = int((~dna["occurrenceID"].isin(occ["occurrenceID"])).sum())
    # many_to_one: several occurrence rows may share an ID, but each DNA ID is unique.
    linked = occ.merge(dna, on="occurrenceID", how="inner", validate="many_to_one")
    return linked, n_unmatched