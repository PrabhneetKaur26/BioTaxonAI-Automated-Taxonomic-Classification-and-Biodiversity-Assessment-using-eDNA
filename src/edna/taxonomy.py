"""Build the held-out taxonomy table.

IMPORTANT: this table is used ONLY after clustering, to interpret and validate the
results (ARI, NMI, cluster profiles, novelty checks). It must never be used as an input
feature for clustering or anomaly detection.
"""
import numpy as np
import pandas as pd

UNASSIGNED = "Unassigned"

# Ranks above family: the reference database could only place the sequence this coarsely.
COARSE_RANKS = {"kingdom", "phylum", "class", "order"}

# raw occurrence column -> clean column name used in this project
RENAME = {
    "phylum": "major_group",   # in this OBIS export 'phylum' holds PR2 major groups
    "class": "tax_class",
    "order": "tax_order",
    "family": "family",
    "genus": "genus",
    "scientificName": "scientific_name",
    "taxonRank": "rank",
}


def _most_common(values):
    """Most frequent non-missing value (alphabetical tie-break); NaN if all missing."""
    values = values.dropna()
    return values.mode().iloc[0] if len(values) else np.nan


def build_taxonomy_table(linked, asv):
    """One row per cleaned ASV with its reference-database taxonomy.

    The same sequence can carry slightly different labels at different stations, so we
    take the most common label per ASV.
    """
    data = linked.assign(taxonRank=linked["taxonRank"].str.lower())  # 'Species' -> 'species'
    data = data.merge(asv[["DNA_sequence"]].reset_index(), on="DNA_sequence")
    tax = data.groupby("asv_id").agg(
        {raw: _most_common for raw in RENAME}
    ).rename(columns=RENAME)
    tax = tax.loc[asv.index]
    tax["major_group"] = tax["major_group"].fillna(UNASSIGNED)
    tax["tax_class"] = tax["tax_class"].fillna(UNASSIGNED)
    tax["has_group"] = tax["major_group"] != UNASSIGNED          # usable for evaluation
    tax["poorly_resolved"] = tax["rank"].isin(COARSE_RANKS)      # database gap, not proof of novelty
    return tax


def count_conflicting_labels(linked, asv):
    """How many ASVs carry more than one distinct scientific name across stations."""
    data = linked.merge(asv[["DNA_sequence"]].reset_index(), on="DNA_sequence")
    n_names = data.groupby("asv_id")["scientificName"].nunique()
    return int((n_names > 1).sum())