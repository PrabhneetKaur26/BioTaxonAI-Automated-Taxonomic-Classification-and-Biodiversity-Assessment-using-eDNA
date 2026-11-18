"""Read the saved pipeline results and answer the questions the app asks.

No Streamlit code lives here, so everything can be tested on its own. Nothing is retrained:
the app only loads what `python -m scripts.run_pipeline` saved.
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.edna import config

NOT_READY_MESSAGE = ("Pipeline results not found. Run `python -m scripts.run_pipeline` "
                     "from the project root first.")

REQUIRED_FILES = [
    config.OUTPUTS_DIR / "asv_master.csv",
    config.OUTPUTS_DIR / "station_biodiversity.csv",
    config.OUTPUTS_DIR / "cluster_profiles.csv",
    config.OUTPUTS_DIR / "cluster_read_share.csv",
    config.OUTPUTS_DIR / "internal_comparison.csv",
    config.OUTPUTS_DIR / "external_comparison.csv",
    config.OUTPUTS_DIR / "dbscan_grid.csv",
    config.OUTPUTS_DIR / "run_metadata.json",
    config.PROCESSED_DIR / "stations.csv",
    config.PROCESSED_DIR / "abundance_matrix.csv",
    config.PROCESSED_DIR / "pca_space.npy",
]


def results_available():
    """True when every file the app needs has been produced by the pipeline."""
    return all(path.exists() for path in REQUIRED_FILES)


def load_tables():
    """Load every saved result table into a dictionary."""
    if not results_available():
        raise FileNotFoundError(NOT_READY_MESSAGE)
    out, processed = config.OUTPUTS_DIR, config.PROCESSED_DIR
    tables = {
        "master": pd.read_csv(out / "asv_master.csv", index_col=0),
        "biodiversity": pd.read_csv(out / "station_biodiversity.csv", index_col=0),
        "profiles": pd.read_csv(out / "cluster_profiles.csv", index_col=0),
        "read_share": pd.read_csv(out / "cluster_read_share.csv", index_col=0),
        "internal": pd.read_csv(out / "internal_comparison.csv", index_col=0),
        "external": pd.read_csv(out / "external_comparison.csv"),
        "dbscan_grid": pd.read_csv(out / "dbscan_grid.csv"),
        "stations": pd.read_csv(processed / "stations.csv", index_col=0),
        "matrix": pd.read_csv(processed / "abundance_matrix.csv", index_col=0),
        "Z": np.load(processed / "pca_space.npy"),
        "metadata": json.loads((out / "run_metadata.json").read_text()),
    }
    k_path = out / "k_selection.csv"
    tables["k_selection"] = pd.read_csv(k_path, index_col=0) if k_path.exists() else None
    return tables


def load_models():
    """Load the fitted scaler, PCA, K-Means and Isolation Forest."""
    names = ("scaler", "pca", "kmeans", "isolation_forest")
    paths = {name: config.MODELS_DIR / f"{name}.joblib" for name in names}
    if not all(path.exists() for path in paths.values()):
        raise FileNotFoundError(NOT_READY_MESSAGE)
    return {name: joblib.load(path) for name, path in paths.items()}


def search_asvs(master, query="", clusters=None, max_if_rank=None, min_reads=0):
    """Filter the master table.

    query       : matches ASV id, scientific name, genus or major group (case-insensitive);
                  a DNA fragment of 8+ letters (A/C/G/T) is searched inside the sequences.
    clusters    : keep only these K-Means clusters
    max_if_rank : keep only the max_if_rank most unusual ASVs (Isolation Forest rank)
    min_reads   : keep ASVs with at least this many reads in total
    """
    result = master
    query = query.strip()
    if query:
        text = (master.index.to_series().str.lower() + " "
                + master["scientific_name"].fillna("").str.lower() + " "
                + master["genus"].fillna("").str.lower() + " "
                + master["major_group"].str.lower())
        hit = text.str.contains(query.lower(), regex=False)
        fragment = query.upper()
        if len(fragment) >= 8 and set(fragment) <= set("ACGT"):
            hit = hit | master["DNA_sequence"].str.contains(fragment, regex=False)
        result = master[hit]
    if clusters:
        result = result[result["cluster_kmeans"].isin(clusters)]
    if max_if_rank is not None:
        result = result[result["if_rank"] <= max_if_rank]
    return result[result["total_reads"] >= min_reads]


def asv_station_profile(matrix, stations, asv_id):
    """Reads of one ASV at every station (zeros included), south to north."""
    return pd.DataFrame({"lat": stations["lat"], "reads": matrix.loc[asv_id]})


def nearest_neighbours(Z, master, asv_id, n=5):
    """The n ASVs closest to `asv_id` in the PCA space (sequence-level investigation)."""
    if len(Z) != len(master):
        raise ValueError("PCA space and master table are out of step; rerun the pipeline.")
    position = master.index.get_loc(asv_id)
    distance = np.linalg.norm(Z - Z[position], axis=1)
    order = [i for i in np.argsort(distance) if i != position][:n]
    neighbours = master.iloc[order][["cluster_kmeans", "major_group", "scientific_name", "rank"]].copy()
    neighbours.insert(0, "distance", distance[order])
    return neighbours


def candidate_shortlist(master, top_n=40, only_priority=False):
    """The top_n most unusual ASVs (by Isolation Forest rank), optionally priority-only.

    top_n is a configurable REVIEW BUDGET, not an estimate of how many novel taxa exist.
    The status is computed here from the saved rank and `robust` flag (seen at 3+ stations
    with 50+ reads), so it always matches the chosen top_n. The `review_status` column saved
    by the pipeline only reflects the pipeline's default budget and is not used here.
    """
    table = master[master["if_rank"] <= top_n].sort_values("if_rank").copy()
    table["review_status"] = np.where(table["robust"], "Priority candidate",
                                      "Low support (possible artefact)")
    if only_priority:
        table = table[table["robust"]]
    return table


def to_csv_bytes(table):
    """Bytes for a Streamlit download button."""
    return table.to_csv().encode("utf-8")