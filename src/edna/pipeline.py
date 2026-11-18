"""Run the whole analysis once and save the models and result tables.

The Streamlit app only READS what this module saves; it never retrains anything.
"""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.edna import config
from src.edna.anomaly import build_candidate_table, score_anomalies
from src.edna.biodiversity import cluster_read_share, station_biodiversity
from src.edna.clustering import choose_k, evaluate_k_range, fit_kmeans
from src.edna.comparison import (NOISE, choose_dbscan_params, compare_internal, dbscan_grid,
                                 fit_dbscan, fit_hierarchical)
from src.edna.data_loading import link_tables, load_raw_tables
from src.edna.dimensionality import scale_and_reduce
from src.edna.embedding import pca_embedding, umap_embedding
from src.edna.evaluation import cluster_profiles, compare_external, random_baseline
from src.edna.features import gc_content, kmer_frequency_matrix
from src.edna.preprocessing import (build_abundance_matrix, build_asv_table, build_stations,
                                    clean_asvs)
from src.edna.taxonomy import build_taxonomy_table


def run_pipeline(k=None, verbose=True):
    """Run every stage. If k is None it is chosen by the stability-gated rule (slow, ~2 min).

    Returns a dictionary holding every fitted model and result table.
    """
    start = time.time()

    def say(message):
        if verbose:
            print(f"[{time.time() - start:6.1f} s] {message}", flush=True)

    say("Loading and linking raw OBIS tables")
    dna, occ = load_raw_tables()
    linked, n_unmatched = link_tables(dna, occ)
    stations = build_stations(linked)
    asv_all = build_asv_table(linked)
    asv, clean_report = clean_asvs(asv_all)
    matrix = build_abundance_matrix(linked, asv, stations)
    tax = build_taxonomy_table(linked, asv)

    say("Extracting k-mer features, scaling and PCA")
    sequences = asv["DNA_sequence"].tolist()
    X, kmer_names = kmer_frequency_matrix(sequences, k=config.KMER_SIZE)
    scaler, pca, Z = scale_and_reduce(X, n_components=config.N_COMPONENTS)
    gc = gc_content(sequences)

    k_table = None
    if k is None:
        say("Evaluating k = 2..15 (slow: 5 seeds x 20 subsamples per k)")
        k_table = evaluate_k_range(Z)
        k = choose_k(k_table, min_stability=config.MIN_STABILITY)
        k_source = f"stability-gated rule (min_stability={config.MIN_STABILITY})"
    else:
        k_source = "set by the user (--k)"
    say(f"Using k = {k} ({k_source})")

    say("Fitting K-Means, Ward hierarchical clustering and DBSCAN")
    kmeans = fit_kmeans(Z, k)
    labels_km = kmeans.labels_
    labels_ward, linkage_matrix = fit_hierarchical(Z, k, "ward")
    grid = dbscan_grid(Z)
    eps, min_samples = choose_dbscan_params(grid)
    labels_db = fit_dbscan(Z, eps, min_samples)
    partitions = {"K-Means": labels_km, "Hierarchical (Ward)": labels_ward, "DBSCAN": labels_db}

    say("Evaluating clusterings (internal, external, random baseline, profiles)")
    internal, n_shared = compare_internal(Z, partitions)
    external = compare_external(partitions, tax)
    baseline = random_baseline(labels_km, tax)
    profiles = cluster_profiles(labels_km, asv, tax, gc, Z)

    say("Scoring unusual sequences (Isolation Forest + LOF)")
    scores, iso_model = score_anomalies(Z, asv.index)
    candidates = build_candidate_table(scores, asv, tax, top_fraction=config.TOP_FRACTION)

    say("Computing depth-corrected biodiversity")
    biodiversity = station_biodiversity(matrix, stations)
    read_share = cluster_read_share(matrix, labels_km)

    say("Building 2-D embeddings (UMAP is slow the first time)")
    pca2 = pca_embedding(Z)
    umap_model, umap2 = umap_embedding(Z)

    say("Assembling the master ASV table")
    master = asv.join(tax)
    master["gc_content"] = gc
    master["cluster_kmeans"] = labels_km
    master["cluster_ward"] = labels_ward
    master["cluster_dbscan"] = labels_db
    master[["pca1", "pca2"]] = pca2
    master[["umap1", "umap2"]] = umap2
    master = master.join(candidates[["if_score", "if_rank", "lof_score", "lof_rank",
                                     "agreement", "robust", "review_status"]])

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "library_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                             "umap-learn": _version("umap")},
        "raw_data_sha256": {p.name: _sha256(p) for p in (config.DNA_FILE, config.OCCURRENCE_FILE)},
        "settings": {"random_seed": config.RANDOM_SEED, "kmer_size": config.KMER_SIZE,
                     "pca_components": config.N_COMPONENTS, "k": int(k), "k_source": k_source,
                     "dbscan_eps": eps, "dbscan_min_samples": min_samples,
                     "anomaly_top_fraction": config.TOP_FRACTION,
                     "rarefaction_depth": int(biodiversity.attrs["rarefaction_depth"])},
        "counts": {"dna_rows": len(dna), "dna_rows_without_occurrence": n_unmatched,
                   "asv_before_cleaning": len(asv_all), "asv": len(asv),
                   "stations": len(stations), "dbscan_noise": int((labels_db == NOISE).sum()),
                   "shared_points_for_comparison": n_shared},
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "kmeans_internal": {"silhouette": float(silhouette_score(Z, labels_km)),
                            "davies_bouldin": float(davies_bouldin_score(Z, labels_km)),
                            "calinski_harabasz": float(calinski_harabasz_score(Z, labels_km))},
        "clean_report": clean_report,
        "random_baseline_class_level": baseline,
    }
    say("Done")
    return {"asv": asv, "tax": tax, "stations": stations, "matrix": matrix, "X": X,
            "kmer_names": kmer_names, "Z": Z, "scaler": scaler, "pca": pca, "kmeans": kmeans,
            "iso_model": iso_model, "linkage_matrix": linkage_matrix, "k_table": k_table,
            "dbscan_grid": grid, "internal": internal, "external": external,
            "profiles": profiles, "candidates": candidates, "biodiversity": biodiversity,
            "read_share": read_share, "master": master, "metadata": metadata}


def save_results(results):
    """Write models to models/, intermediate data to data/processed/, tables to outputs/."""
    for folder in (config.MODELS_DIR, config.PROCESSED_DIR, config.OUTPUTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(results["scaler"], config.MODELS_DIR / "scaler.joblib")
    joblib.dump(results["pca"], config.MODELS_DIR / "pca.joblib")
    joblib.dump(results["kmeans"], config.MODELS_DIR / "kmeans.joblib")
    # compress=3 shrinks the 500-tree forest from about 6 MB to under 2 MB, with identical scores
    joblib.dump(results["iso_model"], config.MODELS_DIR / "isolation_forest.joblib", compress=3)

    processed = config.PROCESSED_DIR
    results["asv"].to_csv(processed / "asv_table.csv")
    results["tax"].to_csv(processed / "taxonomy.csv")
    results["stations"].to_csv(processed / "stations.csv")
    results["matrix"].to_csv(processed / "abundance_matrix.csv")
    np.save(processed / "kmer_features.npy", results["X"])
    np.save(processed / "pca_space.npy", results["Z"])
    np.save(processed / "ward_linkage.npy", results["linkage_matrix"])

    out = config.OUTPUTS_DIR
    master = results["master"]
    master.to_csv(out / "asv_master.csv")
    master[["length", "total_reads", "n_stations", "cluster_kmeans", "cluster_ward",
            "cluster_dbscan", "major_group", "tax_class", "genus", "scientific_name"]
           ].to_csv(out / "cluster_assignments.csv")
    shortlist = results["candidates"]
    shortlist[shortlist["if_flag"]].to_csv(out / "novelty_candidates.csv")
    results["biodiversity"].to_csv(out / "station_biodiversity.csv")
    results["profiles"].to_csv(out / "cluster_profiles.csv")
    results["read_share"].to_csv(out / "cluster_read_share.csv")
    results["internal"].to_csv(out / "internal_comparison.csv")
    results["external"].to_csv(out / "external_comparison.csv", index=False)
    results["dbscan_grid"].to_csv(out / "dbscan_grid.csv", index=False)
    if results["k_table"] is not None:
        results["k_table"].to_csv(out / "k_selection.csv")
    (out / "run_metadata.json").write_text(json.dumps(results["metadata"], indent=2))


def _sha256(path):
    """Fingerprint of a file, so we can tell whether the raw data ever changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(module_name):
    module = __import__(module_name)
    return getattr(module, "__version__", "unknown")