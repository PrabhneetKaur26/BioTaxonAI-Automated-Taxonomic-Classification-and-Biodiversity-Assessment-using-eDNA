"""Unusual-sequence detection (Isolation Forest primary, LOF second opinion).

Terminology: an anomaly is a sequence whose k-mer profile is unusual compared with the
rest of the sample. That is NOT proof of a new species. Results are "potential novelty
candidates" that need external validation (BLAST, phylogenetic placement, a second marker).
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.edna import config

DISPLAY_TAXONOMY = ["major_group", "scientific_name", "rank", "poorly_resolved"]


def score_anomalies(Z, index, seed=config.RANDOM_SEED, n_estimators=500, lof_neighbors=20):
    """Score every sequence. Higher score = more unusual.

    Returns (scores_table, fitted_isolation_forest). Rank 1 = most unusual sequence.
    """
    iso = IsolationForest(n_estimators=n_estimators, random_state=seed).fit(Z)
    lof = LocalOutlierFactor(n_neighbors=lof_neighbors).fit(Z)
    scores = pd.DataFrame({"if_score": -iso.score_samples(Z),
                           "lof_score": -lof.negative_outlier_factor_}, index=index)
    scores["if_rank"] = scores["if_score"].rank(ascending=False, method="first").astype(int)
    scores["lof_rank"] = scores["lof_score"].rank(ascending=False, method="first").astype(int)
    return scores, iso


def build_candidate_table(scores, asv, tax, top_fraction=0.08, min_stations=3, min_reads=50):
    """Combine scores with sequence info into one ranked review table.

    top_fraction is a REVIEW BUDGET (how many of the most unusual sequences to inspect),
    not an estimate of how many novel organisms exist. Taxonomy columns are included for
    interpretation only and play no part in the scoring.
    """
    n_top = int(round(top_fraction * len(scores)))
    table = (scores.join(asv[["DNA_sequence", "length", "total_reads", "n_stations"]])
                   .join(tax[DISPLAY_TAXONOMY]))
    table["if_flag"] = table["if_rank"] <= n_top
    table["lof_flag"] = table["lof_rank"] <= n_top
    table["agreement"] = np.select(
        [table["if_flag"] & table["lof_flag"], table["if_flag"], table["lof_flag"]],
        ["Both detectors", "Isolation Forest only", "LOF only"], default="Neither")
    table["robust"] = (table["n_stations"] >= min_stations) & (table["total_reads"] >= min_reads)
    table["review_status"] = np.select(
        [table["if_flag"] & table["robust"], table["if_flag"]],
        ["Priority candidate", "Low support (possible artefact)"], default="Not flagged")
    return table.sort_values("if_rank")


def ranking_stability(Z, index, seeds=(0, 1, 7, 123), reference_seed=config.RANDOM_SEED,
                      top_n=40):
    """How much does the Isolation Forest ranking change with the random seed?"""
    ref = score_anomalies(Z, index, seed=reference_seed)[0]["if_score"]
    ref_top = set(ref.sort_values(ascending=False).index[:top_n])
    rhos, overlaps = [], []
    for seed in seeds:
        other = score_anomalies(Z, index, seed=seed)[0]["if_score"]
        rhos.append(spearmanr(ref, other)[0])
        overlaps.append(len(ref_top & set(other.sort_values(ascending=False).index[:top_n])))
    return float(np.mean(rhos)), float(np.mean(overlaps))


def posthoc_enrichment(flag, poorly_resolved):
    """POST-HOC sanity check: are flagged ASVs enriched for poorly resolved taxonomy?

    Uses Fisher's exact test. This validates nothing about novelty by itself; it only asks
    whether the detector points at sequences the reference database also finds hard.
    """
    flag, pr = np.asarray(flag, bool), np.asarray(poorly_resolved, bool)
    a, b = int((flag & pr).sum()), int((flag & ~pr).sum())
    c, d = int((~flag & pr).sum()), int((~flag & ~pr).sum())
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
    return {"flagged": int(flag.sum()), "share_poorly_resolved_flagged": a / max(flag.sum(), 1),
            "share_poorly_resolved_overall": float(pr.mean()),
            "odds_ratio": float(odds_ratio), "p_value": float(p_value)}