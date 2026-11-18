"""Biodiversity metrics per sampling station, corrected for uneven sequencing depth.

Stations were sequenced to very different depths (3,685 to 86,010 reads). Deeper
sequencing finds more rare ASVs, so raw richness mostly measures depth. We therefore
compare stations at the SAME depth (rarefaction) alongside the raw numbers.
"""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import spearmanr

from src.edna import config


def shannon_index(counts):
    """Shannon diversity H = -sum(p * ln p) over ASVs with at least one read."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _log_comb(n, k):
    """log of the binomial coefficient C(n, k), stable for large numbers."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def expected_richness(counts, depth):
    """Exact expected number of ASVs seen if only `depth` reads were drawn (no randomness).

    An ASV with n reads out of N is missed with probability C(N-n, depth) / C(N, depth),
    so expected richness = sum over ASVs of (1 - that probability).
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    total = counts.sum()
    if depth > total:
        raise ValueError(f"Depth {depth} exceeds the {int(total)} reads available.")
    remaining = total - counts
    missed = np.zeros_like(counts)
    possible = remaining >= depth  # if fewer reads remain than the depth, it cannot be missed
    missed[possible] = np.exp(_log_comb(remaining[possible], depth) - _log_comb(total, depth))
    return float((1.0 - missed).sum())


def rarefied_shannon(counts, depth, rng, repeats=100):
    """Mean and sd of Shannon diversity over random subsamples of `depth` reads."""
    counts = np.asarray(counts, dtype=np.int64)
    values = [shannon_index(rng.multivariate_hypergeometric(counts, depth))
              for _ in range(repeats)]
    return float(np.mean(values)), float(np.std(values))


def rarefaction_curve(counts, depths):
    """Expected ASV richness at each depth: a flattening curve means the station is well sampled."""
    return np.array([expected_richness(counts, d) for d in depths])


def station_biodiversity(matrix, stations, depth=None, repeats=100, seed=config.RANDOM_SEED):
    """Per-station table of raw and depth-standardised diversity.

    matrix   : ASV x station read counts
    stations : station table (index S01.., column 'lat')
    depth    : common depth for comparison; default = the smallest station library
    """
    depth = int(matrix.sum(axis=0).min()) if depth is None else int(depth)
    rng = np.random.default_rng(seed)
    rows = []
    for station in matrix.columns:
        counts = matrix[station].to_numpy()
        sh_mean, sh_sd = rarefied_shannon(counts, depth, rng, repeats)
        rows.append({
            "station": station, "lat": stations.loc[station, "lat"],
            "reads": int(counts.sum()),
            "observed_richness": int((counts > 0).sum()),
            "rarefied_richness": expected_richness(counts, depth),
            "shannon": shannon_index(counts),
            "rarefied_shannon": sh_mean, "rarefied_shannon_sd": sh_sd,
        })
    table = pd.DataFrame(rows).set_index("station")
    table.attrs["rarefaction_depth"] = depth
    return table


def depth_dependence(table):
    """Spearman correlation of sequencing depth with each richness measure."""
    out = {}
    for column in ("observed_richness", "rarefied_richness", "shannon", "rarefied_shannon"):
        rho, p = spearmanr(table["reads"], table[column])
        out[column] = (float(rho), float(p))
    return out


def latitude_trends(table):
    """Spearman trend of each diversity measure with latitude (descriptive, only 20 stations)."""
    rows = []
    for column in ("observed_richness", "rarefied_richness", "shannon", "rarefied_shannon"):
        rho, p = spearmanr(table["lat"], table[column])
        rows.append({"measure": column, "spearman_rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows).set_index("measure")


def cluster_read_share(matrix, labels):
    """Share of each station's reads that belongs to each cluster (columns sum to 1)."""
    by_cluster = matrix.groupby(np.asarray(labels)).sum()
    return by_cluster.div(by_cluster.sum(axis=0), axis=1)