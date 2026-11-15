"""Turn the linked table into clean ASV and station tables plus an abundance matrix."""
import pandas as pd


def build_stations(linked):
    """One row per sampling station, ordered south -> north and labelled S01, S02, ..."""
    stations = (
        linked.groupby("eventID")
        .agg(lat=("decimalLatitude", "mean"),
             lon=("decimalLongitude", "mean"),
             date=("eventDate", "min"))
        .sort_values("lat")
        .reset_index()
    )
    stations.index = [f"S{i + 1:02d}" for i in range(len(stations))]
    stations.index.name = "station"
    return stations


def build_asv_table(linked):
    """Collapse the linked table to one row per unique DNA sequence (ASV).

    Columns: DNA_sequence, total_reads, n_stations, length. Index: ASV_001, ASV_002, ...
    ASVs are numbered from the most to the least abundant.
    """
    asv = (
        linked.groupby("DNA_sequence")
        .agg(total_reads=("organismQuantity", "sum"),
             n_stations=("eventID", "nunique"))
        .sort_values("total_reads", ascending=False, kind="stable")
        .reset_index()
    )
    asv["asv_id"] = [f"ASV_{i + 1:03d}" for i in range(len(asv))]
    asv["length"] = asv["DNA_sequence"].str.len()
    return asv.set_index("asv_id")


def clean_asvs(asv, window=30):
    """Drop sequences with invalid bases or a length far from the median.

    A sequence is kept if it contains only A/C/G/T and its length is within
    `window` base pairs of the median length. Returns (clean_asv, report).
    """
    median = asv["length"].median()
    valid_length = asv["length"].between(median - window, median + window)
    valid_bases = asv["DNA_sequence"].str.fullmatch(r"[ACGT]+")
    keep = valid_length & valid_bases
    report = {
        "median_length": float(median),
        "removed_by_length": int((~valid_length).sum()),
        "removed_by_invalid_bases": int((~valid_bases).sum()),
        "kept": int(keep.sum()),
    }
    return asv[keep].copy(), report


def build_abundance_matrix(linked, asv, stations):
    """ASV x station matrix of read counts (rows = ASVs, columns = stations)."""
    event_to_station = dict(zip(stations["eventID"], stations.index))
    data = linked.assign(station=linked["eventID"].map(event_to_station))
    # Inner merge keeps only ASVs that survived cleaning and adds their asv_id.
    data = data.merge(asv[["DNA_sequence"]].reset_index(), on="DNA_sequence")
    matrix = data.pivot_table(
        index="asv_id", columns="station", values="organismQuantity",
        aggfunc="sum", fill_value=0,
    )
    return matrix.loc[asv.index, stations.index]