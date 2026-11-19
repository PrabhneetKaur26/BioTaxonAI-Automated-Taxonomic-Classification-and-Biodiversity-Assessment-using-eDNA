"""Cached access to the saved pipeline results (loaded once, not on every click)."""
import hashlib

import streamlit as st

from src.edna import config, store


def _data_version():
    """Changes whenever the pipeline is rerun, so the cache refreshes automatically."""
    marker = config.OUTPUTS_DIR / "run_metadata.json"
    return marker.stat().st_mtime_ns if marker.exists() else 0


@st.cache_data(show_spinner="Loading analysis results...")
def _load_tables(version):
    return store.load_tables()


@st.cache_resource(show_spinner="Loading trained models...")
def _load_models(version):
    return store.load_models()


def get_tables():
    """All saved result tables (cached)."""
    return _load_tables(_data_version())


def get_models():
    """Fitted scaler, PCA, K-Means and Isolation Forest (cached)."""
    return _load_models(_data_version())


def dataset_fingerprint(metadata):
    """One SHA-256 that summarises both raw data files (digest of their individual hashes)."""
    hashes = metadata["raw_data_sha256"]
    joined = "".join(hashes[name] for name in sorted(hashes))
    return hashlib.sha256(joined.encode()).hexdigest()