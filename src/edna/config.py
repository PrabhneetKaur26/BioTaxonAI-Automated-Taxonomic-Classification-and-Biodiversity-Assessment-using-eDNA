"""Project-wide paths and settings. Every other module imports from here."""
from pathlib import Path

# config.py lives in <root>/src/edna/, so the project root is two folders up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DNA_FILE = RAW_DIR / "dnaderiveddata.txt"
OCCURRENCE_FILE = RAW_DIR / "occurrence.txt"

# ---- Analysis settings (change them here, nowhere else) ----
RANDOM_SEED = 42        # one seed everywhere so results are reproducible
KMER_SIZE = 4           # 4**4 = 256 k-mer features per sequence
N_COMPONENTS = 20       # PCA components kept (about 70% of the variance)
MIN_STABILITY = 0.85    # k must reach this subsampling stability to be eligible
TOP_FRACTION = 0.08     # anomaly REVIEW BUDGET: inspect the 8% most unusual ASVs