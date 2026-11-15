"""Project-wide paths and constants. Every other module imports from here."""
from pathlib import Path

# config.py lives in <root>/src/edna/, so the project root is two folders up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DNA_FILE = RAW_DIR / "dnaderiveddata.txt"
OCCURRENCE_FILE = RAW_DIR / "occurrence.txt"

RANDOM_SEED = 42  # one seed everywhere so results are reproducible