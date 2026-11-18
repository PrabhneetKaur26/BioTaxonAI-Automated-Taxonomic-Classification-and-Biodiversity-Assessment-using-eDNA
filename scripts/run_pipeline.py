"""Run the full analysis and save models and tables.

    python -m scripts.run_pipeline          # full run (chooses k, a few minutes)
    python -m scripts.run_pipeline --k 9    # skip the slow k search, use k = 9
"""
import argparse

from src.edna.pipeline import run_pipeline, save_results

parser = argparse.ArgumentParser(description="Run the eDNA analysis pipeline.")
parser.add_argument("--k", type=int, default=None,
                    help="number of K-Means clusters; omit to choose k automatically")
args = parser.parse_args()

results = run_pipeline(k=args.k)
save_results(results)
print("\nSaved models to models/, intermediate data to data/processed/, tables to outputs/")