"""Execute Jupyter notebooks 03 to 08 in-place with outputs and figures captured."""
import sys
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

def execute_notebook(nb_path: Path):
    print(f"Executing {nb_path.name}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[OK] {nb_path.name} executed and saved with outputs.")

def main():
    notebooks_dir = Path("notebooks")
    target_notebooks = [
        "03_dataset_design.ipynb",
        "04_feature_engineering.ipynb",
        "05_eda.ipynb",
        "06_baseline_modeling.ipynb",
        "07_advanced_modeling.ipynb",
        "08_evaluation.ipynb",
    ]

    for nb_name in target_notebooks:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            try:
                execute_notebook(nb_path)
            except Exception as e:
                print(f"[ERROR] Failed executing {nb_name}: {e}")

if __name__ == "__main__":
    main()
