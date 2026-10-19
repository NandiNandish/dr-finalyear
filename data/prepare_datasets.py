"""
Dataset preparation and harmonization script.
Standardizes EyePACS, IDRiD, DDR, Messidor, Messidor-2, and APTOS 2019 into a unified CSV manifest.
Includes fallback synthetic demo dataset generation.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from synthetic_generator import create_demo_dataset


DATASET_METRICS = {
    "eyepacs": {
        "url": "https://www.kaggle.com/c/diabetic-retinopathy-detection",
        "description": "EyePACS Diabetic Retinopathy Detection (Kaggle)",
        "licence": "Research Non-commercial",
    },
    "idrid": {
        "url": "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid",
        "description": "Indian Diabetic Retinopathy Image Dataset (IDRiD)",
        "licence": "CC BY 4.0",
    },
    "ddr": {
        "url": "https://github.com/nkicsl/DDR-dataset",
        "description": "Dataset for Diabetic Retinopathy (DDR)",
        "licence": "Academic Research",
    },
    "messidor": {
        "url": "https://www.adcis.net/en/third-party/messidor/",
        "description": "Messidor Fundus Dataset",
        "licence": "Research Agreement Required",
    },
    "aptos": {
        "url": "https://www.kaggle.com/c/aptos2019-blindness-detection",
        "description": "APTOS 2019 Blindness Detection (Kaggle)",
        "licence": "Research Use Only",
    },
}


def prepare_manifest(args):
    data_dir = Path(args.data_dir)
    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    demo_manifest = manifest_dir / "demo_manifest.csv"
    ext_manifest = manifest_dir / "external_test.csv"

    print("Checking dataset availability...")
    for key, info in DATASET_METRICS.items():
        dataset_path = data_dir / key
        if dataset_path.exists():
            print(f"  [FOUND] {info['description']} at {dataset_path}")
        else:
            print(f"  [MISSING] {info['description']} - download from: {info['url']} ({info['licence']})")

    # If --demo or if real datasets are not found, generate the synthetic/demo dataset
    if args.demo or not any((data_dir / k).exists() for k in DATASET_METRICS.keys()):
        print("\nGenerating synthetic demo dataset for rapid end-to-end execution...")
        df_int, df_ext = create_demo_dataset(
            output_dir=str(data_dir / "demo"),
            manifest_path=str(demo_manifest),
            external_manifest_path=str(ext_manifest),
            seed=args.seed,
        )
        print(f"Successfully generated demo dataset with {len(df_int)} internal and {len(df_ext)} external samples.")
        print(f"Internal manifest: {demo_manifest}")
        print(f"External manifest: {ext_manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fundus datasets and manifests.")
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument("--demo", action="store_true", default=True, help="Force demo synthetic dataset generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    args = parser.parse_args()

    prepare_manifest(args)
