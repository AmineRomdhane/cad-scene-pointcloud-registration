#!/usr/bin/env python3
"""
Create a globally balanced correspondence dataset.

Input:
- results/learning_data/all_correspondences.csv

Output:
- results/learning_data/all_correspondences_balanced_global.csv
- results/learning_data/balanced_global_dataset_summary.csv

Method:
- Keep all samples from the minority class.
- Randomly undersample the majority class to the same size.
- Shuffle the final balanced dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = BASE_DIR / "results" / "learning_data" / "all_correspondences.csv"
OUTPUT_CSV = BASE_DIR / "results" / "learning_data" / "all_correspondences_balanced_global.csv"
SUMMARY_CSV = BASE_DIR / "results" / "learning_data" / "balanced_global_dataset_summary.csv"

TARGET_COL = "target_weight"
RANDOM_STATE = 42


KNOWN_SHAPES = [
    "boxes_cylinder",
    "box_sphere",
    "wall_column",
    "l_shape",
]


def infer_shape_name(sample_id):
    for shape in KNOWN_SHAPES:
        if str(sample_id).startswith(shape + "_"):
            return shape
    return "unknown"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if TARGET_COL not in df.columns:
        raise RuntimeError(f"Missing target column: {TARGET_COL}")

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    pos_df = df[df[TARGET_COL] == 1].copy()
    neg_df = df[df[TARGET_COL] == 0].copy()

    n_pos = len(pos_df)
    n_neg = len(neg_df)

    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"Cannot balance dataset because one class is missing: "
            f"n_pos={n_pos}, n_neg={n_neg}"
        )

    n_keep = min(n_pos, n_neg)

    pos_balanced = pos_df.sample(
        n=n_keep,
        replace=False,
        random_state=RANDOM_STATE,
    )

    neg_balanced = neg_df.sample(
        n=n_keep,
        replace=False,
        random_state=RANDOM_STATE,
    )

    balanced_df = pd.concat([pos_balanced, neg_balanced], axis=0)

    balanced_df = balanced_df.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    summary_rows = []

    summary_rows.append({
        "dataset": "original",
        "num_rows": len(df),
        "num_positive": n_pos,
        "num_negative": n_neg,
        "positive_rate": n_pos / len(df),
    })

    summary_rows.append({
        "dataset": "balanced_global",
        "num_rows": len(balanced_df),
        "num_positive": int((balanced_df[TARGET_COL] == 1).sum()),
        "num_negative": int((balanced_df[TARGET_COL] == 0).sum()),
        "positive_rate": float(balanced_df[TARGET_COL].mean()),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    # Optional detailed summaries
    balanced_df["shape_name"] = balanced_df["sample_id"].apply(infer_shape_name)

    by_shape = (
        balanced_df
        .groupby("shape_name")[TARGET_COL]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={
            "count": "num_rows",
            "sum": "num_positive",
            "mean": "positive_rate",
        })
    )

    by_shape["num_negative"] = by_shape["num_rows"] - by_shape["num_positive"]

    by_shape_path = BASE_DIR / "results" / "learning_data" / "balanced_global_by_shape_summary.csv"
    by_shape.to_csv(by_shape_path, index=False)

    by_scenario = (
        balanced_df
        .groupby("scenario")[TARGET_COL]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={
            "count": "num_rows",
            "sum": "num_positive",
            "mean": "positive_rate",
        })
    )

    by_scenario["num_negative"] = by_scenario["num_rows"] - by_scenario["num_positive"]

    by_scenario_path = BASE_DIR / "results" / "learning_data" / "balanced_global_by_scenario_summary.csv"
    by_scenario.to_csv(by_scenario_path, index=False)

    print("=" * 80)
    print("Balanced dataset created")
    print("=" * 80)
    print(f"Original rows:       {len(df)}")
    print(f"Original positives:  {n_pos}")
    print(f"Original negatives:  {n_neg}")
    print(f"Original pos rate:   {n_pos / len(df):.4f}")
    print("-" * 80)
    print(f"Balanced rows:       {len(balanced_df)}")
    print(f"Balanced positives:  {(balanced_df[TARGET_COL] == 1).sum()}")
    print(f"Balanced negatives:  {(balanced_df[TARGET_COL] == 0).sum()}")
    print(f"Balanced pos rate:   {balanced_df[TARGET_COL].mean():.4f}")
    print("-" * 80)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Saved: {by_shape_path}")
    print(f"Saved: {by_scenario_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
