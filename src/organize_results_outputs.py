#!/usr/bin/env python3
"""
Organize results/tables and results/figures into subfolders.

Default mode is dry-run.
Use --apply to actually move files.

It creates a manifest:
results/organization_manifest.csv
"""

from pathlib import Path
import argparse
import csv
import shutil
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[1]

TABLE_DIR = BASE_DIR / "results" / "tables"
FIGURE_DIR = BASE_DIR / "results" / "figures"
MANIFEST_PATH = BASE_DIR / "results" / "organization_manifest.csv"


TABLE_CATEGORIES = {
    "00_dataset_and_configs": [
        "dataset_summary",
        "dataset_samples",
        "balanced_global",
        "curated",
        "removed",
        "real_rows_selected",
    ],
    "01_feature_analysis": [
        "feature_correlation",
        "feature_target",
        "highly_correlated",
        "correlation",
    ],
    "02_decision_tree": [
        "decision_tree",
    ],
    "03_random_forest": [
        "random_forest",
    ],
    "04_mlp": [
        "mlp",
        "final_mlp",
    ],
    "05_audit": [
        "audit",
    ],
    "06_registration": [
        "icp",
        "registration",
        "real_registration",
    ],
    "99_misc": [],
}


FIGURE_CATEGORIES = {
    "01_feature_analysis": [
        "correlation",
        "feature",
    ],
    "02_decision_tree": [
        "decision_tree",
        "tree",
    ],
    "03_random_forest": [
        "random_forest",
        "forest",
    ],
    "04_mlp": [
        "mlp",
        "loss_curve",
        "confusion_matrix",
        "threshold",
    ],
    "06_registration": [
        "icp",
        "registration",
        "alignment",
    ],
    "99_misc": [],
}


def choose_category(filename, categories):
    name = filename.lower()

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in name:
                return category

    return "99_misc"


def safe_destination(dest):
    """
    Avoid overwriting files.
    If dest exists, append _copyN before suffix.
    """
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent

    i = 1
    while True:
        candidate = parent / f"{stem}_copy{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def collect_files(folder):
    if not folder.exists():
        return []

    files = []

    for path in folder.iterdir():
        if path.is_file():
            files.append(path)

    return sorted(files)


def organize_folder(folder, categories, apply):
    rows = []

    files = collect_files(folder)

    for src in files:
        category = choose_category(src.name, categories)
        dest_dir = folder / category
        dest = safe_destination(dest_dir / src.name)

        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "apply": apply,
            "type": folder.name,
            "category": category,
            "source": str(src.relative_to(BASE_DIR)),
            "destination": str(dest.relative_to(BASE_DIR)),
        })

        print(f"{'[MOVE]' if apply else '[DRY] '} {src.relative_to(BASE_DIR)} -> {dest.relative_to(BASE_DIR)}")

        if apply:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))

    return rows


def write_manifest(rows, append=True):
    if not rows:
        return

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = MANIFEST_PATH.exists()

    mode = "a" if append else "w"

    with MANIFEST_PATH.open(mode, newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "apply",
                "type",
                "category",
                "source",
                "destination",
            ],
        )

        if not file_exists or not append:
            writer.writeheader()

        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this flag, only prints a dry run.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("Organizing result outputs")
    print("=" * 100)
    print(f"Base directory: {BASE_DIR}")
    print(f"Mode: {'APPLY / MOVE FILES' if args.apply else 'DRY RUN ONLY'}")
    print("=" * 100)

    all_rows = []

    all_rows.extend(
        organize_folder(
            folder=TABLE_DIR,
            categories=TABLE_CATEGORIES,
            apply=args.apply,
        )
    )

    all_rows.extend(
        organize_folder(
            folder=FIGURE_DIR,
            categories=FIGURE_CATEGORIES,
            apply=args.apply,
        )
    )

    write_manifest(all_rows)

    print("=" * 100)
    print(f"Files processed: {len(all_rows)}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(BASE_DIR)}")
    print("=" * 100)

    if not args.apply:
        print("Dry run finished. To actually move files, run:")
        print("python3 src/organize_results_outputs.py --apply")


if __name__ == "__main__":
    main()
