#!/usr/bin/env python3
"""
Organize all result tables and figures by experiment/test name.

Input folders:
- results/tables/
- results/figures/

Output folder:
- results/by_test/<test_name>/

Example:
random_forest_results_average.csv
random_forest_confusion_matrix_reduced_wall_column.png
random_forest_confusion_matrix_reduced_l_shape.png

all go to:
results/by_test/random_forest/

Default is dry-run.
Use --apply to copy files.
"""

from pathlib import Path
import argparse
import csv
import re
import shutil
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[1]

SOURCE_ROOTS = [
    BASE_DIR / "results" / "tables",
    BASE_DIR / "results" / "figures",
]

DEST_ROOT = BASE_DIR / "results" / "by_test"
MANIFEST_PATH = DEST_ROOT / "organization_manifest_by_test.csv"

IGNORE_NAMES = {
    "organization_manifest.csv",
    "organization_by_test_manifest.csv",
    "organization_manifest_by_test.csv",
}


def clean_folder_name(name):
    name = name.strip()
    name = name.replace(" ", "_")
    name = re.sub(r"^\d+[_-]+", "", name)   # remove leading numbers like 01_
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("_")

    if not name:
        name = "misc"

    return name


def test_name_from_file(path):
    stem = path.stem

    # Very specific tests first
    known_prefixes = [
        "mlp_real_only_v3_soft_labels",
        "mlp_real_only_v2_soft_labels",
        "mlp_real_only_holdout_v2",
        "mlp_real_holdout_clean_v3",
        "mlp_real_holdout_clean_v2",
        "mlp_real_holdout_clean",
        "mlp_synthetic_plus_real_clean",
        "mlp_balanced_undersampling",
        "mlp_balanced_dataset",
        "mlp_no_pos_weight",
        "mlp_threshold",
        "final_mlp_v3",
        "final_mlp_v2",
        "decision_tree_real_only_v2_case_balanced",
        "decision_tree_real_only_v2",
        "decision_tree",
        "random_forest",
        "audit_real_feature_labeling_v2",
        "audit_real_transform_direction_v2",
        "audit_real_config_v2",
        "audit_real",
        "feature_correlation",
        "feature_target_correlation",
        "highly_correlated_feature_pairs",
        "balanced_global_dataset",
        "balanced_global",
        "curated_real_samples",
        "removed_bad_real_cases_v3",
        "removed_bad_real_cases_v2",
        "removed_mir_cases_from_v2",
        "real_rows_selected_for_learning",
        "dataset_summary",
        "icp",
        "real_registration",
    ]

    for prefix in known_prefixes:
        if stem.startswith(prefix):
            if prefix == "highly_correlated_feature_pairs":
                return "feature_correlation"
            if prefix == "feature_target_correlation":
                return "feature_correlation"
            if prefix == "balanced_global":
                return "balanced_global_dataset"
            return clean_folder_name(prefix)

    # Generic MLP baseline files
    if stem.startswith("mlp_results") or stem.startswith("mlp_confusion_matrix") or stem.startswith("mlp_loss_curve") or stem.startswith("mlp_training_history"):
        return "mlp_baseline"

    # Generic fallback:
    # remove common output suffixes but do not keep shape names as folders
    shape_suffixes = [
        "_box_sphere",
        "_l_shape",
        "_boxes_cylinder",
        "_wall_column",
    ]

    name = stem

    for suffix in shape_suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    suffixes_to_remove = [
        "_results_average",
        "_results_by_case",
        "_results_by_shape",
        "_results_by_run",
        "_threshold_results_average",
        "_threshold_results_by_case",
        "_threshold_results_by_shape",
        "_feature_importance_average",
        "_feature_importance_by_case",
        "_feature_importance_by_run",
        "_permutation_importance_average",
        "_permutation_importance_by_case",
        "_permutation_importance_by_run",
        "_all_real_feature_importance",
        "_rules_all_real",
        "_training_history",
        "_train_sampling_counts",
        "_confusion_matrix",
        "_loss_curve",
        "_summary",
        "_config",
    ]

    changed = True
    while changed:
        changed = False
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break

    return clean_folder_name(name)


def is_temp_or_hidden_file(path):
    name = path.name

    # LibreOffice lock files, e.g. .~lock.file.csv#
    if name.startswith(".~lock."):
        return True

    # macOS/resource-fork style hidden files or accidental hidden temp files
    if name.startswith("._"):
        return True

    # General hidden files
    if name.startswith("."):
        return True

    # LibreOffice lock files usually end with #
    if name.endswith("#"):
        return True

    return False


def collect_files():
    files = []

    for root in SOURCE_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.name in IGNORE_NAMES:
                continue

            if is_temp_or_hidden_file(path):
                continue

            files.append(path)

    return sorted(files)


def safe_destination(dest):
    if not dest.exists():
        return dest

    parent = dest.parent
    stem = dest.stem
    suffix = dest.suffix

    i = 1
    while True:
        candidate = parent / f"{stem}_copy{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def remove_empty_dirs():
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                    print(f"[EMPTY REMOVED] {path.relative_to(BASE_DIR)}")
                except OSError:
                    pass


def write_manifest(rows):
    if not rows:
        return

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    file_exists = MANIFEST_PATH.exists()

    with MANIFEST_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "apply",
                "test_name",
                "source",
                "destination",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this, only a dry run is printed.",
    )

    parser.add_argument(
        "--remove-empty-dirs",
        action="store_true",
        help="Remove empty old folders after moving.",
    )

    args = parser.parse_args()

    mode = "APPLY" if args.apply else "DRY RUN"

    print("=" * 100)
    print("Organize results by test")
    print("=" * 100)
    print(f"Mode: {mode}")
    print(f"Destination root: {DEST_ROOT.relative_to(BASE_DIR)}")
    print("=" * 100)

    rows = []

    for src in collect_files():
        test_name = test_name_from_file(src)
        dest_dir = DEST_ROOT / test_name
        dest = safe_destination(dest_dir / src.name)

        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "apply": args.apply,
            "test_name": test_name,
            "source": str(src.relative_to(BASE_DIR)),
            "destination": str(dest.relative_to(BASE_DIR)),
        })

        tag = "[COPY]" if args.apply else "[DRY] "
        print(f"{tag} {src.relative_to(BASE_DIR)} -> {dest.relative_to(BASE_DIR)}")

        if args.apply:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))

    write_manifest(rows)

    if args.apply and args.remove_empty_dirs:
        remove_empty_dirs()

    print("=" * 100)
    print(f"Files processed: {len(rows)}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(BASE_DIR)}")
    print("=" * 100)

    if not args.apply:
        print("Dry run only. To actually move files, run:")
        print("python3 src/organize_results_by_test.py --apply --remove-empty-dirs")


if __name__ == "__main__":
    main()
