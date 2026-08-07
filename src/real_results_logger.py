import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


COLUMNS = [
    "timestamp",
    "run_name",
    "stage",
    "cad_file",
    "target_file",
    "init_transform",
    "output_folder",
    "voxel",
    "ransac_fitness",
    "ransac_rmse",
    "ransac_time_s",
    "icp_fitness",
    "icp_rmse",
    "icp_time_s",
    "final_eval_fitness",
    "final_eval_rmse",
    "final_eval_correspondences",
    "translation_error_m",
    "rotation_error_deg",
    "matrix_file",
    "registered_pcd_file",
    "combined_pcd_file",
]


def project_root():
    return Path(__file__).resolve().parents[1]


def safe_name(text):
    text = str(text)
    keep = []
    for c in text:
        if c.isalnum() or c in ["_", "-", "."]:
            keep.append(c)
        else:
            keep.append("_")
    return "".join(keep)


def relpath(path, root):
    if path is None:
        return ""

    p = Path(path)

    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


def copy_if_exists(src_path, dst_dir, prefix):
    if src_path is None:
        return ""

    src = Path(src_path)

    if not src.exists():
        return ""

    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / f"{prefix}__{src.name}"
    shutil.copy2(src, dst)

    return str(dst)


def detect_summary(result_dir):
    result_dir = Path(result_dir)

    registration_summary = result_dir / "registration_summary.json"
    refinement_summary = result_dir / "refinement_summary.json"

    if refinement_summary.exists():
        return "refinement", refinement_summary

    if registration_summary.exists():
        return "registration", registration_summary

    raise RuntimeError(
        f"No registration_summary.json or refinement_summary.json found in: {result_dir}"
    )


def build_row_from_registration(data, result_dir, timestamp, run_id, root):
    result_dir = Path(result_dir)

    matrix_src = result_dir / "T_scene_cad.txt"
    registered_src = result_dir / "cad_registered.pcd"
    combined_src = result_dir / "combined_target_and_registered_cad.pcd"

    matrices_dir = root / "real_results" / "matrices"
    pointclouds_dir = root / "real_results" / "pointclouds"

    matrix_dst = copy_if_exists(matrix_src, matrices_dir, run_id)
    registered_dst = copy_if_exists(registered_src, pointclouds_dir, run_id)
    combined_dst = copy_if_exists(combined_src, pointclouds_dir, run_id)

    ransac = data.get("ransac", {})
    icp = data.get("icp", {})
    final_eval = data.get("final_evaluation", {})

    row = {
        "timestamp": timestamp,
        "run_name": run_id,
        "stage": "ransac_icp",
        "cad_file": data.get("cad_file", ""),
        "target_file": data.get("target_file", ""),
        "init_transform": "",
        "output_folder": str(result_dir),
        "voxel": data.get("voxel", ""),
        "ransac_fitness": ransac.get("fitness", ""),
        "ransac_rmse": ransac.get("inlier_rmse", ""),
        "ransac_time_s": ransac.get("time_s", ""),
        "icp_fitness": icp.get("fitness", ""),
        "icp_rmse": icp.get("inlier_rmse", ""),
        "icp_time_s": icp.get("time_s", ""),
        "final_eval_fitness": final_eval.get("fitness", ""),
        "final_eval_rmse": final_eval.get("inlier_rmse", ""),
        "final_eval_correspondences": final_eval.get("correspondences", ""),
        "translation_error_m": "",
        "rotation_error_deg": "",
        "matrix_file": relpath(matrix_dst, root),
        "registered_pcd_file": relpath(registered_dst, root),
        "combined_pcd_file": relpath(combined_dst, root),
    }

    return row


def build_row_from_refinement(data, result_dir, timestamp, run_id, root):
    result_dir = Path(result_dir)

    matrix_src = result_dir / "T_scene_cad_refined.txt"
    registered_src = result_dir / "cad_registered_refined.pcd"
    combined_src = result_dir / "combined_target_and_registered_cad_refined.pcd"

    matrices_dir = root / "real_results" / "matrices"
    pointclouds_dir = root / "real_results" / "pointclouds"

    matrix_dst = copy_if_exists(matrix_src, matrices_dir, run_id)
    registered_dst = copy_if_exists(registered_src, pointclouds_dir, run_id)
    combined_dst = copy_if_exists(combined_src, pointclouds_dir, run_id)

    row = {
        "timestamp": timestamp,
        "run_name": run_id,
        "stage": "icp_refinement",
        "cad_file": data.get("cad", ""),
        "target_file": data.get("target", ""),
        "init_transform": data.get("init", ""),
        "output_folder": str(result_dir),
        "voxel": data.get("voxel", ""),
        "ransac_fitness": "",
        "ransac_rmse": "",
        "ransac_time_s": "",
        "icp_fitness": data.get("icp_fitness", ""),
        "icp_rmse": data.get("icp_rmse", ""),
        "icp_time_s": "",
        "final_eval_fitness": data.get("icp_fitness", ""),
        "final_eval_rmse": data.get("icp_rmse", ""),
        "final_eval_correspondences": "",
        "translation_error_m": "",
        "rotation_error_deg": "",
        "matrix_file": relpath(matrix_dst, root),
        "registered_pcd_file": relpath(registered_dst, root),
        "combined_pcd_file": relpath(combined_dst, root),
    }

    return row


def append_csv(row, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def create_markdown_table(csv_path, md_path):
    if not csv_path.exists():
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    selected_columns = [
        "timestamp",
        "run_name",
        "stage",
        "voxel",
        "ransac_fitness",
        "icp_fitness",
        "icp_rmse",
        "matrix_file",
    ]

    lines = []
    lines.append("| " + " | ".join(selected_columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(selected_columns)) + " |")

    for row in rows:
        values = []
        for col in selected_columns:
            value = str(row.get(col, ""))
            value = value.replace("|", "/")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")

    md_path.write_text("\n".join(lines))


def log_result_folder(result_dir):
    root = project_root()
    result_dir = Path(result_dir)

    if not result_dir.is_absolute():
        result_dir = root / result_dir

    result_type, summary_path = detect_summary(result_dir)

    with open(summary_path, "r") as f:
        data = json.load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_for_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_id = f"{timestamp_for_file}_{safe_name(result_dir.name)}"

    if result_type == "registration":
        row = build_row_from_registration(
            data=data,
            result_dir=result_dir,
            timestamp=timestamp,
            run_id=run_id,
            root=root,
        )
    elif result_type == "refinement":
        row = build_row_from_refinement(
            data=data,
            result_dir=result_dir,
            timestamp=timestamp,
            run_id=run_id,
            root=root,
        )
    else:
        raise RuntimeError(f"Unknown result type: {result_type}")

    csv_path = root / "real_results" / "tables" / "real_registration_results.csv"
    md_path = root / "real_results" / "tables" / "real_registration_results.md"

    append_csv(row, csv_path)
    create_markdown_table(csv_path, md_path)

    print("[OK] Real registration result logged.")
    print(f"[OK] CSV table: {csv_path}")
    print(f"[OK] Markdown table: {md_path}")
    print(f"[OK] Run name: {run_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Log a real CAD registration result into real_results."
    )

    parser.add_argument(
        "--result_dir",
        required=True,
        help="Registration output folder containing registration_summary.json or refinement_summary.json",
    )

    args = parser.parse_args()

    log_result_folder(args.result_dir)


if __name__ == "__main__":
    main()
