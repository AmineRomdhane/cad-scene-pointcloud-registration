#!/usr/bin/env python3
"""
Audit real-case feature extraction and labeling for the v2 dataset.

This script checks:
1. Real config rows.
2. Path existence.
3. Label transform direction.
4. Whether target_weight matches label_distance < label_threshold_m.
5. Whether distance_T0 respects max_corr_distance_m.
6. Feature statistics.
7. Whether the raw transform or inverse transform better aligns observation to reference.

Dataset:
- configs/dataset_samples_synthetic_plus_real_curated_clean_v2.csv
- results/learning_data_synthetic_plus_real_curated_clean_v2/

Outputs:
- results/tables/audit_real_config_v2.csv
- results/tables/audit_real_feature_labeling_v2_by_sample.csv
- results/tables/audit_real_feature_labeling_v2_by_case.csv
- results/tables/audit_real_transform_direction_v2.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

try:
    import open3d as o3d
    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = BASE_DIR / "configs" / "dataset_samples_synthetic_plus_real_curated_clean_v2.csv"
DATA_DIR = BASE_DIR / "results" / "learning_data_synthetic_plus_real_curated_clean_v2"
SUMMARY_PATH = DATA_DIR / "dataset_summary.csv"
TABLE_DIR = BASE_DIR / "results" / "tables"

FEATURES = [
    "distance_T0",
    "point_to_plane_residual",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]

REQUIRED_CORR_COLS = [
    "sample_id",
    "scenario",
    "source_index",
    "target_index",
    "label_distance",
    "target_weight",
] + FEATURES


def resolve_path(path_value):
    p = Path(str(path_value))

    if p.is_absolute():
        return p

    return BASE_DIR / p


def read_transform(path):
    T = np.loadtxt(path)

    if T.shape != (4, 4):
        raise RuntimeError(f"Transform file is not 4x4: {path}")

    return T


def transform_points(points, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


def get_real_case_id(sample_id):
    sample_id = str(sample_id)

    if sample_id.startswith("real_"):
        case_id = sample_id[len("real_"):]
    else:
        case_id = sample_id

    for suffix in ["_easy", "_medium", "_hard"]:
        if case_id.endswith(suffix):
            case_id = case_id[:-len(suffix)]
            break

    return case_id


def load_cloud_points(path, voxel_size):
    pcd = o3d.io.read_point_cloud(str(path))

    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {path}")

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    points = np.asarray(pcd.points, dtype=np.float64)

    if len(points) == 0:
        raise RuntimeError(f"No points after downsampling: {path}")

    return points


def nearest_distances_to_reference(reference_points, query_points):
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(reference_points)

    kdtree = o3d.geometry.KDTreeFlann(ref_pcd)

    distances = np.empty(len(query_points), dtype=np.float64)

    for i, p in enumerate(query_points):
        _, _, dist2 = kdtree.search_knn_vector_3d(p, 1)
        distances[i] = np.sqrt(dist2[0])

    return distances


def summarize_distances(distances, label_threshold):
    return {
        "nn_mean": float(np.mean(distances)),
        "nn_median": float(np.median(distances)),
        "nn_p90": float(np.quantile(distances, 0.90)),
        "nn_p95": float(np.quantile(distances, 0.95)),
        "nn_inlier_rate_label_threshold": float(np.mean(distances < label_threshold)),
        "nn_inlier_rate_005": float(np.mean(distances < 0.05)),
        "nn_inlier_rate_010": float(np.mean(distances < 0.10)),
        "nn_inlier_rate_020": float(np.mean(distances < 0.20)),
    }


def audit_config(config_df):
    rows = []

    real_df = config_df[config_df["sample_id"].astype(str).str.startswith("real_")].copy()

    for _, row in real_df.iterrows():
        sample_id = str(row["sample_id"])
        real_case_id = get_real_case_id(sample_id)

        reference_path = resolve_path(row["reference_path"])
        observation_path = resolve_path(row["observation_path"])
        matrix_path = resolve_path(row["label_transform_path"])

        rows.append({
            "sample_id": sample_id,
            "real_case_id": real_case_id,
            "reference_path": str(row["reference_path"]),
            "observation_path": str(row["observation_path"]),
            "label_transform_path": str(row["label_transform_path"]),
            "label_transform_direction": row["label_transform_direction"],
            "T0_mode": row["T0_mode"],
            "perturb_rot_deg": row["perturb_rot_deg"],
            "perturb_trans_m": row["perturb_trans_m"],
            "voxel_size": row["voxel_size"],
            "normal_radius": row["normal_radius"],
            "fpfh_radius": row["fpfh_radius"],
            "density_radius": row["density_radius"],
            "max_corr_distance_m": row["max_corr_distance_m"],
            "label_threshold_m": row["label_threshold_m"],
            "max_source_points": row["max_source_points"],
            "reference_exists": reference_path.exists(),
            "observation_exists": observation_path.exists(),
            "matrix_exists": matrix_path.exists(),
        })

    return pd.DataFrame(rows)


def audit_correspondence_csvs(config_df):
    sample_rows = []

    real_df = config_df[config_df["sample_id"].astype(str).str.startswith("real_")].copy()

    for _, cfg in real_df.iterrows():
        sample_id = str(cfg["sample_id"])
        real_case_id = get_real_case_id(sample_id)

        corr_path = DATA_DIR / f"{sample_id}_correspondences.csv"

        if not corr_path.exists():
            sample_rows.append({
                "sample_id": sample_id,
                "real_case_id": real_case_id,
                "correspondence_csv_exists": False,
            })
            continue

        df = pd.read_csv(corr_path)

        missing = [c for c in REQUIRED_CORR_COLS if c not in df.columns]

        if missing:
            raise RuntimeError(f"{corr_path} missing columns: {missing}")

        label_threshold = float(cfg["label_threshold_m"])
        max_corr_distance = float(cfg["max_corr_distance_m"])

        y_from_distance = (df["label_distance"].values < label_threshold).astype(int)
        y_saved = df["target_weight"].astype(int).values

        label_mismatch_count = int(np.sum(y_from_distance != y_saved))

        distance_T0 = df["distance_T0"].values
        candidate_above_max_count = int(np.sum(distance_T0 > max_corr_distance + 1e-9))

        row = {
            "sample_id": sample_id,
            "real_case_id": real_case_id,
            "correspondence_csv_exists": True,
            "num_rows": len(df),
            "num_positive": int(df["target_weight"].sum()),
            "num_negative": int(len(df) - df["target_weight"].sum()),
            "positive_rate": float(df["target_weight"].mean()),
            "label_threshold_m": label_threshold,
            "max_corr_distance_m": max_corr_distance,
            "label_mismatch_count": label_mismatch_count,
            "label_mismatch_rate": float(label_mismatch_count / max(len(df), 1)),
            "candidate_above_max_corr_count": candidate_above_max_count,
            "candidate_above_max_corr_rate": float(candidate_above_max_count / max(len(df), 1)),
            "label_distance_min": float(df["label_distance"].min()),
            "label_distance_median": float(df["label_distance"].median()),
            "label_distance_mean": float(df["label_distance"].mean()),
            "label_distance_p90": float(df["label_distance"].quantile(0.90)),
            "label_distance_max": float(df["label_distance"].max()),
        }

        for feature in FEATURES:
            values = df[feature].astype(float)

            row[f"{feature}_min"] = float(values.min())
            row[f"{feature}_median"] = float(values.median())
            row[f"{feature}_mean"] = float(values.mean())
            row[f"{feature}_p90"] = float(values.quantile(0.90))
            row[f"{feature}_max"] = float(values.max())

        # Extra label separability check: feature means for positives vs negatives.
        pos = df[df["target_weight"] == 1]
        neg = df[df["target_weight"] == 0]

        for feature in FEATURES:
            if len(pos) > 0:
                row[f"{feature}_positive_mean"] = float(pos[feature].mean())
            else:
                row[f"{feature}_positive_mean"] = np.nan

            if len(neg) > 0:
                row[f"{feature}_negative_mean"] = float(neg[feature].mean())
            else:
                row[f"{feature}_negative_mean"] = np.nan

            row[f"{feature}_positive_minus_negative_mean"] = (
                row[f"{feature}_positive_mean"] - row[f"{feature}_negative_mean"]
            )

        sample_rows.append(row)

    sample_df = pd.DataFrame(sample_rows)

    case_df = (
        sample_df
        .groupby("real_case_id", as_index=False)
        .agg(
            num_samples=("sample_id", "nunique"),
            total_rows=("num_rows", "sum"),
            total_positive=("num_positive", "sum"),
            total_negative=("num_negative", "sum"),
            mean_positive_rate=("positive_rate", "mean"),
            min_positive_rate=("positive_rate", "min"),
            max_positive_rate=("positive_rate", "max"),
            total_label_mismatch_count=("label_mismatch_count", "sum"),
            total_candidate_above_max_corr_count=("candidate_above_max_corr_count", "sum"),
            mean_label_distance=("label_distance_mean", "mean"),
            median_label_distance=("label_distance_median", "mean"),
            mean_distance_T0=("distance_T0_mean", "mean"),
            mean_normal_dot_abs=("normal_dot_abs_mean", "mean"),
            mean_fpfh_distance=("fpfh_distance_mean", "mean"),
            mean_log_normalized_density_ratio=("log_normalized_density_ratio_mean", "mean"),
        )
    )

    case_df["global_positive_rate"] = (
        case_df["total_positive"] / (case_df["total_positive"] + case_df["total_negative"])
    )

    return sample_df, case_df


def audit_transform_direction(config_df):
    if not HAS_OPEN3D:
        print("Open3D not available. Skipping transform-direction audit.")
        return pd.DataFrame()

    rows = []

    real_df = config_df[config_df["sample_id"].astype(str).str.startswith("real_")].copy()
    real_df["real_case_id"] = real_df["sample_id"].apply(get_real_case_id)

    # Same real case may have easy/medium/hard variants.
    # We only need one transform-direction check per unique real case.
    unique_cases = real_df.drop_duplicates(subset=["real_case_id"]).copy()

    for _, cfg in unique_cases.iterrows():
        sample_id = str(cfg["sample_id"])
        real_case_id = str(cfg["real_case_id"])

        reference_path = resolve_path(cfg["reference_path"])
        observation_path = resolve_path(cfg["observation_path"])
        matrix_path = resolve_path(cfg["label_transform_path"])

        voxel_size = float(cfg["voxel_size"])
        label_threshold = float(cfg["label_threshold_m"])

        print(f"Checking transform direction: {real_case_id}")

        try:
            reference_points = load_cloud_points(reference_path, voxel_size)
            observation_points = load_cloud_points(observation_path, voxel_size)
            T_raw = read_transform(matrix_path)

            # Limit observation points for faster audit.
            max_points = 5000
            if len(observation_points) > max_points:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(observation_points), size=max_points, replace=False)
                observation_points_eval = observation_points[indices]
            else:
                observation_points_eval = observation_points

            obs_raw_to_ref = transform_points(observation_points_eval, T_raw)
            obs_inv_to_ref = transform_points(observation_points_eval, np.linalg.inv(T_raw))

            distances_raw = nearest_distances_to_reference(reference_points, obs_raw_to_ref)
            distances_inv = nearest_distances_to_reference(reference_points, obs_inv_to_ref)

            raw_stats = summarize_distances(distances_raw, label_threshold)
            inv_stats = summarize_distances(distances_inv, label_threshold)

            if raw_stats["nn_median"] < inv_stats["nn_median"]:
                better_direction = "raw_transform"
            else:
                better_direction = "inverse_transform"

            row = {
                "real_case_id": real_case_id,
                "sample_id_used": sample_id,
                "reference_path": str(cfg["reference_path"]),
                "observation_path": str(cfg["observation_path"]),
                "label_transform_path": str(cfg["label_transform_path"]),
                "configured_label_transform_direction": cfg["label_transform_direction"],
                "voxel_size": voxel_size,
                "label_threshold_m": label_threshold,
                "num_reference_points_down": len(reference_points),
                "num_observation_points_down": len(observation_points),
                "num_observation_points_eval": len(observation_points_eval),
                "better_direction_by_median_nn": better_direction,
            }

            for k, v in raw_stats.items():
                row[f"raw_{k}"] = v

            for k, v in inv_stats.items():
                row[f"inverse_{k}"] = v

            row["raw_minus_inverse_median_nn"] = (
                row["raw_nn_median"] - row["inverse_nn_median"]
            )

            row["raw_minus_inverse_inlier_rate_label_threshold"] = (
                row["raw_nn_inlier_rate_label_threshold"]
                - row["inverse_nn_inlier_rate_label_threshold"]
            )

            rows.append(row)

        except Exception as e:
            rows.append({
                "real_case_id": real_case_id,
                "sample_id_used": sample_id,
                "reference_path": str(cfg["reference_path"]),
                "observation_path": str(cfg["observation_path"]),
                "label_transform_path": str(cfg["label_transform_path"]),
                "configured_label_transform_direction": cfg["label_transform_direction"],
                "error": str(e),
            })

    return pd.DataFrame(rows)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    config_df = pd.read_csv(CONFIG_PATH)

    print("=" * 100)
    print("Audit real feature extraction and labeling: v2")
    print("=" * 100)
    print(f"Config: {CONFIG_PATH}")
    print(f"Data:   {DATA_DIR}")
    print("=" * 100)

    config_audit_df = audit_config(config_df)
    sample_audit_df, case_audit_df = audit_correspondence_csvs(config_df)
    transform_audit_df = audit_transform_direction(config_df)

    config_path = TABLE_DIR / "audit_real_config_v2.csv"
    sample_path = TABLE_DIR / "audit_real_feature_labeling_v2_by_sample.csv"
    case_path = TABLE_DIR / "audit_real_feature_labeling_v2_by_case.csv"
    transform_path = TABLE_DIR / "audit_real_transform_direction_v2.csv"

    config_audit_df.to_csv(config_path, index=False)
    sample_audit_df.to_csv(sample_path, index=False)
    case_audit_df.to_csv(case_path, index=False)
    transform_audit_df.to_csv(transform_path, index=False)

    print("\nReal config summary:")
    print(f"Real sample rows: {len(config_audit_df)}")
    print(f"Unique real cases: {config_audit_df['real_case_id'].nunique()}")
    print(f"Missing reference paths: {int((~config_audit_df['reference_exists']).sum())}")
    print(f"Missing observation paths: {int((~config_audit_df['observation_exists']).sum())}")
    print(f"Missing matrix paths: {int((~config_audit_df['matrix_exists']).sum())}")

    print("\nLabeling consistency:")
    print(f"Total label mismatches: {int(sample_audit_df['label_mismatch_count'].sum())}")
    print(f"Total candidates above max_corr_distance: {int(sample_audit_df['candidate_above_max_corr_count'].sum())}")

    print("\nTop cases by label mismatch:")
    print(
        case_audit_df
        .sort_values("total_label_mismatch_count", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nTransform direction audit:")
    if not transform_audit_df.empty:
        cols = [
            "real_case_id",
            "configured_label_transform_direction",
            "better_direction_by_median_nn",
            "raw_nn_median",
            "inverse_nn_median",
            "raw_nn_inlier_rate_label_threshold",
            "inverse_nn_inlier_rate_label_threshold",
        ]

        existing_cols = [c for c in cols if c in transform_audit_df.columns]

        print(transform_audit_df[existing_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {config_path}")
    print(f"  {sample_path}")
    print(f"  {case_path}")
    print(f"  {transform_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
