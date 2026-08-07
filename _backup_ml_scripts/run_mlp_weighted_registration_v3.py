#!/usr/bin/env python3
"""
Run ML-weighted registration on one sample from the v3 config.

Comparison:
1. Initial transform T0
2. Normal point-to-plane ICP from T0
3. MLP-weighted SVD correction from T0

The MLP predicts correspondence reliability weights.
Then a weighted rigid correction is estimated from candidate correspondences.

Output:
results/by_test/mlp_weighted_registration_v3/<sample_id>/
"""

from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_CANDIDATES = [
    BASE_DIR / "configs" / "dataset_samples_synthetic_plus_real_curated_clean_v3.csv",
    BASE_DIR / "configs" / "dataset_samples_synthetic_plus_real_curated_clean_v2.csv",
]

MODEL_DIR = BASE_DIR / "results" / "by_test" / "mlp_synthetic_plus_all_real_v3_final"

MODEL_PATH = MODEL_DIR / "mlp_synthetic_plus_all_real_v3_final_model.pt"
SCALER_PATH = MODEL_DIR / "mlp_synthetic_plus_all_real_v3_final_scaler.pkl"

TEST_NAME = "mlp_weighted_registration_v3"
OUT_ROOT = BASE_DIR / "results" / "by_test" / TEST_NAME

FEATURES = [
    "distance_T0",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]


class CorrespondenceMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def resolve_path(path_str):
    p = Path(str(path_str))

    if p.is_absolute():
        return p

    return BASE_DIR / p


def load_transform(path):
    T = np.loadtxt(path)

    if T.shape != (4, 4):
        raise RuntimeError(f"Transform is not 4x4: {path}")

    return T


def save_transform(path, T):
    np.savetxt(path, T, fmt="%.10f")


def inverse_transform(T):
    return np.linalg.inv(T)


def rotation_error_deg(T_est, T_ref):
    R_est = T_est[:3, :3]
    R_ref = T_ref[:3, :3]

    R_delta = R_ref.T @ R_est

    value = (np.trace(R_delta) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)

    return float(np.degrees(np.arccos(value)))


def translation_error(T_est, T_ref):
    return float(np.linalg.norm(T_est[:3, 3] - T_ref[:3, 3]))


def preprocess_pcd(pcd, voxel_size, normal_radius, fpfh_radius):
    if voxel_size > 0:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = pcd

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=30,
        )
    )

    pcd_down.normalize_normals()

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius,
            max_nn=100,
        )
    )

    return pcd_down, np.asarray(fpfh.data).T


def local_density(points, radius):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    tree = o3d.geometry.KDTreeFlann(pcd)

    density = np.zeros(len(points), dtype=np.float64)

    for i, p in enumerate(points):
        k, _, _ = tree.search_radius_vector_3d(p, radius)
        density[i] = max(k - 1, 0)

    return density


def transform_points(points, T):
    points_h = np.hstack([points, np.ones((len(points), 1))])
    out = (T @ points_h.T).T[:, :3]
    return out


def transform_normals(normals, T):
    R = T[:3, :3]
    out = (R @ normals.T).T

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return out / norms


def build_candidate_features(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    T0,
    max_corr_distance,
    density_radius,
):
    source_pts = np.asarray(source_down.points)
    target_pts = np.asarray(target_down.points)

    source_normals = np.asarray(source_down.normals)
    target_normals = np.asarray(target_down.normals)

    source_pts_t0 = transform_points(source_pts, T0)
    source_normals_t0 = transform_normals(source_normals, T0)

    target_tree = o3d.geometry.KDTreeFlann(target_down)

    rows = []

    target_indices = []
    source_indices = []
    distances = []

    for i, p in enumerate(source_pts_t0):
        k, idx, dist2 = target_tree.search_knn_vector_3d(p, 1)

        if k <= 0:
            continue

        j = int(idx[0])
        d = float(np.sqrt(dist2[0]))

        if d <= max_corr_distance:
            source_indices.append(i)
            target_indices.append(j)
            distances.append(d)

    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)

    if len(source_indices) == 0:
        raise RuntimeError("No candidate correspondences found. Increase max_corr_distance.")

    # Mutual nearest neighbor
    source_t0_pcd = o3d.geometry.PointCloud()
    source_t0_pcd.points = o3d.utility.Vector3dVector(source_pts_t0)
    source_t0_tree = o3d.geometry.KDTreeFlann(source_t0_pcd)

    is_mutual = np.zeros(len(source_indices), dtype=np.float64)

    for k_idx, (i, j) in enumerate(zip(source_indices, target_indices)):
        kt, idx_back, _ = source_t0_tree.search_knn_vector_3d(target_pts[j], 1)

        if kt > 0 and int(idx_back[0]) == int(i):
            is_mutual[k_idx] = 1.0

    # Density feature
    src_density = local_density(source_pts_t0, density_radius)
    tgt_density = local_density(target_pts, density_radius)

    med_src = np.median(src_density[src_density > 0]) if np.any(src_density > 0) else 1.0
    med_tgt = np.median(tgt_density[tgt_density > 0]) if np.any(tgt_density > 0) else 1.0

    eps = 1e-6

    # Features
    normal_dot_abs = np.abs(
        np.sum(source_normals_t0[source_indices] * target_normals[target_indices], axis=1)
    )

    fpfh_distance = np.linalg.norm(
        source_fpfh[source_indices] - target_fpfh[target_indices],
        axis=1,
    )

    log_density_ratio = np.log(
        ((src_density[source_indices] / med_src) + eps)
        /
        ((tgt_density[target_indices] / med_tgt) + eps)
    )

    features_df = pd.DataFrame({
        "source_index": source_indices,
        "target_index": target_indices,
        "distance_T0": distances,
        "normal_dot_abs": normal_dot_abs,
        "fpfh_distance": fpfh_distance,
        "log_normalized_density_ratio": log_density_ratio,
        "is_mutual_nn": is_mutual,
    })

    return features_df


def weighted_svd_correction(source_down, target_down, features_df, weights, T0, min_weight=0.0):
    source_pts = np.asarray(source_down.points)
    target_pts = np.asarray(target_down.points)

    src_idx = features_df["source_index"].values.astype(int)
    tgt_idx = features_df["target_index"].values.astype(int)

    A = transform_points(source_pts[src_idx], T0)
    B = target_pts[tgt_idx]

    w = np.asarray(weights, dtype=np.float64)

    keep = w > min_weight

    A = A[keep]
    B = B[keep]
    w = w[keep]

    if len(A) < 3:
        raise RuntimeError("Not enough weighted correspondences for SVD registration.")

    w = np.clip(w, 1e-6, None)
    w = w / np.sum(w)

    centroid_A = np.sum(A * w[:, None], axis=0)
    centroid_B = np.sum(B * w[:, None], axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ (B_centered * w[:, None])

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    delta_T = np.eye(4)
    delta_T[:3, :3] = R
    delta_T[:3, 3] = t

    final_T = delta_T @ T0

    stats = {
        "num_candidates_total": int(len(features_df)),
        "num_candidates_used": int(len(A)),
        "mean_weight_used": float(np.mean(w)),
        "min_raw_weight": float(np.min(weights)),
        "max_raw_weight": float(np.max(weights)),
        "mean_raw_weight": float(np.mean(weights)),
        "svd_singular_1": float(S[0]),
        "svd_singular_2": float(S[1]),
        "svd_singular_3": float(S[2]),
    }

    return final_T, delta_T, stats


def evaluate_transform(source_down, target_down, T, max_corr_distance):
    eval_result = o3d.pipelines.registration.evaluate_registration(
        source_down,
        target_down,
        max_corr_distance,
        T,
    )

    return {
        "fitness": float(eval_result.fitness),
        "rmse": float(eval_result.inlier_rmse),
        "num_correspondences": int(len(eval_result.correspondence_set)),
    }


def run_icp(source_down, target_down, T0, max_corr_distance):
    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_corr_distance,
        T0,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )

    return result.transformation, result


def find_config_path():
    for p in CONFIG_CANDIDATES:
        if p.exists():
            return p

    raise FileNotFoundError("Could not find v3 config CSV.")


def load_model_and_scaler(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    dropout = float(checkpoint.get("dropout", 0.10))
    input_dim = int(checkpoint.get("input_dim", len(FEATURES)))

    model = CorrespondenceMLP(input_dim=input_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with SCALER_PATH.open("rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_weights(model, scaler, features_df, device):
    X = features_df[FEATURES].values
    Xs = scaler.transform(X)

    xt = torch.tensor(Xs, dtype=torch.float32).to(device)

    probs = []

    with torch.no_grad():
        for start in range(0, len(xt), 8192):
            xb = xt[start:start + 8192]
            logits = model(xb)
            prob = torch.sigmoid(logits)
            probs.append(prob.cpu().numpy())

    return np.concatenate(probs, axis=0)


def choose_row(config_df, sample_id):
    config_df["sample_id"] = config_df["sample_id"].astype(str)

    matches = config_df[config_df["sample_id"] == sample_id]

    if len(matches) == 0:
        print("Available real sample_ids:")
        real_ids = sorted(config_df[config_df["sample_id"].str.startswith("real_")]["sample_id"].unique())
        for sid in real_ids[:200]:
            print("  ", sid)

        raise RuntimeError(f"sample_id not found: {sample_id}")

    return matches.iloc[0].to_dict()


def get_column(row, candidates, default=None):
    for c in candidates:
        if c in row and not pd.isna(row[c]):
            return row[c]

    return default


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-id",
        required=True,
        help="sample_id from configs/dataset_samples_synthetic_plus_real_curated_clean_v3.csv",
    )

    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=0.0,
        help="Minimum MLP weight used by weighted SVD. Use 0.0 for soft weights over all candidates.",
    )

    parser.add_argument(
        "--max-corr-distance",
        type=float,
        default=None,
        help="Override max correspondence distance.",
    )

    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train it first.")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}. Train it first.")

    config_path = find_config_path()
    config_df = pd.read_csv(config_path)

    row = choose_row(config_df, args.sample_id)

    reference_path = resolve_path(get_column(row, ["reference_path", "target_file", "target_path"]))
    observation_path = resolve_path(get_column(row, ["observation_path", "cad_file", "source_path"]))
    label_transform_path = resolve_path(get_column(row, ["label_transform_path", "matrix_file"]))

    voxel_size = float(get_column(row, ["voxel_size", "voxel"], 0.05))
    normal_radius = float(get_column(row, ["normal_radius"], 0.15))
    fpfh_radius = float(get_column(row, ["fpfh_radius"], 0.25))
    density_radius = float(get_column(row, ["density_radius"], 0.15))
    max_corr_distance = float(args.max_corr_distance if args.max_corr_distance is not None else get_column(row, ["max_corr_distance"], 0.50))

    label_direction = str(get_column(row, ["label_transform_direction"], "obs_to_ref"))

    T_label = load_transform(label_transform_path)

    if label_direction == "ref_to_obs":
        T_label = inverse_transform(T_label)

    # Controlled initial transform:
    # Use T0 from config if available, otherwise use identity.
    # For now we use identity unless an explicit initial_transform_path exists.
    init_path_value = get_column(row, ["initial_transform_path", "T0_path"], None)

    if init_path_value is not None:
        init_path = resolve_path(init_path_value)
        T0 = load_transform(init_path)
        init_source = str(init_path)
    else:
        T0 = np.eye(4)
        init_source = "identity"

    out_dir = OUT_ROOT / args.sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MLP weighted registration v3")
    print("=" * 100)
    print("sample_id:", args.sample_id)
    print("config:", config_path)
    print("reference:", reference_path)
    print("observation:", observation_path)
    print("label transform:", label_transform_path)
    print("initial transform source:", init_source)
    print("voxel_size:", voxel_size)
    print("normal_radius:", normal_radius)
    print("fpfh_radius:", fpfh_radius)
    print("density_radius:", density_radius)
    print("max_corr_distance:", max_corr_distance)
    print("weight_threshold:", args.weight_threshold)
    print("output:", out_dir)
    print("=" * 100)

    source = o3d.io.read_point_cloud(str(observation_path))
    target = o3d.io.read_point_cloud(str(reference_path))

    if source.is_empty():
        raise RuntimeError(f"Observation/source cloud is empty: {observation_path}")

    if target.is_empty():
        raise RuntimeError(f"Reference/target cloud is empty: {reference_path}")

    source_down, source_fpfh = preprocess_pcd(source, voxel_size, normal_radius, fpfh_radius)
    target_down, target_fpfh = preprocess_pcd(target, voxel_size, normal_radius, fpfh_radius)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler = load_model_and_scaler(device)

    features_df = build_candidate_features(
        source_down=source_down,
        target_down=target_down,
        source_fpfh=source_fpfh,
        target_fpfh=target_fpfh,
        T0=T0,
        max_corr_distance=max_corr_distance,
        density_radius=density_radius,
    )

    weights = predict_weights(model, scaler, features_df, device)
    features_df["mlp_weight"] = weights

    T_mlp, T_delta, mlp_stats = weighted_svd_correction(
        source_down=source_down,
        target_down=target_down,
        features_df=features_df,
        weights=weights,
        T0=T0,
        min_weight=args.weight_threshold,
    )

    T_icp, icp_result = run_icp(
        source_down=source_down,
        target_down=target_down,
        T0=T0,
        max_corr_distance=max_corr_distance,
    )

    eval_initial = evaluate_transform(source_down, target_down, T0, max_corr_distance)
    eval_mlp = evaluate_transform(source_down, target_down, T_mlp, max_corr_distance)
    eval_icp = evaluate_transform(source_down, target_down, T_icp, max_corr_distance)
    eval_label = evaluate_transform(source_down, target_down, T_label, max_corr_distance)

    metrics = []

    for method, T, ev in [
        ("initial_T0", T0, eval_initial),
        ("mlp_weighted_svd", T_mlp, eval_mlp),
        ("normal_icp", T_icp, eval_icp),
        ("pseudo_gt_label", T_label, eval_label),
    ]:
        row_metrics = {
            "sample_id": args.sample_id,
            "method": method,
            "fitness": ev["fitness"],
            "rmse": ev["rmse"],
            "num_correspondences": ev["num_correspondences"],
            "translation_error_vs_label_m": translation_error(T, T_label),
            "rotation_error_vs_label_deg": rotation_error_deg(T, T_label),
            "max_corr_distance": max_corr_distance,
            "voxel_size": voxel_size,
        }

        metrics.append(row_metrics)

    metrics_df = pd.DataFrame(metrics)

    metrics_path = out_dir / "registration_metrics.csv"
    features_path = out_dir / "candidate_features_with_weights.csv"

    T0_path = out_dir / "T_initial.txt"
    T_mlp_path = out_dir / "T_mlp_weighted_svd.txt"
    T_delta_path = out_dir / "T_mlp_delta.txt"
    T_icp_path = out_dir / "T_normal_icp.txt"
    T_label_path = out_dir / "T_pseudo_gt_label.txt"

    save_transform(T0_path, T0)
    save_transform(T_mlp_path, T_mlp)
    save_transform(T_delta_path, T_delta)
    save_transform(T_icp_path, T_icp)
    save_transform(T_label_path, T_label)

    metrics_df.to_csv(metrics_path, index=False)
    features_df.to_csv(features_path, index=False)

    # Save registered clouds
    source_mlp = o3d.geometry.PointCloud(source_down)
    source_mlp.transform(T_mlp)
    o3d.io.write_point_cloud(str(out_dir / "source_registered_mlp_weighted_svd.ply"), source_mlp)

    source_icp = o3d.geometry.PointCloud(source_down)
    source_icp.transform(T_icp)
    o3d.io.write_point_cloud(str(out_dir / "source_registered_normal_icp.ply"), source_icp)

    source_initial = o3d.geometry.PointCloud(source_down)
    source_initial.transform(T0)
    o3d.io.write_point_cloud(str(out_dir / "source_registered_initial_T0.ply"), source_initial)

    o3d.io.write_point_cloud(str(out_dir / "target_downsampled_reference.ply"), target_down)

    stats_path = out_dir / "mlp_weight_stats.json"

    with stats_path.open("w") as f:
        json.dump(mlp_stats, f, indent=2)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print("\nMLP weight stats:")
    print(json.dumps(mlp_stats, indent=2))

    print("\nSaved:")
    print(metrics_path)
    print(features_path)
    print(T_mlp_path)
    print(T_icp_path)
    print(stats_path)


if __name__ == "__main__":
    main()
