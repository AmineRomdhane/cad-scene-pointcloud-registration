import open3d as o3d
import numpy as np
from pathlib import Path
import time
import csv
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "synthetic_cases"

MATRICES_DIR = BASE_DIR / "results" / "matrices"
TABLES_DIR = BASE_DIR / "results" / "tables"
POINTCLOUDS_DIR = BASE_DIR / "results" / "pointclouds"

MATRICES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
POINTCLOUDS_DIR.mkdir(parents=True, exist_ok=True)

VISUALIZE = False

FIELDNAMES = [
    "timestamp",
    "scenario",
    "method",
    "voxel_size",
    "threshold",
    "max_iteration",
    "normal_radius",
    "normal_max_nn",
    "num_reference_points",
    "num_observation_points",
    "num_reference_down",
    "num_observation_down",
    "fitness",
    "rmse",
    "computation_time_s",
    "translation_error_m",
    "rotation_error_deg"
]

def rotation_error_deg(R_true, R_est):
    R_diff = R_true.T @ R_est
    value = (np.trace(R_diff) - 1) / 2
    value = np.clip(value, -1.0, 1.0)
    return np.rad2deg(np.arccos(value))

def save_metrics_csv(metrics, csv_path):
    file_exists = csv_path.exists()

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        writer.writerow({key: metrics.get(key, "N/A") for key in FIELDNAMES})

if __name__ == "__main__":

    reference_path = DATA_DIR / "reference_t0.ply"
    observation_path = DATA_DIR / "observation_t1.ply"
    true_transform_path = DATA_DIR / "T_true.txt"

    reference = o3d.io.read_point_cloud(str(reference_path))
    observation = o3d.io.read_point_cloud(str(observation_path))
    T_true = np.loadtxt(true_transform_path)

    if len(reference.points) == 0 or len(observation.points) == 0:
        raise RuntimeError("Point cloud is empty. Run script 01 first.")

    voxel_size = 0.05
    threshold = 0.2
    max_iteration = 50

    init = np.eye(4)

    reference_down = reference.voxel_down_sample(voxel_size)
    observation_down = observation.voxel_down_sample(voxel_size)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iteration
    )

    start_time = time.time()

    result = o3d.pipelines.registration.registration_icp(
        observation_down,
        reference_down,
        threshold,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria
    )

    elapsed_time = time.time() - start_time

    T_est = result.transformation
    T_expected = np.linalg.inv(T_true)

    translation_error = np.linalg.norm(T_expected[:3, 3] - T_est[:3, 3])
    rotation_error = rotation_error_deg(T_expected[:3, :3], T_est[:3, :3])

    aligned_observation = o3d.geometry.PointCloud(observation_down)
    aligned_observation.transform(T_est)

    np.savetxt(MATRICES_DIR / "T_est_point_to_point.txt", T_est)
    np.savetxt(MATRICES_DIR / "T_expected_inverse_T_true.txt", T_expected)

    o3d.io.write_point_cloud(
        str(POINTCLOUDS_DIR / "aligned_observation_point_to_point.ply"),
        aligned_observation
    )

    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": "synthetic_static_noise",
        "method": "ICP_point_to_point",
        "voxel_size": voxel_size,
        "threshold": threshold,
        "max_iteration": max_iteration,
        "normal_radius": "N/A",
        "normal_max_nn": "N/A",
        "num_reference_points": len(reference.points),
        "num_observation_points": len(observation.points),
        "num_reference_down": len(reference_down.points),
        "num_observation_down": len(observation_down.points),
        "fitness": result.fitness,
        "rmse": result.inlier_rmse,
        "computation_time_s": elapsed_time,
        "translation_error_m": translation_error,
        "rotation_error_deg": rotation_error
    }

    save_metrics_csv(metrics, TABLES_DIR / "icp_results.csv")

    print("\n=== ICP Point-to-Point Results ===\n")
    print("Estimated transformation T_est:")
    print(T_est)

    print("\nExpected transformation inv(T_true):")
    print(T_expected)

    print("\nMetrics:")
    print(f"Fitness: {result.fitness:.4f}")
    print(f"RMSE: {result.inlier_rmse:.6f}")
    print(f"Computation time: {elapsed_time:.4f} s")
    print(f"Translation error: {translation_error:.6f} m")
    print(f"Rotation error: {rotation_error:.6f} deg")

    print("\nSaved:")
    print(MATRICES_DIR / "T_est_point_to_point.txt")
    print(TABLES_DIR / "icp_results.csv")
    print(POINTCLOUDS_DIR / "aligned_observation_point_to_point.ply")

    if VISUALIZE:
        reference_down.paint_uniform_color([0, 0, 1])
        observation_down.paint_uniform_color([1, 0, 0])
        aligned_observation.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries(
            [reference_down, observation_down],
            window_name="Before ICP point-to-point"
        )

        o3d.visualization.draw_geometries(
            [reference_down, aligned_observation],
            window_name="After ICP point-to-point"
        )
