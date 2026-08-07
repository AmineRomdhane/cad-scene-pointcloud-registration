#!/usr/bin/env python3
"""
Create additional synthetic point-cloud scenarios for correspondence-weight learning.

Generated scenarios:
1. synthetic_partial_overlap
2. synthetic_added_outliers
3. synthetic_strong_noise
4. synthetic_scene_change

Each scenario saves:
- reference.ply
- observation.ply
- T_true.txt

Convention:
T_true maps reference -> observation.
The extraction script will invert it when label_transform_direction = ref_to_obs.
"""

from pathlib import Path
import shutil
import numpy as np
import open3d as o3d


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_REF = BASE_DIR / "data" / "synthetic_cases" / "reference_t0.ply"
INPUT_T_TRUE = BASE_DIR / "data" / "synthetic_cases" / "T_true.txt"

OUT_BASE = BASE_DIR / "data" / "learning_synthetic_cases"
OUT_BASE.mkdir(parents=True, exist_ok=True)


def load_reference():
    ref = o3d.io.read_point_cloud(str(INPUT_REF))
    if ref.is_empty():
        raise RuntimeError(f"Empty reference cloud: {INPUT_REF}")
    return ref


def load_transform():
    T = np.loadtxt(INPUT_T_TRUE)
    if T.shape != (4, 4):
        raise RuntimeError(f"T_true must be 4x4, got {T.shape}")
    return T


def apply_transform(cloud, T):
    out = o3d.geometry.PointCloud(cloud)
    out.transform(T)
    return out


def add_gaussian_noise(cloud, sigma, rng):
    pts = np.asarray(cloud.points)
    noisy = pts + rng.normal(0.0, sigma, size=pts.shape)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(noisy)
    return out


def keep_partial_overlap(cloud, keep_ratio=0.65):
    """
    Keep only part of the observation cloud to simulate partial overlap / occlusion.
    Here we keep points below a quantile along x.
    """
    pts = np.asarray(cloud.points)
    x_threshold = np.quantile(pts[:, 0], keep_ratio)
    kept = pts[pts[:, 0] <= x_threshold]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(kept)
    return out


def add_random_outliers(cloud, outlier_ratio, rng):
    """
    Add random points around the observation bounding box.
    These points have no true correspondence in the reference cloud.
    """
    pts = np.asarray(cloud.points)
    n_outliers = int(len(pts) * outlier_ratio)

    min_bound = pts.min(axis=0)
    max_bound = pts.max(axis=0)

    margin = 0.5
    low = min_bound - margin
    high = max_bound + margin

    outliers = rng.uniform(low=low, high=high, size=(n_outliers, 3))
    combined = np.vstack([pts, outliers])

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(combined)
    return out


def add_scene_change_object(cloud, rng):
    """
    Add an extra object in the observation cloud to simulate a scene change.
    The added object does not exist in the reference cloud.
    """
    pts = np.asarray(cloud.points)

    extra_box = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.4, depth=0.5)
    extra_box.translate([1.2, 0.8, 0.1])
    extra_cloud = extra_box.sample_points_uniformly(number_of_points=700)

    extra_pts = np.asarray(extra_cloud.points)

    # Add small noise to avoid perfectly regular points
    extra_pts = extra_pts + rng.normal(0.0, 0.005, size=extra_pts.shape)

    combined = np.vstack([pts, extra_pts])

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(combined)
    return out


def remove_region(cloud, remove_ratio=0.20):
    """
    Remove part of the observation cloud to simulate missing structure.
    """
    pts = np.asarray(cloud.points)

    # Remove high-y region
    y_threshold = np.quantile(pts[:, 1], 1.0 - remove_ratio)
    kept = pts[pts[:, 1] <= y_threshold]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(kept)
    return out


def save_case(case_name, reference, observation, T_true):
    case_dir = OUT_BASE / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    ref_path = case_dir / "reference.ply"
    obs_path = case_dir / "observation.ply"
    T_path = case_dir / "T_true.txt"

    o3d.io.write_point_cloud(str(ref_path), reference)
    o3d.io.write_point_cloud(str(obs_path), observation)
    np.savetxt(T_path, T_true)

    print(f"\nSaved case: {case_name}")
    print(f"  reference:   {ref_path}")
    print(f"  observation: {obs_path}")
    print(f"  T_true:      {T_path}")
    print(f"  ref points:  {len(reference.points)}")
    print(f"  obs points:  {len(observation.points)}")


def main():
    rng = np.random.default_rng(42)

    reference = load_reference()
    T_true = load_transform()

    base_observation = apply_transform(reference, T_true)

    # Case 1: partial overlap
    obs_partial = add_gaussian_noise(base_observation, sigma=0.01, rng=rng)
    obs_partial = keep_partial_overlap(obs_partial, keep_ratio=0.65)
    save_case("synthetic_partial_overlap", reference, obs_partial, T_true)

    # Case 2: added random outliers
    obs_outliers = add_gaussian_noise(base_observation, sigma=0.01, rng=rng)
    obs_outliers = add_random_outliers(obs_outliers, outlier_ratio=0.20, rng=rng)
    save_case("synthetic_added_outliers", reference, obs_outliers, T_true)

    # Case 3: stronger Gaussian noise
    obs_strong_noise = add_gaussian_noise(base_observation, sigma=0.03, rng=rng)
    save_case("synthetic_strong_noise", reference, obs_strong_noise, T_true)

    # Case 4: scene change = missing region + added object
    obs_scene_change = add_gaussian_noise(base_observation, sigma=0.01, rng=rng)
    obs_scene_change = remove_region(obs_scene_change, remove_ratio=0.20)
    obs_scene_change = add_scene_change_object(obs_scene_change, rng=rng)
    save_case("synthetic_scene_change", reference, obs_scene_change, T_true)

    print("\nAll learning synthetic cases created.")


if __name__ == "__main__":
    main()

