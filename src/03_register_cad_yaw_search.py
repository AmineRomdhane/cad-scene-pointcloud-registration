import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import open3d as o3d


def load_pcd(path, name):
    pcd = o3d.io.read_point_cloud(path)

    if pcd.is_empty():
        raise RuntimeError(f"{name} is empty or could not be loaded: {path}")

    pcd.remove_non_finite_points()

    bbox = pcd.get_axis_aligned_bounding_box()

    print(f"[INFO] Loaded {name}: {path}")
    print(f"[INFO] {name} points: {len(pcd.points)}")
    print(f"[INFO] {name} extent: {bbox.get_extent()}")
    print(f"[INFO] {name} center: {bbox.get_center()}")

    return pcd


def preprocess(pcd, voxel, normal_radius, normal_max_nn, name):
    if voxel > 0:
        down = pcd.voxel_down_sample(voxel)
    else:
        down = copy.deepcopy(pcd)

    down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn
        )
    )
    down.normalize_normals()

    print(f"[INFO] {name} downsampled points: {len(down.points)}")

    return down


def rotz(deg):
    theta = math.radians(deg)
    c = math.cos(theta)
    s = math.sin(theta)

    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return R


def make_center_yaw_transform(cad, target, yaw_deg):
    cad_center = np.asarray(cad.get_center())
    target_center = np.asarray(target.get_center())

    R = rotz(yaw_deg)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = target_center - R @ cad_center

    return T


def save_combined(target, cad_registered, output_path):
    target_c = copy.deepcopy(target)
    cad_c = copy.deepcopy(cad_registered)

    target_colors = np.zeros((len(target_c.points), 3))
    cad_colors = np.zeros((len(cad_c.points), 3))

    target_colors[:, :] = [0.0, 0.7, 1.0]
    cad_colors[:, :] = [1.0, 0.2, 0.0]

    target_c.colors = o3d.utility.Vector3dVector(target_colors)
    cad_c.colors = o3d.utility.Vector3dVector(cad_colors)

    combined = target_c + cad_c

    o3d.io.write_point_cloud(output_path, combined)


def evaluate(source_down, target_down, transform, distance_threshold):
    result = o3d.pipelines.registration.evaluate_registration(
        source_down,
        target_down,
        distance_threshold,
        transform
    )

    return {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "correspondences": int(len(result.correspondence_set))
    }


def parse_angles(text):
    return [float(x.strip()) for x in text.split(",") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser(
        description="Try multiple yaw initializations and ICP to avoid inverted RANSAC solutions."
    )

    parser.add_argument("--cad", required=True, help="CAD proxy point cloud")
    parser.add_argument("--target", required=True, help="Target scan cluster")
    parser.add_argument("--output", required=True, help="Output folder")

    parser.add_argument(
        "--angles",
        default="0,45,90,135,180,225,270,315",
        help="Yaw angles in degrees, comma-separated"
    )

    parser.add_argument("--voxel", type=float, default=0.05)
    parser.add_argument("--normal_radius_factor", type=float, default=3.0)
    parser.add_argument("--icp_dist_factor", type=float, default=2.0)
    parser.add_argument("--normal_max_nn", type=int, default=50)
    parser.add_argument("--icp_max_iter", type=int, default=100)

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cad = load_pcd(args.cad, "CAD")
    target = load_pcd(args.target, "target")

    normal_radius = args.voxel * args.normal_radius_factor
    icp_dist = args.voxel * args.icp_dist_factor

    print(f"[INFO] voxel: {args.voxel}")
    print(f"[INFO] normal_radius: {normal_radius}")
    print(f"[INFO] icp_dist: {icp_dist}")

    cad_down = preprocess(cad, args.voxel, normal_radius, args.normal_max_nn, "CAD")
    target_down = preprocess(target, args.voxel, normal_radius, args.normal_max_nn, "target")

    angles = parse_angles(args.angles)

    rows = []
    best = None

    for idx, angle in enumerate(angles):
        candidate_name = f"candidate_{idx:02d}_yaw_{angle:g}"
        candidate_dir = output_dir / candidate_name
        candidate_dir.mkdir(parents=True, exist_ok=True)

        print("")
        print(f"[INFO] Testing {candidate_name}")

        T_init = make_center_yaw_transform(cad, target, angle)

        result = o3d.pipelines.registration.registration_icp(
            cad_down,
            target_down,
            icp_dist,
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=args.icp_max_iter
            )
        )

        T_final = result.transformation

        eval_data = evaluate(
            cad_down,
            target_down,
            T_final,
            icp_dist
        )

        cad_registered = copy.deepcopy(cad)
        cad_registered.transform(T_final)

        cad_registered_path = candidate_dir / "cad_registered.pcd"
        combined_path = candidate_dir / "combined_target_and_registered_cad.pcd"
        matrix_path = candidate_dir / "T_scene_cad.txt"

        o3d.io.write_point_cloud(str(cad_registered_path), cad_registered)
        save_combined(target, cad_registered, str(combined_path))
        np.savetxt(str(matrix_path), T_final, fmt="%.10f")

        summary = {
            "cad_file": args.cad,
            "target_file": args.target,
            "output": str(candidate_dir),
            "method": "yaw_search_icp",
            "yaw_deg": float(angle),
            "voxel": args.voxel,
            "normal_radius": normal_radius,
            "icp_dist": icp_dist,
            "use_center_init": True,
            "skip_ransac": True,
            "ransac": {
                "skipped": True,
                "fitness": "",
                "inlier_rmse": "",
                "time_s": "",
                "transformation": np.eye(4).tolist()
            },
            "icp": {
                "fitness": float(result.fitness),
                "inlier_rmse": float(result.inlier_rmse),
                "time_s": "",
                "transformation_from_registration_input": T_final.tolist()
            },
            "final_evaluation": eval_data,
            "T_scene_cad": T_final.tolist(),
            "outputs": {
                "registered_cad": "cad_registered.pcd",
                "combined_visualization": "combined_target_and_registered_cad.pcd",
                "matrix": "T_scene_cad.txt"
            }
        }

        with open(candidate_dir / "registration_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        row = {
            "candidate": candidate_name,
            "yaw_deg": angle,
            "icp_fitness": float(result.fitness),
            "icp_rmse": float(result.inlier_rmse),
            "eval_fitness": eval_data["fitness"],
            "eval_rmse": eval_data["inlier_rmse"],
            "correspondences": eval_data["correspondences"],
            "candidate_dir": str(candidate_dir)
        }

        rows.append(row)

        print(f"[RESULT] yaw={angle:g} | fitness={result.fitness:.6f} | rmse={result.inlier_rmse:.6f}")

        score = float(result.fitness) - float(result.inlier_rmse)

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "candidate": candidate_name,
                "candidate_dir": str(candidate_dir),
                "yaw_deg": angle,
                "fitness": float(result.fitness),
                "rmse": float(result.inlier_rmse)
            }

    csv_path = output_dir / "yaw_search_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "yaw_deg",
                "icp_fitness",
                "icp_rmse",
                "eval_fitness",
                "eval_rmse",
                "correspondences",
                "candidate_dir"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "best_candidate_by_metric.json", "w") as f:
        json.dump(best, f, indent=4)

    print("")
    print("[DONE] Yaw-search ICP finished.")
    print(f"[DONE] Results table: {csv_path}")
    print(f"[DONE] Best by metric: {best}")
    print("")
    print("[IMPORTANT] Metric best is not always visually best. Open the candidates and choose the correct one visually.")


if __name__ == "__main__":
    main()
