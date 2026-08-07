import argparse
import copy
import json
import os
import time

import numpy as np
import open3d as o3d


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def load_point_cloud(path, name):
    pcd = o3d.io.read_point_cloud(path)

    if pcd.is_empty():
        raise RuntimeError(f"{name} point cloud is empty or could not be loaded: {path}")

    pcd.remove_non_finite_points()

    print(f"[INFO] Loaded {name}: {path}")
    print(f"[INFO] {name} points: {len(pcd.points)}")

    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"[INFO] {name} extent xyz: {bbox.get_extent()}")
    print(f"[INFO] {name} center xyz: {bbox.get_center()}")

    return pcd


def save_matrix(path, matrix):
    np.savetxt(path, matrix, fmt="%.10f")
    print(f"[INFO] Saved matrix: {path}")


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Saved JSON: {path}")


def preprocess_point_cloud(pcd, voxel_size, normal_radius, normal_max_nn, fpfh_radius, fpfh_max_nn, name):
    print(f"[INFO] Preprocessing {name}...")

    if voxel_size > 0:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = copy.deepcopy(pcd)

    print(f"[INFO] {name} downsampled points: {len(pcd_down.points)}")

    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn
        )
    )
    pcd_down.normalize_normals()

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius,
            max_nn=fpfh_max_nn
        )
    )

    return pcd_down, fpfh


def center_cad_to_target(cad_pcd, target_pcd):
    cad_center = np.asarray(cad_pcd.get_center())
    target_center = np.asarray(target_pcd.get_center())

    translation = target_center - cad_center

    T = np.eye(4)
    T[:3, 3] = translation

    cad_centered = copy.deepcopy(cad_pcd)
    cad_centered.transform(T)

    print(f"[INFO] Center initialization translation: {translation}")

    return cad_centered, T


def run_ransac(source_down, target_down, source_fpfh, target_fpfh, distance_threshold, ransac_n, max_iter, confidence):
    print("[INFO] Running FPFH + RANSAC global registration...")

    start = time.time()

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iter,
            confidence
        )
    )

    elapsed = time.time() - start

    print("[INFO] RANSAC finished.")
    print(f"[INFO] RANSAC time: {elapsed:.2f} s")
    print(f"[INFO] RANSAC fitness: {result.fitness:.6f}")
    print(f"[INFO] RANSAC RMSE: {result.inlier_rmse:.6f}")
    print("[INFO] RANSAC transform:")
    print(result.transformation)

    return result, elapsed


def run_icp(source_down, target_down, init_transform, distance_threshold, max_iter):
    print("[INFO] Running point-to-plane ICP refinement...")

    start = time.time()

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        distance_threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter
        )
    )

    elapsed = time.time() - start

    print("[INFO] ICP finished.")
    print(f"[INFO] ICP time: {elapsed:.2f} s")
    print(f"[INFO] ICP fitness: {result.fitness:.6f}")
    print(f"[INFO] ICP RMSE: {result.inlier_rmse:.6f}")
    print("[INFO] ICP transform:")
    print(result.transformation)

    return result, elapsed


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


def save_combined_cloud(target_pcd, cad_registered, output_path):
    target_colored = copy.deepcopy(target_pcd)
    cad_colored = copy.deepcopy(cad_registered)

    target_colors = np.zeros((len(target_colored.points), 3))
    cad_colors = np.zeros((len(cad_colored.points), 3))

    target_colors[:, :] = [0.0, 0.7, 1.0]
    cad_colors[:, :] = [1.0, 0.2, 0.0]

    target_colored.colors = o3d.utility.Vector3dVector(target_colors)
    cad_colored.colors = o3d.utility.Vector3dVector(cad_colors)

    combined = target_colored + cad_colored
    o3d.io.write_point_cloud(output_path, combined)

    print(f"[INFO] Saved combined visualization cloud: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Register CAD proxy point cloud to segmented target cluster using FPFH/RANSAC + point-to-plane ICP."
    )

    parser.add_argument("--cad", required=True, help="CAD proxy point cloud .pcd/.ply")
    parser.add_argument("--target", required=True, help="Target segmented cluster .pcd/.ply")
    parser.add_argument("--output", required=True, help="Output folder")

    parser.add_argument("--cad_scale", type=float, default=1.0, help="Scale CAD before registration. Keep 1.0 if proxy is already in meters.")

    parser.add_argument("--voxel", type=float, default=0.03, help="Registration voxel size")
    parser.add_argument("--normal_radius_factor", type=float, default=2.5, help="normal_radius = voxel * factor")
    parser.add_argument("--fpfh_radius_factor", type=float, default=5.0, help="fpfh_radius = voxel * factor")

    parser.add_argument("--normal_max_nn", type=int, default=40)
    parser.add_argument("--fpfh_max_nn", type=int, default=100)

    parser.add_argument("--ransac_dist_factor", type=float, default=2.0, help="RANSAC distance = voxel * factor")
    parser.add_argument("--ransac_n", type=int, default=3)
    parser.add_argument("--ransac_max_iter", type=int, default=100000)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)

    parser.add_argument("--icp_dist_factor", type=float, default=1.5, help="ICP distance = voxel * factor")
    parser.add_argument("--icp_max_iter", type=int, default=100)

    parser.add_argument("--use_center_init", action="store_true", help="Move CAD center to target center before registration")
    parser.add_argument("--skip_ransac", action="store_true", help="Skip RANSAC and run ICP from center initialization")

    args = parser.parse_args()

    make_dir(args.output)

    normal_radius = args.voxel * args.normal_radius_factor
    fpfh_radius = args.voxel * args.fpfh_radius_factor
    ransac_dist = args.voxel * args.ransac_dist_factor
    icp_dist = args.voxel * args.icp_dist_factor

    print("[INFO] Parameters:")
    print(f"  voxel: {args.voxel}")
    print(f"  normal_radius: {normal_radius}")
    print(f"  fpfh_radius: {fpfh_radius}")
    print(f"  ransac_dist: {ransac_dist}")
    print(f"  icp_dist: {icp_dist}")

    cad_original = load_point_cloud(args.cad, "CAD")
    target = load_point_cloud(args.target, "target")

    if args.cad_scale != 1.0:
        print(f"[INFO] Scaling CAD by: {args.cad_scale}")
        cad_original.scale(args.cad_scale, center=(0.0, 0.0, 0.0))

    o3d.io.write_point_cloud(os.path.join(args.output, "cad_original_input.pcd"), cad_original)
    o3d.io.write_point_cloud(os.path.join(args.output, "target_original_input.pcd"), target)

    if args.use_center_init:
        cad_for_registration, T_center = center_cad_to_target(cad_original, target)
    else:
        cad_for_registration = copy.deepcopy(cad_original)
        T_center = np.eye(4)

    cad_down, cad_fpfh = preprocess_point_cloud(
        cad_for_registration,
        voxel_size=args.voxel,
        normal_radius=normal_radius,
        normal_max_nn=args.normal_max_nn,
        fpfh_radius=fpfh_radius,
        fpfh_max_nn=args.fpfh_max_nn,
        name="CAD"
    )

    target_down, target_fpfh = preprocess_point_cloud(
        target,
        voxel_size=args.voxel,
        normal_radius=normal_radius,
        normal_max_nn=args.normal_max_nn,
        fpfh_radius=fpfh_radius,
        fpfh_max_nn=args.fpfh_max_nn,
        name="target"
    )

    o3d.io.write_point_cloud(os.path.join(args.output, "cad_downsampled_for_registration.pcd"), cad_down)
    o3d.io.write_point_cloud(os.path.join(args.output, "target_downsampled_for_registration.pcd"), target_down)

    if args.skip_ransac:
        print("[INFO] Skipping RANSAC.")
        T_ransac = np.eye(4)
        ransac_data = {
            "skipped": True,
            "fitness": None,
            "inlier_rmse": None,
            "time_s": None,
            "transformation": T_ransac.tolist()
        }
    else:
        ransac_result, ransac_time = run_ransac(
            source_down=cad_down,
            target_down=target_down,
            source_fpfh=cad_fpfh,
            target_fpfh=target_fpfh,
            distance_threshold=ransac_dist,
            ransac_n=args.ransac_n,
            max_iter=args.ransac_max_iter,
            confidence=args.ransac_confidence
        )

        T_ransac = ransac_result.transformation

        ransac_data = {
            "skipped": False,
            "fitness": float(ransac_result.fitness),
            "inlier_rmse": float(ransac_result.inlier_rmse),
            "time_s": float(ransac_time),
            "transformation": T_ransac.tolist()
        }

    icp_result, icp_time = run_icp(
        source_down=cad_down,
        target_down=target_down,
        init_transform=T_ransac,
        distance_threshold=icp_dist,
        max_iter=args.icp_max_iter
    )

    T_icp_from_registration_input = icp_result.transformation

    # Final transform from original CAD proxy frame to target scene frame.
    # If center initialization was used:
    # CAD_original -> T_center -> CAD_centered -> T_icp -> target
    T_scene_cad = T_icp_from_registration_input @ T_center

    cad_registered = copy.deepcopy(cad_original)
    cad_registered.transform(T_scene_cad)

    cad_registered_path = os.path.join(args.output, "cad_registered.pcd")
    o3d.io.write_point_cloud(cad_registered_path, cad_registered)
    print(f"[INFO] Saved registered CAD: {cad_registered_path}")

    combined_path = os.path.join(args.output, "combined_target_and_registered_cad.pcd")
    save_combined_cloud(target, cad_registered, combined_path)

    save_matrix(os.path.join(args.output, "T_scene_cad.txt"), T_scene_cad)
    save_matrix(os.path.join(args.output, "T_center_init.txt"), T_center)
    save_matrix(os.path.join(args.output, "T_ransac.txt"), T_ransac)
    save_matrix(os.path.join(args.output, "T_icp_from_registration_input.txt"), T_icp_from_registration_input)

    eval_data = evaluate(
        source_down=cad_down,
        target_down=target_down,
        transform=T_icp_from_registration_input,
        distance_threshold=icp_dist
    )

    summary = {
        "cad_file": args.cad,
        "target_file": args.target,
        "output": args.output,
        "cad_scale": args.cad_scale,
        "voxel": args.voxel,
        "normal_radius": normal_radius,
        "fpfh_radius": fpfh_radius,
        "ransac_dist": ransac_dist,
        "icp_dist": icp_dist,
        "use_center_init": bool(args.use_center_init),
        "skip_ransac": bool(args.skip_ransac),
        "ransac": ransac_data,
        "icp": {
            "fitness": float(icp_result.fitness),
            "inlier_rmse": float(icp_result.inlier_rmse),
            "time_s": float(icp_time),
            "transformation_from_registration_input": T_icp_from_registration_input.tolist()
        },
        "final_evaluation": eval_data,
        "T_scene_cad": T_scene_cad.tolist(),
        "outputs": {
            "registered_cad": "cad_registered.pcd",
            "combined_visualization": "combined_target_and_registered_cad.pcd",
            "matrix": "T_scene_cad.txt"
        }
    }

    save_json(os.path.join(args.output, "registration_summary.json"), summary)

    print("")
    print("[DONE] CAD-to-cluster registration finished.")
    print(f"[DONE] Final transform: {os.path.join(args.output, 'T_scene_cad.txt')}")
    print(f"[DONE] Registered CAD: {cad_registered_path}")
    print(f"[DONE] Combined visualization: {combined_path}")


if __name__ == "__main__":
    main()
