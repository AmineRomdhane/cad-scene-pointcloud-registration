import argparse
import copy
import json
import os
import numpy as np
import open3d as o3d


def load_pcd(path, name):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"{name} is empty or could not be loaded: {path}")
    pcd.remove_non_finite_points()
    print(f"[INFO] Loaded {name}: {path}")
    print(f"[INFO] {name} points: {len(pcd.points)}")
    print(f"[INFO] {name} extent: {pcd.get_axis_aligned_bounding_box().get_extent()}")
    print(f"[INFO] {name} center: {pcd.get_axis_aligned_bounding_box().get_center()}")
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
    print(f"[INFO] Saved combined cloud: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Refine CAD registration using an existing transform matrix and ICP only."
    )

    parser.add_argument("--cad", required=True, help="CAD proxy point cloud")
    parser.add_argument("--target", required=True, help="Target scan cluster")
    parser.add_argument("--init", required=True, help="Initial transform matrix T_scene_cad.txt")
    parser.add_argument("--output", required=True, help="Output directory")

    parser.add_argument("--voxel", type=float, default=0.025)
    parser.add_argument("--normal_radius_factor", type=float, default=3.0)
    parser.add_argument("--icp_dist_factor", type=float, default=1.5)
    parser.add_argument("--normal_max_nn", type=int, default=50)
    parser.add_argument("--icp_max_iter", type=int, default=100)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cad = load_pcd(args.cad, "CAD")
    target = load_pcd(args.target, "target")

    T_init = np.loadtxt(args.init)
    print("[INFO] Initial transform:")
    print(T_init)

    normal_radius = args.voxel * args.normal_radius_factor
    icp_dist = args.voxel * args.icp_dist_factor

    print(f"[INFO] voxel: {args.voxel}")
    print(f"[INFO] normal_radius: {normal_radius}")
    print(f"[INFO] icp_dist: {icp_dist}")

    cad_down = preprocess(cad, args.voxel, normal_radius, args.normal_max_nn, "CAD")
    target_down = preprocess(target, args.voxel, normal_radius, args.normal_max_nn, "target")

    print("[INFO] Running ICP refinement from existing transform...")

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

    T_refined = result.transformation

    print("[INFO] ICP refined transform:")
    print(T_refined)
    print(f"[INFO] ICP fitness: {result.fitness}")
    print(f"[INFO] ICP RMSE: {result.inlier_rmse}")

    cad_registered = copy.deepcopy(cad)
    cad_registered.transform(T_refined)

    o3d.io.write_point_cloud(os.path.join(args.output, "cad_registered_refined.pcd"), cad_registered)
    o3d.io.write_point_cloud(os.path.join(args.output, "target_input.pcd"), target)

    combined_path = os.path.join(args.output, "combined_target_and_registered_cad_refined.pcd")
    save_combined(target, cad_registered, combined_path)

    np.savetxt(os.path.join(args.output, "T_scene_cad_refined.txt"), T_refined, fmt="%.10f")

    summary = {
        "cad": args.cad,
        "target": args.target,
        "init": args.init,
        "voxel": args.voxel,
        "normal_radius": normal_radius,
        "icp_dist": icp_dist,
        "icp_fitness": float(result.fitness),
        "icp_rmse": float(result.inlier_rmse),
        "T_scene_cad_refined": T_refined.tolist()
    }

    with open(os.path.join(args.output, "refinement_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("[DONE] Refinement finished.")
    print(f"[DONE] Refined matrix: {os.path.join(args.output, 'T_scene_cad_refined.txt')}")
    print(f"[DONE] Visualization: {combined_path}")


if __name__ == "__main__":
    main()
