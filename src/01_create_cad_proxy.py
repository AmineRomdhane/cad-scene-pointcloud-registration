import argparse
import json
import os
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(
        description="Convert a very large CAD mesh to a lighter downsampled CAD proxy point cloud."
    )

    parser.add_argument("--input", required=True, help="Input CAD mesh file: .ply/.stl/.obj")
    parser.add_argument("--output", required=True, help="Output proxy point cloud: .pcd/.ply")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor. Use 0.001 if CAD is in mm and scene is in meters.")
    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size after scaling")
    parser.add_argument("--max_points", type=int, default=200000, help="Maximum points after downsampling. 0 disables random limit.")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading CAD mesh: {args.input}")
    mesh = o3d.io.read_triangle_mesh(args.input)

    if mesh.is_empty():
        raise RuntimeError(f"Mesh is empty or could not be loaded: {args.input}")

    print(f"[INFO] Mesh vertices: {len(mesh.vertices)}")
    print(f"[INFO] Mesh triangles: {len(mesh.triangles)}")

    bbox_before = mesh.get_axis_aligned_bounding_box()
    original_extent = bbox_before.get_extent()
    original_center = bbox_before.get_center()

    print(f"[INFO] Original extent xyz: {original_extent}")
    print(f"[INFO] Original center xyz: {original_center}")

    print("[INFO] Using mesh vertices as CAD proxy points...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    pcd.remove_non_finite_points()

    if args.scale != 1.0:
        print(f"[INFO] Applying scale: {args.scale}")
        pcd.scale(args.scale, center=(0.0, 0.0, 0.0))

    bbox_scaled = pcd.get_axis_aligned_bounding_box()
    print(f"[INFO] Scaled extent xyz: {bbox_scaled.get_extent()}")
    print(f"[INFO] Scaled center xyz: {bbox_scaled.get_center()}")

    if args.voxel > 0:
        print(f"[INFO] Voxel downsampling with voxel size: {args.voxel}")
        pcd = pcd.voxel_down_sample(args.voxel)

    print(f"[INFO] Points after voxel downsampling: {len(pcd.points)}")

    if args.max_points > 0 and len(pcd.points) > args.max_points:
        print(f"[INFO] Randomly limiting to max_points: {args.max_points}")
        indices = np.random.default_rng(42).choice(
            len(pcd.points),
            size=args.max_points,
            replace=False
        )
        pcd = pcd.select_by_index(indices.tolist())

    print(f"[INFO] Final proxy points: {len(pcd.points)}")

    bbox_final = pcd.get_axis_aligned_bounding_box()
    final_extent = bbox_final.get_extent()
    final_center = bbox_final.get_center()

    print(f"[INFO] Final extent xyz: {final_extent}")
    print(f"[INFO] Final center xyz: {final_center}")

    o3d.io.write_point_cloud(args.output, pcd)
    print(f"[DONE] Saved CAD proxy point cloud: {args.output}")

    summary = {
        "input": args.input,
        "output": args.output,
        "scale": args.scale,
        "voxel": args.voxel,
        "max_points": args.max_points,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
        "original_extent_xyz": original_extent.tolist(),
        "original_center_xyz": original_center.tolist(),
        "final_points": int(len(pcd.points)),
        "final_extent_xyz": final_extent.tolist(),
        "final_center_xyz": final_center.tolist()
    }

    summary_path = args.output + "_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[DONE] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
