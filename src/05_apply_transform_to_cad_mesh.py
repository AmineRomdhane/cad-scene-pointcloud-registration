import argparse
import copy
import json
import os
import numpy as np
import open3d as o3d


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)

    if mesh.is_empty():
        raise RuntimeError(f"Could not load CAD mesh or mesh is empty: {path}")

    print(f"[INFO] Loaded CAD mesh: {path}")
    print(f"[INFO] vertices: {len(mesh.vertices)}")
    print(f"[INFO] triangles: {len(mesh.triangles)}")

    bbox = mesh.get_axis_aligned_bounding_box()
    print(f"[INFO] original extent: {bbox.get_extent()}")
    print(f"[INFO] original center: {bbox.get_center()}")

    return mesh


def load_transform(path):
    T = np.loadtxt(path)

    if T.shape != (4, 4):
        raise RuntimeError(f"Transform must be 4x4, got shape {T.shape}")

    print(f"[INFO] Loaded transform: {path}")
    print(T)

    return T


def save_combined_scan_and_mesh(scan_path, mesh_registered, output_path):
    scan = o3d.io.read_point_cloud(scan_path)

    if scan.is_empty():
        print(f"[WARNING] Scan cloud empty or not found: {scan_path}")
        return

    scan.paint_uniform_color([0.0, 0.7, 1.0])
    mesh_registered.paint_uniform_color([1.0, 0.2, 0.0])

    combined = [scan, mesh_registered]

    o3d.io.write_triangle_mesh(output_path, mesh_registered)
    print(f"[INFO] Saved registered mesh for visualization: {output_path}")

    print("[INFO] Opening visualization: scan + registered CAD mesh")
    o3d.visualization.draw_geometries(combined)


def main():
    parser = argparse.ArgumentParser(
        description="Apply a registration transform to the original CAD mesh."
    )

    parser.add_argument("--cad_mesh", required=True, help="Original CAD mesh .ply/.stl/.obj")
    parser.add_argument("--transform", required=True, help="T_scene_cad.txt or T_scene_cad_refined.txt")
    parser.add_argument("--output", required=True, help="Output registered CAD mesh .ply")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale CAD mesh before applying transform. Keep 1.0 if CAD is already in meters.")
    parser.add_argument("--scan", default=None, help="Optional scan/cluster cloud for visualization")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mesh = load_mesh(args.cad_mesh)
    T = load_transform(args.transform)

    mesh_registered = copy.deepcopy(mesh)

    if args.scale != 1.0:
        print(f"[INFO] Applying CAD scale before transform: {args.scale}")
        mesh_registered.scale(args.scale, center=(0.0, 0.0, 0.0))

    print("[INFO] Applying transform to CAD mesh...")
    mesh_registered.transform(T)
    mesh_registered.compute_vertex_normals()

    bbox = mesh_registered.get_axis_aligned_bounding_box()
    print(f"[INFO] registered extent: {bbox.get_extent()}")
    print(f"[INFO] registered center: {bbox.get_center()}")

    o3d.io.write_triangle_mesh(args.output, mesh_registered)
    print(f"[DONE] Saved registered CAD mesh: {args.output}")

    summary = {
        "cad_mesh": args.cad_mesh,
        "transform": args.transform,
        "output": args.output,
        "scale": args.scale,
        "registered_extent_xyz": bbox.get_extent().tolist(),
        "registered_center_xyz": bbox.get_center().tolist(),
        "T_scene_cad": T.tolist()
    }

    summary_path = args.output + "_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[DONE] Saved summary: {summary_path}")

    if args.scan is not None:
        save_combined_scan_and_mesh(
            scan_path=args.scan,
            mesh_registered=mesh_registered,
            output_path=args.output
        )


if __name__ == "__main__":
    main()
