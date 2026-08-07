import argparse
import json
import os
import numpy as np
import open3d as o3d


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)

    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh: {path}")

    if len(mesh.triangles) == 0:
        raise RuntimeError(f"Mesh has no triangles: {path}")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    return mesh


def mesh_extent_center(mesh):
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    center = np.asarray(bbox.get_center())
    return extent, center


def anisotropic_scale_mesh(mesh, sx, sy, sz, center_mode):
    vertices = np.asarray(mesh.vertices).copy()

    if center_mode == "center":
        _, center = mesh_extent_center(mesh)
    elif center_mode == "origin":
        center = np.array([0.0, 0.0, 0.0])
    else:
        raise ValueError("center_mode must be 'center' or 'origin'")

    print(f"[INFO] Scaling around: {center}")

    scale_vector = np.array([sx, sy, sz], dtype=np.float64)
    vertices_scaled = (vertices - center) * scale_vector + center

    mesh_scaled = o3d.geometry.TriangleMesh()
    mesh_scaled.vertices = o3d.utility.Vector3dVector(vertices_scaled)
    mesh_scaled.triangles = mesh.triangles
    mesh_scaled.compute_vertex_normals()

    return mesh_scaled


def sample_proxy(mesh, samples, voxel):
    print(f"[INFO] Sampling {samples} points uniformly on adjusted CAD surface...")

    pcd = mesh.sample_points_uniformly(number_of_points=samples)

    print(f"[INFO] Sampled points before voxel: {len(pcd.points)}")

    if voxel > 0:
        print(f"[INFO] Voxel downsampling with voxel={voxel}")
        pcd = pcd.voxel_down_sample(voxel)

    if len(pcd.points) == 0:
        raise RuntimeError("Proxy point cloud became empty after voxel downsampling.")

    normal_radius = max(voxel * 3.0, 0.05)

    print("[INFO] Estimating proxy normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=50
        )
    )
    pcd.normalize_normals()

    print(f"[INFO] Final proxy points: {len(pcd.points)}")

    return pcd


def main():
    parser = argparse.ArgumentParser(
        description="Anisotropically scale a CAD mesh and create a registration proxy point cloud."
    )

    parser.add_argument("--input", required=True, help="Input CAD mesh .ply/.stl/.obj")
    parser.add_argument("--output_mesh", required=True, help="Output adjusted CAD mesh .ply")
    parser.add_argument("--output_proxy", required=True, help="Output adjusted CAD proxy .pcd")

    parser.add_argument("--sx", type=float, required=True, help="Scale factor along CAD X")
    parser.add_argument("--sy", type=float, required=True, help="Scale factor along CAD Y")
    parser.add_argument("--sz", type=float, required=True, help="Scale factor along CAD Z")

    parser.add_argument(
        "--center_mode",
        choices=["center", "origin"],
        default="center",
        help="Scale around CAD AABB center or world origin"
    )

    parser.add_argument("--samples", type=int, default=300000, help="Surface samples before voxel downsampling")
    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size for final proxy")

    args = parser.parse_args()

    output_mesh_dir = os.path.dirname(args.output_mesh)
    output_proxy_dir = os.path.dirname(args.output_proxy)

    if output_mesh_dir:
        os.makedirs(output_mesh_dir, exist_ok=True)

    if output_proxy_dir:
        os.makedirs(output_proxy_dir, exist_ok=True)

    print(f"[INFO] Loading CAD mesh: {args.input}")
    mesh = load_mesh(args.input)

    original_extent, original_center = mesh_extent_center(mesh)

    print("[INFO] Original CAD")
    print(f"  vertices: {len(mesh.vertices)}")
    print(f"  triangles: {len(mesh.triangles)}")
    print(f"  extent: {original_extent}")
    print(f"  center: {original_center}")

    print("[INFO] Scale factors")
    print(f"  sx: {args.sx}")
    print(f"  sy: {args.sy}")
    print(f"  sz: {args.sz}")

    mesh_scaled = anisotropic_scale_mesh(
        mesh=mesh,
        sx=args.sx,
        sy=args.sy,
        sz=args.sz,
        center_mode=args.center_mode
    )

    adjusted_extent, adjusted_center = mesh_extent_center(mesh_scaled)

    print("[INFO] Adjusted CAD")
    print(f"  extent: {adjusted_extent}")
    print(f"  center: {adjusted_center}")

    print(f"[INFO] Saving adjusted mesh: {args.output_mesh}")
    o3d.io.write_triangle_mesh(args.output_mesh, mesh_scaled)

    proxy = sample_proxy(mesh_scaled, samples=args.samples, voxel=args.voxel)

    proxy_bbox = proxy.get_axis_aligned_bounding_box()
    proxy_extent = np.asarray(proxy_bbox.get_extent())
    proxy_center = np.asarray(proxy_bbox.get_center())

    print("[INFO] Proxy")
    print(f"  points: {len(proxy.points)}")
    print(f"  extent: {proxy_extent}")
    print(f"  center: {proxy_center}")

    print(f"[INFO] Saving adjusted proxy: {args.output_proxy}")
    o3d.io.write_point_cloud(args.output_proxy, proxy)

    summary = {
        "input": args.input,
        "output_mesh": args.output_mesh,
        "output_proxy": args.output_proxy,
        "center_mode": args.center_mode,
        "scale_factors": {
            "sx": float(args.sx),
            "sy": float(args.sy),
            "sz": float(args.sz)
        },
        "original_extent": original_extent.tolist(),
        "original_center": original_center.tolist(),
        "adjusted_extent": adjusted_extent.tolist(),
        "adjusted_center": adjusted_center.tolist(),
        "proxy_points": int(len(proxy.points)),
        "proxy_extent": proxy_extent.tolist(),
        "proxy_center": proxy_center.tolist(),
        "samples": int(args.samples),
        "voxel": float(args.voxel)
    }

    summary_path = args.output_proxy + "_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[OK] Saved adjusted mesh: {args.output_mesh}")
    print(f"[OK] Saved adjusted proxy: {args.output_proxy}")
    print(f"[OK] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
