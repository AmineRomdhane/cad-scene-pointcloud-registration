import argparse
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CAD mesh .ply")
    parser.add_argument("--output", required=True, help="Output sampled CAD proxy .pcd")
    parser.add_argument("--samples", type=int, default=300000, help="Number of surface points sampled before voxel downsampling")
    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size in meters")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor. Use 1.0 if already in meters")
    args = parser.parse_args()

    print(f"[INFO] Reading CAD mesh: {args.input}")
    mesh = o3d.io.read_triangle_mesh(args.input)

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Input file was not read as a valid triangle mesh.")

    if args.scale != 1.0:
        print(f"[INFO] Scaling mesh by {args.scale}")
        mesh.scale(args.scale, center=(0, 0, 0))

    print("[INFO] Mesh")
    print("vertices:", len(mesh.vertices))
    print("triangles:", len(mesh.triangles))
    print("extent:", mesh.get_axis_aligned_bounding_box().get_extent())
    print("center:", mesh.get_axis_aligned_bounding_box().get_center())

    print(f"[INFO] Sampling {args.samples} points uniformly on mesh surface")
    pcd = mesh.sample_points_uniformly(number_of_points=args.samples)

    print("[INFO] Sampled surface cloud before voxel")
    print("points:", len(pcd.points))
    print("extent:", pcd.get_axis_aligned_bounding_box().get_extent())
    print("center:", pcd.get_axis_aligned_bounding_box().get_center())

    print(f"[INFO] Voxel downsampling with voxel={args.voxel}")
    pcd = pcd.voxel_down_sample(args.voxel)

    print("[INFO] Estimating normals")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.voxel * 5,
            max_nn=30
        )
    )

    print("[INFO] Final proxy")
    print("points:", len(pcd.points))
    print("extent:", pcd.get_axis_aligned_bounding_box().get_extent())
    print("center:", pcd.get_axis_aligned_bounding_box().get_center())

    o3d.io.write_point_cloud(args.output, pcd)
    print(f"[OK] Saved: {args.output}")


if __name__ == "__main__":
    main()
