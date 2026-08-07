import argparse
import os
import json
import numpy as np
import open3d as o3d


def load_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Could not load point cloud or point cloud is empty: {path}")
    return pcd


def preprocess_cloud(pcd, voxel_size, nb_neighbors, std_ratio):
    print(f"[INFO] Original points: {len(pcd.points)}")

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"[INFO] After voxel downsampling: {len(pcd.points)}")

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    print(f"[INFO] After statistical outlier removal: {len(pcd.points)}")

    return pcd


def is_horizontal_plane(plane_model, horizontal_threshold):
    """
    Plane equation: ax + by + cz + d = 0
    A horizontal plane has a normal close to the Z axis.
    """
    normal = np.asarray(plane_model[:3], dtype=float)
    norm = np.linalg.norm(normal)

    if norm < 1e-9:
        return False

    normal = normal / norm
    return abs(normal[2]) >= horizontal_threshold


def remove_floor_only(
    pcd,
    distance_threshold,
    ransac_n,
    num_iterations,
    max_planes,
    min_plane_ratio,
    min_plane_points,
    horizontal_threshold,
    z_margin,
    min_xy_extent,
    output_dir
):
    """
    Code 4:
    Remove only the floor.

    The ceiling is assumed to be already manually removed.

    A plane is removed only if:
    1. It is horizontal.
    2. It is close to the lowest Z value of the scene.
    3. It has enough points.
    4. Its XY extent is large enough.

    This avoids removing shelves, vitrine surfaces, tabletops, or object parts.
    """

    remaining = pcd
    removed_planes = []
    removed_plane_clouds = []

    all_points = np.asarray(pcd.points)
    scene_z_min = float(np.min(all_points[:, 2]))
    scene_z_max = float(np.max(all_points[:, 2]))

    floor_limit = scene_z_min + z_margin

    print(f"[INFO] Scene Z min: {scene_z_min:.3f}")
    print(f"[INFO] Scene Z max: {scene_z_max:.3f}")
    print(f"[INFO] Floor zone: z <= {floor_limit:.3f}")
    print("[INFO] Ceiling removal disabled because ceiling was manually removed.")

    for i in range(max_planes):
        remaining_count = len(remaining.points)

        if remaining_count < min_plane_points:
            print("[INFO] Remaining cloud too small. Stop floor removal.")
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        inlier_count = len(inliers)
        ratio_remaining = inlier_count / remaining_count

        plane_cloud = remaining.select_by_index(inliers)
        plane_points = np.asarray(plane_cloud.points)

        plane_z_median = float(np.median(plane_points[:, 2]))

        aabb = plane_cloud.get_axis_aligned_bounding_box()
        extent = np.asarray(aabb.get_extent())

        horizontal = is_horizontal_plane(plane_model, horizontal_threshold)
        near_floor = plane_z_median <= floor_limit
        xy_extent_ok = extent[0] >= min_xy_extent and extent[1] >= min_xy_extent

        print(
            f"[INFO] Plane {i}: "
            f"inliers={inlier_count}, "
            f"ratio_remaining={ratio_remaining:.3f}, "
            f"z_median={plane_z_median:.3f}, "
            f"extent={extent}, "
            f"horizontal={horizontal}, "
            f"near_floor={near_floor}, "
            f"xy_extent_ok={xy_extent_ok}, "
            f"model={plane_model}"
        )

        if inlier_count < min_plane_points:
            print("[INFO] Plane has too few points. Stop floor removal.")
            break

        if ratio_remaining < min_plane_ratio:
            print("[INFO] Plane ratio too small. Stop floor removal.")
            break

        if not horizontal:
            print("[INFO] Plane is not horizontal. Keep it and stop.")
            break

        if not near_floor:
            print("[INFO] Horizontal plane is not near floor. Probably object part. Keep it and stop.")
            break

        if not xy_extent_ok:
            print("[INFO] Plane XY extent too small. Probably object surface. Keep it and stop.")
            break

        remaining = remaining.select_by_index(inliers, invert=True)

        plane_path = os.path.join(output_dir, f"removed_floor_{i:02d}.pcd")
        o3d.io.write_point_cloud(plane_path, plane_cloud)
        print(f"[INFO] Removed floor candidate: {plane_path}")

        removed_plane_clouds.append(plane_cloud)

        removed_planes.append({
            "index": i,
            "type": "floor",
            "model": [float(x) for x in plane_model],
            "inliers": int(inlier_count),
            "ratio_remaining": float(ratio_remaining),
            "z_median": float(plane_z_median),
            "extent_xyz": extent.tolist(),
            "file": os.path.basename(plane_path)
        })

    if len(removed_plane_clouds) > 0:
        combined = removed_plane_clouds[0]

        for pc in removed_plane_clouds[1:]:
            combined += pc

        combined_path = os.path.join(output_dir, "removed_floor_combined.pcd")
        o3d.io.write_point_cloud(combined_path, combined)
        print(f"[INFO] Saved removed floor combined: {combined_path}")
    else:
        print("[INFO] No floor plane was removed.")

    print(f"[INFO] After floor removal: {len(remaining.points)}")

    return remaining, removed_planes


def cluster_dbscan(pcd, eps, min_points):
    print("[INFO] Running DBSCAN clustering...")

    labels = np.array(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=True
        )
    )

    if labels.size == 0:
        return labels, 0

    n_clusters = labels.max() + 1
    n_noise = int(np.sum(labels == -1))

    print(f"[INFO] DBSCAN clusters found: {n_clusters}")
    print(f"[INFO] DBSCAN noise points: {n_noise}")

    return labels, n_clusters


def save_colored_clusters(pcd, labels, output_path):
    labels = np.asarray(labels)

    if labels.size == 0:
        print("[INFO] No labels to save.")
        return

    max_label = labels.max()
    colors = np.zeros((len(labels), 3))

    if max_label >= 0:
        rng = np.random.default_rng(42)
        random_colors = rng.random((max_label + 1, 3))

        for i, label in enumerate(labels):
            if label == -1:
                colors[i] = [0.0, 0.0, 0.0]
            else:
                colors[i] = random_colors[label]

    colored = o3d.geometry.PointCloud()
    colored.points = pcd.points
    colored.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, colored)
    print(f"[INFO] Colored clustered cloud saved to: {output_path}")


def filter_and_save_clusters(
    pcd,
    labels,
    n_clusters,
    output_dir,
    min_cluster_points,
    min_extent,
    max_extent
):
    os.makedirs(output_dir, exist_ok=True)

    cluster_info = []
    saved_count = 0

    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        point_count = len(indices)

        if point_count < min_cluster_points:
            continue

        cluster = pcd.select_by_index(indices)

        aabb = cluster.get_axis_aligned_bounding_box()
        extent = np.asarray(aabb.get_extent())
        center = np.asarray(aabb.get_center())

        if np.any(extent < min_extent):
            continue

        if max_extent > 0 and np.any(extent > max_extent):
            continue

        cluster_name = f"cluster_{saved_count:03d}.pcd"
        cluster_path = os.path.join(output_dir, cluster_name)

        o3d.io.write_point_cloud(cluster_path, cluster)

        info = {
            "cluster_file": cluster_name,
            "original_dbscan_label": int(cluster_id),
            "num_points": int(point_count),
            "center_xyz": center.tolist(),
            "extent_xyz": extent.tolist(),
            "volume_aabb": float(extent[0] * extent[1] * extent[2])
        }

        cluster_info.append(info)

        print(
            f"[SAVED] {cluster_name} | "
            f"points={point_count} | "
            f"center={center} | "
            f"extent={extent}"
        )

        saved_count += 1

    summary_path = os.path.join(output_dir, "clusters_summary.json")

    with open(summary_path, "w") as f:
        json.dump(cluster_info, f, indent=4)

    print(f"[INFO] Saved clusters: {saved_count}")
    print(f"[INFO] Summary saved to: {summary_path}")

    return cluster_info


def main():
    parser = argparse.ArgumentParser(
        description="Code 4: object segmentation with floor-only plane removal."
    )

    parser.add_argument("--input", required=True, help="Input scene point cloud: .pcd/.ply/.xyz")
    parser.add_argument("--output", default="segmented_scene_trial4_floor_only", help="Output directory")

    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size in meters")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Statistical outlier removal neighbors")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Statistical outlier removal std ratio")

    parser.add_argument("--plane_dist", type=float, default=0.03, help="RANSAC plane distance threshold")
    parser.add_argument("--ransac_n", type=int, default=3, help="RANSAC points per sample")
    parser.add_argument("--plane_iter", type=int, default=1000, help="RANSAC iterations")
    parser.add_argument("--max_planes", type=int, default=4, help="Maximum floor candidates to test")

    parser.add_argument("--min_plane_ratio", type=float, default=0.015, help="Minimum plane ratio against remaining cloud")
    parser.add_argument("--min_plane_points", type=int, default=3000, help="Minimum number of plane points")
    parser.add_argument("--horizontal_threshold", type=float, default=0.90, help="abs(normal_z) threshold for horizontal planes")
    parser.add_argument("--z_margin", type=float, default=0.20, help="Distance from scene min Z to accept floor")
    parser.add_argument("--min_xy_extent", type=float, default=1.0, help="Minimum X and Y extent for floor plane")

    parser.add_argument("--eps", type=float, default=0.10, help="DBSCAN radius in meters")
    parser.add_argument("--min_points", type=int, default=30, help="DBSCAN minimum points")

    parser.add_argument("--min_cluster_points", type=int, default=100, help="Minimum points per saved cluster")
    parser.add_argument("--min_extent", type=float, default=0.05, help="Minimum x/y/z extent in meters")
    parser.add_argument("--max_extent", type=float, default=0.0, help="Maximum x/y/z extent in meters. 0 disables this filter.")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pcd = load_cloud(args.input)

    clean = preprocess_cloud(
        pcd,
        voxel_size=args.voxel,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio
    )

    clean_path = os.path.join(args.output, "clean_downsampled.pcd")
    o3d.io.write_point_cloud(clean_path, clean)
    print(f"[INFO] Saved clean cloud: {clean_path}")

    objects_cloud, removed_planes = remove_floor_only(
        clean,
        distance_threshold=args.plane_dist,
        ransac_n=args.ransac_n,
        num_iterations=args.plane_iter,
        max_planes=args.max_planes,
        min_plane_ratio=args.min_plane_ratio,
        min_plane_points=args.min_plane_points,
        horizontal_threshold=args.horizontal_threshold,
        z_margin=args.z_margin,
        min_xy_extent=args.min_xy_extent,
        output_dir=args.output
    )

    objects_path = os.path.join(args.output, "objects_after_floor_removal.pcd")
    o3d.io.write_point_cloud(objects_path, objects_cloud)
    print(f"[INFO] Saved object cloud after floor removal: {objects_path}")

    planes_path = os.path.join(args.output, "removed_floor.json")

    with open(planes_path, "w") as f:
        json.dump(removed_planes, f, indent=4)

    print(f"[INFO] Saved removed floor summary: {planes_path}")

    labels, n_clusters = cluster_dbscan(
        objects_cloud,
        eps=args.eps,
        min_points=args.min_points
    )

    colored_path = os.path.join(args.output, "clusters_colored.pcd")
    save_colored_clusters(objects_cloud, labels, colored_path)

    filter_and_save_clusters(
        objects_cloud,
        labels,
        n_clusters,
        output_dir=args.output,
        min_cluster_points=args.min_cluster_points,
        min_extent=args.min_extent,
        max_extent=args.max_extent
    )


if __name__ == "__main__":
    main()