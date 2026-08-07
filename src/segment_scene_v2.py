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


def classify_plane_orientation(plane_model, horizontal_threshold=0.85, vertical_threshold=0.25):
    """
    Plane equation: ax + by + cz + d = 0
    Normal = [a, b, c]

    If abs(c) is high, plane normal is close to z-axis:
        floor / ceiling / horizontal surfaces

    If abs(c) is low, plane normal is mostly horizontal:
        wall / vertical surface
    """
    normal = np.asarray(plane_model[:3], dtype=float)
    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-9:
        return "unknown"

    normal = normal / normal_norm
    abs_z = abs(normal[2])

    if abs_z >= horizontal_threshold:
        return "horizontal"

    if abs_z <= vertical_threshold:
        return "vertical"

    return "inclined"


def remove_structural_planes(
    pcd,
    distance_threshold,
    ransac_n,
    num_iterations,
    max_planes,
    min_plane_ratio,
    min_plane_points,
    output_dir,
    remove_horizontal,
    remove_vertical,
    remove_inclined
):
    """
    Stronger plane removal for trial 2.

    Difference from code 1:
    - Ratio is computed against the remaining cloud, not the original cloud.
    - A plane can be accepted using min_plane_points.
    - Removed planes are saved separately for visual checking.
    - Plane orientation is estimated and saved.
    """

    remaining = pcd
    removed_planes = []
    removed_plane_clouds = []

    for i in range(max_planes):
        remaining_count = len(remaining.points)

        if remaining_count < min_plane_points:
            print("[INFO] Remaining cloud too small. Stop plane removal.")
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        inlier_count = len(inliers)
        ratio_remaining = inlier_count / remaining_count
        orientation = classify_plane_orientation(plane_model)

        print(
            f"[INFO] Plane {i}: "
            f"inliers={inlier_count}, "
            f"ratio_remaining={ratio_remaining:.3f}, "
            f"orientation={orientation}, "
            f"model={plane_model}"
        )

        if inlier_count < min_plane_points:
            print("[INFO] Plane has too few points. Stop plane removal.")
            break

        if ratio_remaining < min_plane_ratio:
            print("[INFO] Plane ratio too small. Stop plane removal.")
            break

        should_remove = False

        if orientation == "horizontal" and remove_horizontal:
            should_remove = True

        if orientation == "vertical" and remove_vertical:
            should_remove = True

        if orientation == "inclined" and remove_inclined:
            should_remove = True

        if not should_remove:
            print(f"[INFO] Plane {i} kept because orientation '{orientation}' is not selected for removal.")
            break

        plane_cloud = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)

        plane_path = os.path.join(output_dir, f"removed_plane_{i:02d}_{orientation}.pcd")
        o3d.io.write_point_cloud(plane_path, plane_cloud)
        print(f"[INFO] Saved removed plane: {plane_path}")

        removed_plane_clouds.append(plane_cloud)

        removed_planes.append({
            "index": i,
            "model": [float(x) for x in plane_model],
            "inliers": int(inlier_count),
            "ratio_remaining": float(ratio_remaining),
            "orientation": orientation,
            "file": os.path.basename(plane_path)
        })

    if len(removed_plane_clouds) > 0:
        combined = removed_plane_clouds[0]

        for pc in removed_plane_clouds[1:]:
            combined += pc

        combined_path = os.path.join(output_dir, "removed_planes_combined.pcd")
        o3d.io.write_point_cloud(combined_path, combined)
        print(f"[INFO] Saved combined removed planes: {combined_path}")

    print(f"[INFO] After plane removal: {len(remaining.points)}")

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
        description="Code 2: stronger geometric object segmentation with stronger plane removal."
    )

    parser.add_argument("--input", required=True, help="Input scene point cloud: .pcd/.ply/.xyz")
    parser.add_argument("--output", default="segmented_scene_trial2", help="Output directory")

    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size in meters")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Statistical outlier removal neighbors")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Statistical outlier removal std ratio")

    parser.add_argument("--plane_dist", type=float, default=0.03, help="RANSAC plane distance threshold")
    parser.add_argument("--ransac_n", type=int, default=3, help="RANSAC points per sample")
    parser.add_argument("--plane_iter", type=int, default=1000, help="RANSAC iterations")
    parser.add_argument("--max_planes", type=int, default=10, help="Maximum structural planes to remove")

    parser.add_argument(
        "--min_plane_ratio",
        type=float,
        default=0.015,
        help="Minimum plane ratio against remaining cloud"
    )

    parser.add_argument(
        "--min_plane_points",
        type=int,
        default=3000,
        help="Minimum number of plane points required to remove a plane"
    )

    parser.add_argument(
        "--remove_horizontal",
        action="store_true",
        help="Remove horizontal planes such as floor/ceiling/tabletops"
    )

    parser.add_argument(
        "--remove_vertical",
        action="store_true",
        help="Remove vertical planes such as walls/background panels"
    )

    parser.add_argument(
        "--remove_inclined",
        action="store_true",
        help="Remove inclined planes"
    )

    parser.add_argument("--eps", type=float, default=0.10, help="DBSCAN radius in meters")
    parser.add_argument("--min_points", type=int, default=30, help="DBSCAN minimum points")

    parser.add_argument("--min_cluster_points", type=int, default=100, help="Minimum points per saved cluster")
    parser.add_argument("--min_extent", type=float, default=0.05, help="Minimum x/y/z extent in meters")
    parser.add_argument(
        "--max_extent",
        type=float,
        default=0.0,
        help="Maximum x/y/z extent in meters. 0 disables this filter."
    )

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

    objects_cloud, removed_planes = remove_structural_planes(
        clean,
        distance_threshold=args.plane_dist,
        ransac_n=args.ransac_n,
        num_iterations=args.plane_iter,
        max_planes=args.max_planes,
        min_plane_ratio=args.min_plane_ratio,
        min_plane_points=args.min_plane_points,
        output_dir=args.output,
        remove_horizontal=args.remove_horizontal,
        remove_vertical=args.remove_vertical,
        remove_inclined=args.remove_inclined
    )

    objects_path = os.path.join(args.output, "objects_after_plane_removal.pcd")
    o3d.io.write_point_cloud(objects_path, objects_cloud)
    print(f"[INFO] Saved object cloud after plane removal: {objects_path}")

    planes_path = os.path.join(args.output, "removed_planes.json")

    with open(planes_path, "w") as f:
        json.dump(removed_planes, f, indent=4)

    print(f"[INFO] Saved removed plane summary: {planes_path}")

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