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
    normal = np.asarray(plane_model[:3], dtype=float)
    norm = np.linalg.norm(normal)

    if norm < 1e-9:
        return False

    normal = normal / norm

    return abs(normal[2]) >= horizontal_threshold


def normalize_plane(plane_model):
    plane = np.asarray(plane_model, dtype=float)
    normal = plane[:3]
    d = plane[3]

    norm = np.linalg.norm(normal)

    if norm < 1e-9:
        raise RuntimeError("Invalid plane normal.")

    normal = normal / norm
    d = d / norm

    return normal, d


def find_floor_plane_candidate(
    pcd,
    distance_threshold,
    ransac_n,
    num_iterations,
    max_plane_candidates,
    min_plane_ratio,
    min_plane_points,
    horizontal_threshold,
    floor_zone_limit,
    min_xy_extent,
    output_dir
):
    """
    Searches for a low horizontal floor plane.

    Important improvement:
    If RANSAC finds a large non-floor plane first, we temporarily remove it
    from the search cloud and continue searching. We do NOT remove it from
    the final cloud. This helps find the floor even when another object
    surface is dominant.
    """

    search_cloud = pcd
    skipped_candidates = []

    for i in range(max_plane_candidates):
        search_count = len(search_cloud.points)

        if search_count < min_plane_points:
            print("[INFO] Search cloud too small. Stop floor search.")
            break

        plane_model, inliers = search_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        inlier_count = len(inliers)
        ratio_search = inlier_count / search_count

        candidate_cloud = search_cloud.select_by_index(inliers)
        candidate_points = np.asarray(candidate_cloud.points)

        plane_z_median = float(np.median(candidate_points[:, 2]))

        aabb = candidate_cloud.get_axis_aligned_bounding_box()
        extent = np.asarray(aabb.get_extent())

        horizontal = is_horizontal_plane(plane_model, horizontal_threshold)
        near_floor = plane_z_median <= floor_zone_limit
        xy_extent_ok = extent[0] >= min_xy_extent and extent[1] >= min_xy_extent

        print(
            f"[INFO] Floor candidate {i}: "
            f"inliers={inlier_count}, "
            f"ratio_search={ratio_search:.3f}, "
            f"z_median={plane_z_median:.3f}, "
            f"extent={extent}, "
            f"horizontal={horizontal}, "
            f"near_floor={near_floor}, "
            f"xy_extent_ok={xy_extent_ok}, "
            f"model={plane_model}"
        )

        candidate_path = os.path.join(output_dir, f"floor_candidate_{i:02d}.pcd")
        o3d.io.write_point_cloud(candidate_path, candidate_cloud)

        if inlier_count < min_plane_points:
            print("[INFO] Candidate has too few points. Stop floor search.")
            break

        if ratio_search < min_plane_ratio:
            print("[INFO] Candidate ratio too small. Stop floor search.")
            break

        if horizontal and near_floor and xy_extent_ok:
            print(f"[INFO] Accepted floor candidate {i}.")
            return plane_model, candidate_cloud, {
                "candidate_index": i,
                "model": [float(x) for x in plane_model],
                "inliers": int(inlier_count),
                "ratio_search": float(ratio_search),
                "z_median": float(plane_z_median),
                "extent_xyz": extent.tolist(),
                "candidate_file": os.path.basename(candidate_path)
            }

        print("[INFO] Candidate rejected for floor. Temporarily skipping it in floor search only.")

        skipped_candidates.append({
            "candidate_index": i,
            "model": [float(x) for x in plane_model],
            "inliers": int(inlier_count),
            "ratio_search": float(ratio_search),
            "z_median": float(plane_z_median),
            "extent_xyz": extent.tolist(),
            "horizontal": bool(horizontal),
            "near_floor": bool(near_floor),
            "xy_extent_ok": bool(xy_extent_ok),
            "candidate_file": os.path.basename(candidate_path)
        })

        search_cloud = search_cloud.select_by_index(inliers, invert=True)

    print("[WARNING] No floor plane candidate accepted.")
    return None, None, {
        "candidate_index": None,
        "model": None,
        "message": "No floor plane accepted.",
        "skipped_candidates": skipped_candidates
    }


def remove_floor_strong(
    pcd,
    distance_threshold,
    ransac_n,
    num_iterations,
    max_plane_candidates,
    min_plane_ratio,
    min_plane_points,
    horizontal_threshold,
    floor_percentile,
    floor_z_margin,
    floor_plane_thickness,
    floor_z_band_height,
    min_xy_extent,
    disable_z_band,
    output_dir
):
    """
    Code 5:
    Stronger floor removal.

    It removes:
    1. Points close to the detected floor plane.
    2. Points inside a low-Z floor band.

    This is useful when the floor is not perfectly planar or when RANSAC
    removes only part of it.
    """

    points = np.asarray(pcd.points)

    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))
    z_floor_ref = float(np.percentile(points[:, 2], floor_percentile))

    floor_zone_limit = z_floor_ref + floor_z_margin
    floor_band_limit = z_floor_ref + floor_z_band_height

    print(f"[INFO] Scene Z min: {z_min:.3f}")
    print(f"[INFO] Scene Z max: {z_max:.3f}")
    print(f"[INFO] Floor reference Z percentile {floor_percentile}%: {z_floor_ref:.3f}")
    print(f"[INFO] Floor search zone: z <= {floor_zone_limit:.3f}")
    print(f"[INFO] Floor Z-band removal: z <= {floor_band_limit:.3f}")

    floor_model, floor_candidate_cloud, floor_info = find_floor_plane_candidate(
        pcd=pcd,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        max_plane_candidates=max_plane_candidates,
        min_plane_ratio=min_plane_ratio,
        min_plane_points=min_plane_points,
        horizontal_threshold=horizontal_threshold,
        floor_zone_limit=floor_zone_limit,
        min_xy_extent=min_xy_extent,
        output_dir=output_dir
    )

    remove_mask = np.zeros(len(points), dtype=bool)

    if floor_model is not None:
        normal, d = normalize_plane(floor_model)
        signed_distances = points @ normal + d
        abs_distances = np.abs(signed_distances)

        plane_mask = (abs_distances <= floor_plane_thickness) & (points[:, 2] <= floor_zone_limit)

        remove_mask = remove_mask | plane_mask

        print(f"[INFO] Points removed by floor plane thickness: {int(np.sum(plane_mask))}")
    else:
        print("[WARNING] No floor plane detected. Plane-distance removal skipped.")

    if not disable_z_band:
        z_band_mask = points[:, 2] <= floor_band_limit
        remove_mask = remove_mask | z_band_mask
        print(f"[INFO] Points removed by low-Z band: {int(np.sum(z_band_mask))}")
    else:
        print("[INFO] Low-Z band removal disabled.")

    removed_indices = np.where(remove_mask)[0]
    kept_indices = np.where(~remove_mask)[0]

    removed_floor = pcd.select_by_index(removed_indices)
    remaining = pcd.select_by_index(kept_indices)

    removed_floor_path = os.path.join(output_dir, "removed_floor_strong.pcd")
    remaining_path = os.path.join(output_dir, "objects_after_floor_removal.pcd")

    o3d.io.write_point_cloud(removed_floor_path, removed_floor)
    o3d.io.write_point_cloud(remaining_path, remaining)

    print(f"[INFO] Removed floor points total: {len(removed_indices)}")
    print(f"[INFO] Remaining points after floor removal: {len(remaining.points)}")
    print(f"[INFO] Saved strong removed floor: {removed_floor_path}")
    print(f"[INFO] Saved remaining cloud: {remaining_path}")

    floor_summary = {
        "z_min": z_min,
        "z_max": z_max,
        "floor_percentile": floor_percentile,
        "z_floor_ref": z_floor_ref,
        "floor_zone_limit": floor_zone_limit,
        "floor_band_limit": floor_band_limit,
        "floor_plane_thickness": floor_plane_thickness,
        "floor_z_band_height": floor_z_band_height,
        "disable_z_band": bool(disable_z_band),
        "removed_points": int(len(removed_indices)),
        "remaining_points": int(len(remaining.points)),
        "floor_info": floor_info
    }

    return remaining, removed_floor, floor_summary


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
        description="Code 5: stronger floor removal using floor plane + low-Z band, then DBSCAN clustering."
    )

    parser.add_argument("--input", required=True, help="Input scene point cloud: .pcd/.ply/.xyz")
    parser.add_argument("--output", default="segmented_scene_trial5_floor_strong", help="Output directory")

    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size in meters")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Statistical outlier removal neighbors")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Statistical outlier removal std ratio")

    parser.add_argument("--plane_dist", type=float, default=0.03, help="RANSAC floor candidate distance threshold")
    parser.add_argument("--ransac_n", type=int, default=3, help="RANSAC points per sample")
    parser.add_argument("--plane_iter", type=int, default=1000, help="RANSAC iterations")
    parser.add_argument("--max_plane_candidates", type=int, default=10, help="Maximum candidate planes to test during floor search")

    parser.add_argument("--min_plane_ratio", type=float, default=0.005, help="Minimum plane ratio inside floor search cloud")
    parser.add_argument("--min_plane_points", type=int, default=1500, help="Minimum number of points for floor plane candidate")
    parser.add_argument("--horizontal_threshold", type=float, default=0.85, help="abs(normal_z) threshold for horizontal planes")

    parser.add_argument("--floor_percentile", type=float, default=1.0, help="Low percentile used as floor Z reference")
    parser.add_argument("--floor_z_margin", type=float, default=0.35, help="Floor search zone height above floor reference")
    parser.add_argument("--floor_plane_thickness", type=float, default=0.08, help="Remove points within this distance from detected floor plane")
    parser.add_argument("--floor_z_band_height", type=float, default=0.12, help="Remove all points below floor reference plus this height")
    parser.add_argument("--min_xy_extent", type=float, default=0.8, help="Minimum X and Y extent for accepted floor plane")

    parser.add_argument(
        "--disable_z_band",
        action="store_true",
        help="Disable low-Z band removal and use only plane-distance removal"
    )

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

    objects_cloud, removed_floor, floor_summary = remove_floor_strong(
        pcd=clean,
        distance_threshold=args.plane_dist,
        ransac_n=args.ransac_n,
        num_iterations=args.plane_iter,
        max_plane_candidates=args.max_plane_candidates,
        min_plane_ratio=args.min_plane_ratio,
        min_plane_points=args.min_plane_points,
        horizontal_threshold=args.horizontal_threshold,
        floor_percentile=args.floor_percentile,
        floor_z_margin=args.floor_z_margin,
        floor_plane_thickness=args.floor_plane_thickness,
        floor_z_band_height=args.floor_z_band_height,
        min_xy_extent=args.min_xy_extent,
        disable_z_band=args.disable_z_band,
        output_dir=args.output
    )

    floor_summary_path = os.path.join(args.output, "removed_floor_summary.json")

    with open(floor_summary_path, "w") as f:
        json.dump(floor_summary, f, indent=4)

    print(f"[INFO] Saved floor removal summary: {floor_summary_path}")

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