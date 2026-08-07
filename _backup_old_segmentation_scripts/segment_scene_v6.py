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


def is_floor_like_plane(plane_model, horizontal_threshold):
    """
    Plane equation: ax + by + cz + d = 0

    A perfect horizontal floor has abs(normal_z) close to 1.
    A slightly tilted floor still has a high abs(normal_z), but not exactly 1.

    Example:
        horizontal_threshold = 0.90 -> strict
        horizontal_threshold = 0.75 -> allows more tilted floor patches
    """
    normal, _ = normalize_plane(plane_model)
    return abs(normal[2]) >= horizontal_threshold


def make_cloud_from_indices(pcd, indices):
    if len(indices) == 0:
        return o3d.geometry.PointCloud()
    return pcd.select_by_index(indices.tolist())


def find_next_floor_patch(
    remaining,
    floor_search_limit,
    distance_threshold,
    ransac_n,
    num_iterations,
    max_candidates_per_patch,
    min_plane_ratio,
    min_plane_points,
    horizontal_threshold,
    min_xy_extent,
    output_dir,
    patch_index
):
    """
    Finds one floor-like patch inside the low-Z region.

    Important:
    If RANSAC finds an object surface that is not floor-like, that candidate
    is skipped from the search only. It is NOT removed from the real cloud.
    """

    points = np.asarray(remaining.points)

    available_indices = np.where(points[:, 2] <= floor_search_limit)[0]

    print(f"[INFO] Low-Z search points available: {len(available_indices)}")

    if len(available_indices) < min_plane_points:
        print("[INFO] Not enough low-Z points to search for another floor patch.")
        return None, None, None

    skipped_candidates = []

    for candidate_id in range(max_candidates_per_patch):
        if len(available_indices) < min_plane_points:
            print("[INFO] Search cloud too small after skipped candidates.")
            break

        search_cloud = remaining.select_by_index(available_indices.tolist())
        search_count = len(search_cloud.points)

        plane_model, local_inliers = search_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        local_inliers = np.asarray(local_inliers, dtype=int)
        inlier_indices_in_remaining = available_indices[local_inliers]

        inlier_count = len(inlier_indices_in_remaining)
        ratio_search = inlier_count / search_count

        candidate_cloud = remaining.select_by_index(inlier_indices_in_remaining.tolist())
        candidate_points = np.asarray(candidate_cloud.points)

        z_median = float(np.median(candidate_points[:, 2]))
        z_min = float(np.min(candidate_points[:, 2]))
        z_max = float(np.max(candidate_points[:, 2]))

        aabb = candidate_cloud.get_axis_aligned_bounding_box()
        extent = np.asarray(aabb.get_extent())

        floor_like = is_floor_like_plane(plane_model, horizontal_threshold)
        xy_extent_ok = extent[0] >= min_xy_extent and extent[1] >= min_xy_extent

        candidate_name = f"floor_patch_search_{patch_index:02d}_candidate_{candidate_id:02d}.pcd"
        candidate_path = os.path.join(output_dir, candidate_name)
        o3d.io.write_point_cloud(candidate_path, candidate_cloud)

        print(
            f"[INFO] Patch {patch_index}, candidate {candidate_id}: "
            f"inliers={inlier_count}, "
            f"ratio_search={ratio_search:.3f}, "
            f"z_median={z_median:.3f}, "
            f"z_range=[{z_min:.3f}, {z_max:.3f}], "
            f"extent={extent}, "
            f"floor_like={floor_like}, "
            f"xy_extent_ok={xy_extent_ok}, "
            f"model={plane_model}"
        )

        accepted = True

        if inlier_count < min_plane_points:
            accepted = False

        if ratio_search < min_plane_ratio:
            accepted = False

        if not floor_like:
            accepted = False

        if not xy_extent_ok:
            accepted = False

        if accepted:
            info = {
                "patch_index": int(patch_index),
                "candidate_id": int(candidate_id),
                "model": [float(x) for x in plane_model],
                "inliers": int(inlier_count),
                "ratio_search": float(ratio_search),
                "z_median": float(z_median),
                "z_min": float(z_min),
                "z_max": float(z_max),
                "extent_xyz": extent.tolist(),
                "candidate_file": candidate_name
            }

            print(f"[INFO] Accepted floor patch {patch_index} from candidate {candidate_id}.")
            return plane_model, inlier_indices_in_remaining, info

        skipped_candidates.append({
            "candidate_id": int(candidate_id),
            "model": [float(x) for x in plane_model],
            "inliers": int(inlier_count),
            "ratio_search": float(ratio_search),
            "z_median": float(z_median),
            "z_min": float(z_min),
            "z_max": float(z_max),
            "extent_xyz": extent.tolist(),
            "floor_like": bool(floor_like),
            "xy_extent_ok": bool(xy_extent_ok),
            "candidate_file": candidate_name
        })

        print("[INFO] Candidate rejected. Skipping it only inside the search cloud.")

        keep_available_mask = np.ones(len(available_indices), dtype=bool)
        keep_available_mask[local_inliers] = False
        available_indices = available_indices[keep_available_mask]

    print(f"[INFO] No accepted floor patch found for patch index {patch_index}.")
    return None, None, {
        "patch_index": int(patch_index),
        "message": "No accepted floor patch.",
        "skipped_candidates": skipped_candidates
    }


def remove_multi_patch_floor(
    pcd,
    floor_percentile,
    floor_search_height,
    floor_remove_height,
    floor_plane_thickness,
    distance_threshold,
    ransac_n,
    num_iterations,
    max_floor_patches,
    max_candidates_per_patch,
    min_plane_ratio,
    min_plane_points,
    horizontal_threshold,
    min_xy_extent,
    extra_low_z_cleanup_height,
    output_dir
):
    """
    Code 6:
    Multi-patch floor removal.

    It handles floors that are not a single plane:
    - main floor plane
    - slightly tilted floor part
    - separate floor patches
    - noisy floor areas

    The search is limited to low-Z points to protect object surfaces.
    """

    remaining = pcd
    removed_floor_clouds = []
    floor_patches_info = []

    original_points = np.asarray(pcd.points)

    z_min = float(np.min(original_points[:, 2]))
    z_max = float(np.max(original_points[:, 2]))
    z_floor_ref = float(np.percentile(original_points[:, 2], floor_percentile))

    floor_search_limit = z_floor_ref + floor_search_height
    floor_remove_limit = z_floor_ref + floor_remove_height
    low_z_cleanup_limit = z_floor_ref + extra_low_z_cleanup_height

    print(f"[INFO] Scene Z min: {z_min:.3f}")
    print(f"[INFO] Scene Z max: {z_max:.3f}")
    print(f"[INFO] Floor reference Z percentile {floor_percentile}%: {z_floor_ref:.3f}")
    print(f"[INFO] Floor search limit: z <= {floor_search_limit:.3f}")
    print(f"[INFO] Floor remove limit: z <= {floor_remove_limit:.3f}")
    print(f"[INFO] Extra low-Z cleanup limit: z <= {low_z_cleanup_limit:.3f}")

    for patch_index in range(max_floor_patches):
        print("")
        print(f"[INFO] Searching for floor patch {patch_index}...")

        if len(remaining.points) < min_plane_points:
            print("[INFO] Remaining cloud too small. Stop multi-patch floor removal.")
            break

        floor_model, floor_inlier_indices, floor_info = find_next_floor_patch(
            remaining=remaining,
            floor_search_limit=floor_search_limit,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            max_candidates_per_patch=max_candidates_per_patch,
            min_plane_ratio=min_plane_ratio,
            min_plane_points=min_plane_points,
            horizontal_threshold=horizontal_threshold,
            min_xy_extent=min_xy_extent,
            output_dir=output_dir,
            patch_index=patch_index
        )

        if floor_model is None:
            if floor_info is not None:
                floor_patches_info.append(floor_info)
            print("[INFO] No more floor patches found.")
            break

        points = np.asarray(remaining.points)
        normal, d = normalize_plane(floor_model)

        distances = np.abs(points @ normal + d)

        plane_distance_mask = distances <= floor_plane_thickness
        low_enough_mask = points[:, 2] <= floor_remove_limit

        remove_mask = plane_distance_mask & low_enough_mask

        removed_count = int(np.sum(remove_mask))

        print(f"[INFO] Floor patch {patch_index}: expanded removal points={removed_count}")

        if removed_count < min_plane_points:
            print("[INFO] Expanded removal too small. Stop to avoid unstable floor removal.")
            break

        removed_indices = np.where(remove_mask)[0]
        kept_indices = np.where(~remove_mask)[0]

        removed_patch_cloud = remaining.select_by_index(removed_indices.tolist())
        remaining = remaining.select_by_index(kept_indices.tolist())

        removed_patch_name = f"removed_floor_patch_{patch_index:02d}.pcd"
        removed_patch_path = os.path.join(output_dir, removed_patch_name)
        o3d.io.write_point_cloud(removed_patch_path, removed_patch_cloud)

        print(f"[INFO] Saved removed floor patch: {removed_patch_path}")
        print(f"[INFO] Remaining points after patch {patch_index}: {len(remaining.points)}")

        removed_floor_clouds.append(removed_patch_cloud)

        floor_info["expanded_removed_points"] = removed_count
        floor_info["removed_patch_file"] = removed_patch_name
        floor_info["floor_plane_thickness"] = float(floor_plane_thickness)
        floor_info["floor_remove_limit"] = float(floor_remove_limit)

        floor_patches_info.append(floor_info)

    if extra_low_z_cleanup_height > 0:
        print("")
        print("[INFO] Running extra low-Z cleanup...")

        points = np.asarray(remaining.points)
        cleanup_mask = points[:, 2] <= low_z_cleanup_limit
        cleanup_count = int(np.sum(cleanup_mask))

        print(f"[INFO] Extra low-Z cleanup points: {cleanup_count}")

        if cleanup_count > 0:
            cleanup_indices = np.where(cleanup_mask)[0]
            kept_indices = np.where(~cleanup_mask)[0]

            cleanup_cloud = remaining.select_by_index(cleanup_indices.tolist())
            remaining = remaining.select_by_index(kept_indices.tolist())

            cleanup_path = os.path.join(output_dir, "removed_floor_extra_low_z_cleanup.pcd")
            o3d.io.write_point_cloud(cleanup_path, cleanup_cloud)

            removed_floor_clouds.append(cleanup_cloud)

            floor_patches_info.append({
                "type": "extra_low_z_cleanup",
                "removed_points": int(cleanup_count),
                "z_limit": float(low_z_cleanup_limit),
                "file": "removed_floor_extra_low_z_cleanup.pcd"
            })

            print(f"[INFO] Saved extra low-Z cleanup: {cleanup_path}")
            print(f"[INFO] Remaining points after cleanup: {len(remaining.points)}")

    if len(removed_floor_clouds) > 0:
        combined = removed_floor_clouds[0]

        for cloud in removed_floor_clouds[1:]:
            combined += cloud

        combined_path = os.path.join(output_dir, "removed_floor_multi_combined.pcd")
        o3d.io.write_point_cloud(combined_path, combined)

        print(f"[INFO] Saved combined removed floor: {combined_path}")
    else:
        print("[WARNING] No floor points removed.")

    remaining_path = os.path.join(output_dir, "objects_after_floor_removal.pcd")
    o3d.io.write_point_cloud(remaining_path, remaining)

    summary = {
        "z_min": float(z_min),
        "z_max": float(z_max),
        "floor_percentile": float(floor_percentile),
        "z_floor_ref": float(z_floor_ref),
        "floor_search_limit": float(floor_search_limit),
        "floor_remove_limit": float(floor_remove_limit),
        "low_z_cleanup_limit": float(low_z_cleanup_limit),
        "floor_plane_thickness": float(floor_plane_thickness),
        "max_floor_patches": int(max_floor_patches),
        "remaining_points": int(len(remaining.points)),
        "floor_patches": floor_patches_info
    }

    summary_path = os.path.join(output_dir, "removed_floor_multi_summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[INFO] Saved remaining cloud: {remaining_path}")
    print(f"[INFO] Saved floor summary: {summary_path}")

    return remaining, summary


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

        cluster = pcd.select_by_index(indices.tolist())

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
        description="Code 6: multi-patch floor removal for non-perfect floors, then DBSCAN clustering."
    )

    parser.add_argument("--input", required=True, help="Input scene point cloud: .pcd/.ply/.xyz")
    parser.add_argument("--output", default="segmented_scene_trial6_floor_multi", help="Output directory")

    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size in meters")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Statistical outlier removal neighbors")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Statistical outlier removal std ratio")

    parser.add_argument("--floor_percentile", type=float, default=1.0, help="Low percentile used as floor Z reference")
    parser.add_argument("--floor_search_height", type=float, default=0.60, help="Search floor patches below floor reference + this height")
    parser.add_argument("--floor_remove_height", type=float, default=0.65, help="Remove floor points below floor reference + this height")
    parser.add_argument("--floor_plane_thickness", type=float, default=0.10, help="Distance thickness around each floor plane")

    parser.add_argument("--plane_dist", type=float, default=0.04, help="RANSAC plane distance threshold")
    parser.add_argument("--ransac_n", type=int, default=3, help="RANSAC points per sample")
    parser.add_argument("--plane_iter", type=int, default=1500, help="RANSAC iterations")

    parser.add_argument("--max_floor_patches", type=int, default=8, help="Maximum accepted floor patches to remove")
    parser.add_argument("--max_candidates_per_patch", type=int, default=8, help="Rejected candidates to skip during each patch search")

    parser.add_argument("--min_plane_ratio", type=float, default=0.003, help="Minimum candidate plane ratio inside low-Z search cloud")
    parser.add_argument("--min_plane_points", type=int, default=800, help="Minimum number of points for a floor patch")
    parser.add_argument("--horizontal_threshold", type=float, default=0.75, help="abs(normal_z) threshold for floor-like planes")
    parser.add_argument("--min_xy_extent", type=float, default=0.40, help="Minimum X and Y extent for accepted floor patch")

    parser.add_argument("--extra_low_z_cleanup_height", type=float, default=0.04, help="Final low-Z cleanup height. Set 0 to disable.")

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

    objects_cloud, floor_summary = remove_multi_patch_floor(
        pcd=clean,
        floor_percentile=args.floor_percentile,
        floor_search_height=args.floor_search_height,
        floor_remove_height=args.floor_remove_height,
        floor_plane_thickness=args.floor_plane_thickness,
        distance_threshold=args.plane_dist,
        ransac_n=args.ransac_n,
        num_iterations=args.plane_iter,
        max_floor_patches=args.max_floor_patches,
        max_candidates_per_patch=args.max_candidates_per_patch,
        min_plane_ratio=args.min_plane_ratio,
        min_plane_points=args.min_plane_points,
        horizontal_threshold=args.horizontal_threshold,
        min_xy_extent=args.min_xy_extent,
        extra_low_z_cleanup_height=args.extra_low_z_cleanup_height,
        output_dir=args.output
    )

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