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


def get_axis_values(points, axis):
    if axis == "x":
        return points[:, 0]
    if axis == "y":
        return points[:, 1]
    if axis == "z":
        return points[:, 2]

    raise ValueError("axis must be x, y, or z")


def save_height_colored_cloud(pcd, axis, output_path):
    points = np.asarray(pcd.points)
    values = get_axis_values(points, axis)

    v_min = float(np.min(values))
    v_max = float(np.max(values))

    denom = max(v_max - v_min, 1e-9)
    t = (values - v_min) / denom

    colors = np.zeros((len(points), 3))

    # Simple blue-to-red style height coloring without external libraries.
    colors[:, 0] = t
    colors[:, 1] = 1.0 - np.abs(t - 0.5) * 2.0
    colors[:, 2] = 1.0 - t

    colored = o3d.geometry.PointCloud()
    colored.points = pcd.points
    colored.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, colored)

    print(f"[INFO] Saved {axis.upper()}-colored cloud: {output_path}")


def print_axis_statistics(pcd, axis):
    points = np.asarray(pcd.points)
    values = get_axis_values(points, axis)

    percentiles = [0, 0.5, 1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99, 99.5, 100]
    stats = {}

    print("")
    print(f"[INFO] {axis.upper()} statistics:")

    for p in percentiles:
        val = float(np.percentile(values, p))
        stats[f"p{p}"] = val
        print(f"  p{p:>5}: {val:.4f}")

    return stats


def remove_floor_by_axis_cut(
    pcd,
    axis,
    floor_percentile,
    floor_offset,
    absolute_cut,
    remove_side,
    output_dir
):
    points = np.asarray(pcd.points)
    values = get_axis_values(points, axis)

    if absolute_cut is not None:
        cut_value = absolute_cut
        cut_mode = "absolute"
    else:
        ref_value = float(np.percentile(values, floor_percentile))
        cut_value = ref_value + floor_offset
        cut_mode = "percentile_plus_offset"

    if remove_side == "below":
        remove_mask = values <= cut_value
    elif remove_side == "above":
        remove_mask = values >= cut_value
    else:
        raise ValueError("remove_side must be below or above")

    removed_indices = np.where(remove_mask)[0]
    kept_indices = np.where(~remove_mask)[0]

    removed = pcd.select_by_index(removed_indices.tolist())
    remaining = pcd.select_by_index(kept_indices.tolist())

    removed_path = os.path.join(output_dir, "removed_floor_z_cut.pcd")
    remaining_path = os.path.join(output_dir, "objects_after_floor_removal.pcd")

    o3d.io.write_point_cloud(removed_path, removed)
    o3d.io.write_point_cloud(remaining_path, remaining)

    print("")
    print(f"[INFO] Axis used: {axis}")
    print(f"[INFO] Cut mode: {cut_mode}")
    print(f"[INFO] Remove side: {remove_side}")
    print(f"[INFO] Cut value: {cut_value:.4f}")
    print(f"[INFO] Removed points: {len(removed.points)}")
    print(f"[INFO] Remaining points: {len(remaining.points)}")
    print(f"[INFO] Saved removed floor: {removed_path}")
    print(f"[INFO] Saved remaining cloud: {remaining_path}")

    info = {
        "axis": axis,
        "cut_mode": cut_mode,
        "floor_percentile": floor_percentile,
        "floor_offset": floor_offset,
        "absolute_cut": absolute_cut,
        "remove_side": remove_side,
        "cut_value": float(cut_value),
        "removed_points": int(len(removed.points)),
        "remaining_points": int(len(remaining.points))
    }

    return remaining, removed, info


def cluster_dbscan(pcd, eps, min_points):
    print("")
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
        description="Code 8: direct floor Z-cut diagnostic/removal, then DBSCAN clustering."
    )

    parser.add_argument("--input", required=True, help="Input scene point cloud: .pcd/.ply/.xyz")
    parser.add_argument("--output", default="segmented_scene_trial8_z_floor_debug", help="Output directory")

    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size in meters")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Statistical outlier removal neighbors")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Statistical outlier removal std ratio")

    parser.add_argument("--axis", choices=["x", "y", "z"], default="z", help="Axis used for floor cut")
    parser.add_argument("--floor_percentile", type=float, default=1.0, help="Low percentile used as floor reference")
    parser.add_argument("--floor_offset", type=float, default=0.08, help="Cut value = percentile + offset")
    parser.add_argument("--absolute_cut", type=float, default=None, help="Use fixed absolute cut instead of percentile + offset")
    parser.add_argument("--remove_side", choices=["below", "above"], default="below", help="Remove values below or above cut")

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

    height_colored_path = os.path.join(args.output, f"{args.axis}_colored_cloud.pcd")
    save_height_colored_cloud(clean, args.axis, height_colored_path)

    axis_stats = print_axis_statistics(clean, args.axis)

    objects_cloud, removed_floor, cut_info = remove_floor_by_axis_cut(
        pcd=clean,
        axis=args.axis,
        floor_percentile=args.floor_percentile,
        floor_offset=args.floor_offset,
        absolute_cut=args.absolute_cut,
        remove_side=args.remove_side,
        output_dir=args.output
    )

    summary = {
        "axis_statistics": axis_stats,
        "cut_info": cut_info
    }

    summary_path = os.path.join(args.output, "z_floor_cut_summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[INFO] Saved Z-cut summary: {summary_path}")

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