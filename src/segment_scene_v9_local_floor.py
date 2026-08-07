import argparse
import os
import json
from collections import defaultdict

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


def save_z_colored_cloud(pcd, output_path):
    points = np.asarray(pcd.points)
    z = points[:, 2]

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    denom = max(z_max - z_min, 1e-9)

    t = (z - z_min) / denom

    colors = np.zeros((len(points), 3))
    colors[:, 0] = t
    colors[:, 1] = 1.0 - np.abs(t - 0.5) * 2.0
    colors[:, 2] = 1.0 - t

    colored = o3d.geometry.PointCloud()
    colored.points = pcd.points
    colored.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, colored)
    print(f"[INFO] Saved Z-colored cloud: {output_path}")


def build_grid_indices(points, cell_size):
    x = points[:, 0]
    y = points[:, 1]

    x_min = float(np.min(x))
    y_min = float(np.min(y))

    ix = np.floor((x - x_min) / cell_size).astype(np.int32)
    iy = np.floor((y - y_min) / cell_size).astype(np.int32)

    cells = defaultdict(list)

    for idx, key in enumerate(zip(ix, iy)):
        cells[key].append(idx)

    grid_info = {
        "x_min": x_min,
        "y_min": y_min,
        "cell_size": cell_size
    }

    return cells, grid_info


def estimate_local_floor_and_remove(
    pcd,
    cell_size,
    local_floor_percentile,
    global_floor_percentile,
    floor_thickness,
    max_floor_height_above_global,
    min_cell_points,
    min_floor_points_per_cell,
    output_dir
):
    """
    Code 9:
    Local floor removal using an XY grid.

    This does NOT assume that the whole floor is one perfect plane.

    For each XY cell:
        1. Estimate local floor height using a low Z percentile.
        2. Accept the cell as floor-like if it is not too high above global floor.
        3. Remove points close to the local floor height.

    This handles:
        - slightly sloped floor
        - large scene floor variation
        - floor made of several weakly angled patches
    """

    points = np.asarray(pcd.points)
    z = points[:, 2]

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    global_floor_ref = float(np.percentile(z, global_floor_percentile))
    global_floor_limit = global_floor_ref + max_floor_height_above_global

    print(f"[INFO] Scene Z min: {z_min:.4f}")
    print(f"[INFO] Scene Z max: {z_max:.4f}")
    print(f"[INFO] Global floor reference p{global_floor_percentile}: {global_floor_ref:.4f}")
    print(f"[INFO] Max accepted local floor Z: {global_floor_limit:.4f}")

    cells, grid_info = build_grid_indices(points, cell_size)

    print(f"[INFO] Number of occupied XY cells: {len(cells)}")
    print(f"[INFO] Cell size: {cell_size}")

    remove_mask = np.zeros(len(points), dtype=bool)

    floor_cells_info = []
    rejected_cells_info = []

    floor_map_points = []
    floor_map_colors = []

    accepted_cell_count = 0
    rejected_cell_count = 0

    for cell_key, indices_list in cells.items():
        indices = np.asarray(indices_list, dtype=int)

        if len(indices) < min_cell_points:
            rejected_cell_count += 1
            continue

        cell_points = points[indices]
        cell_z = cell_points[:, 2]

        local_floor_z = float(np.percentile(cell_z, local_floor_percentile))

        near_floor_mask_local = cell_z <= local_floor_z + floor_thickness
        near_floor_count = int(np.sum(near_floor_mask_local))

        cell_is_low_enough = local_floor_z <= global_floor_limit
        cell_has_floor_support = near_floor_count >= min_floor_points_per_cell

        if cell_is_low_enough and cell_has_floor_support:
            accepted_cell_count += 1

            global_near_floor_indices = indices[near_floor_mask_local]
            remove_mask[global_near_floor_indices] = True

            ix, iy = cell_key

            x_center = grid_info["x_min"] + (ix + 0.5) * cell_size
            y_center = grid_info["y_min"] + (iy + 0.5) * cell_size

            floor_map_points.append([x_center, y_center, local_floor_z])
            floor_map_colors.append([0.0, 1.0, 0.0])

            floor_cells_info.append({
                "cell_ix": int(ix),
                "cell_iy": int(iy),
                "num_points": int(len(indices)),
                "local_floor_z": float(local_floor_z),
                "near_floor_count": int(near_floor_count),
                "removed_points_in_cell": int(len(global_near_floor_indices))
            })

        else:
            rejected_cell_count += 1

            if len(rejected_cells_info) < 200:
                ix, iy = cell_key
                rejected_cells_info.append({
                    "cell_ix": int(ix),
                    "cell_iy": int(iy),
                    "num_points": int(len(indices)),
                    "local_floor_z": float(local_floor_z),
                    "near_floor_count": int(near_floor_count),
                    "cell_is_low_enough": bool(cell_is_low_enough),
                    "cell_has_floor_support": bool(cell_has_floor_support)
                })

    removed_indices = np.where(remove_mask)[0]
    kept_indices = np.where(~remove_mask)[0]

    removed_floor = pcd.select_by_index(removed_indices.tolist())
    remaining = pcd.select_by_index(kept_indices.tolist())

    removed_path = os.path.join(output_dir, "removed_floor_local_grid.pcd")
    remaining_path = os.path.join(output_dir, "objects_after_floor_removal.pcd")

    o3d.io.write_point_cloud(removed_path, removed_floor)
    o3d.io.write_point_cloud(remaining_path, remaining)

    print(f"[INFO] Accepted floor cells: {accepted_cell_count}")
    print(f"[INFO] Rejected cells: {rejected_cell_count}")
    print(f"[INFO] Removed floor points: {len(removed_indices)}")
    print(f"[INFO] Remaining points: {len(kept_indices)}")
    print(f"[INFO] Saved removed local floor: {removed_path}")
    print(f"[INFO] Saved remaining cloud: {remaining_path}")

    if len(floor_map_points) > 0:
        floor_map_cloud = o3d.geometry.PointCloud()
        floor_map_cloud.points = o3d.utility.Vector3dVector(np.asarray(floor_map_points))
        floor_map_cloud.colors = o3d.utility.Vector3dVector(np.asarray(floor_map_colors))

        floor_map_path = os.path.join(output_dir, "local_floor_map_points.pcd")
        o3d.io.write_point_cloud(floor_map_path, floor_map_cloud)
        print(f"[INFO] Saved local floor map points: {floor_map_path}")
    else:
        floor_map_path = None
        print("[WARNING] No local floor cells accepted.")

    summary = {
        "method": "local_xy_grid_floor_removal",
        "z_min": z_min,
        "z_max": z_max,
        "global_floor_percentile": global_floor_percentile,
        "global_floor_ref": global_floor_ref,
        "max_floor_height_above_global": max_floor_height_above_global,
        "global_floor_limit": global_floor_limit,
        "cell_size": cell_size,
        "local_floor_percentile": local_floor_percentile,
        "floor_thickness": floor_thickness,
        "min_cell_points": min_cell_points,
        "min_floor_points_per_cell": min_floor_points_per_cell,
        "occupied_cells": int(len(cells)),
        "accepted_floor_cells": int(accepted_cell_count),
        "rejected_cells": int(rejected_cell_count),
        "removed_points": int(len(removed_indices)),
        "remaining_points": int(len(kept_indices)),
        "removed_floor_file": "removed_floor_local_grid.pcd",
        "remaining_file": "objects_after_floor_removal.pcd",
        "local_floor_map_file": os.path.basename(floor_map_path) if floor_map_path else None,
        "floor_cells": floor_cells_info,
        "sample_rejected_cells": rejected_cells_info
    }

    summary_path = os.path.join(output_dir, "local_floor_removal_summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[INFO] Saved local floor summary: {summary_path}")

    return remaining, removed_floor, summary


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
        description="Code 9: local XY grid floor removal for non-planar / large floors, then DBSCAN clustering."
    )

    parser.add_argument("--input", required=True, help="Input scene point cloud: .pcd/.ply/.xyz")
    parser.add_argument("--output", default="segmented_scene_trial9_local_floor", help="Output directory")

    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size in meters")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Statistical outlier removal neighbors")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Statistical outlier removal std ratio")

    parser.add_argument("--cell_size", type=float, default=0.30, help="XY grid cell size in meters")
    parser.add_argument("--local_floor_percentile", type=float, default=5.0, help="Local low-Z percentile per cell")
    parser.add_argument("--global_floor_percentile", type=float, default=1.0, help="Global low-Z floor reference percentile")
    parser.add_argument("--floor_thickness", type=float, default=0.08, help="Remove points up to local floor Z + this value")
    parser.add_argument("--max_floor_height_above_global", type=float, default=0.80, help="Reject local floor cells higher than global floor reference + this value")

    parser.add_argument("--min_cell_points", type=int, default=20, help="Minimum points in an XY cell")
    parser.add_argument("--min_floor_points_per_cell", type=int, default=5, help="Minimum near-floor points required in a cell")

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

    z_colored_path = os.path.join(args.output, "z_colored_cloud.pcd")
    save_z_colored_cloud(clean, z_colored_path)

    objects_cloud, removed_floor, floor_summary = estimate_local_floor_and_remove(
        pcd=clean,
        cell_size=args.cell_size,
        local_floor_percentile=args.local_floor_percentile,
        global_floor_percentile=args.global_floor_percentile,
        floor_thickness=args.floor_thickness,
        max_floor_height_above_global=args.max_floor_height_above_global,
        min_cell_points=args.min_cell_points,
        min_floor_points_per_cell=args.min_floor_points_per_cell,
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