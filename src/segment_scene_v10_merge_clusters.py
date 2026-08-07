import argparse
import os
import json
from collections import defaultdict

import numpy as np
import open3d as o3d


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


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
        "floor_cells": floor_cells_info
    }

    summary_path = os.path.join(output_dir, "local_floor_removal_summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[INFO] Saved local floor summary: {summary_path}")

    return remaining, removed_floor, summary


def cluster_dbscan(pcd, eps, min_points):
    print("")
    print("[INFO] Running raw DBSCAN clustering...")

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

    print(f"[INFO] Raw DBSCAN clusters found: {n_clusters}")
    print(f"[INFO] Raw DBSCAN noise points: {n_noise}")

    return labels, n_clusters


def save_labeled_cloud(pcd, labels, output_path):
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
    print(f"[INFO] Saved labeled cloud: {output_path}")


def get_cluster_infos(pcd, labels, n_clusters, min_raw_cluster_points):
    cluster_infos = []

    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]

        if len(indices) < min_raw_cluster_points:
            continue

        cluster = pcd.select_by_index(indices.tolist())

        aabb = cluster.get_axis_aligned_bounding_box()
        min_bound = np.asarray(aabb.get_min_bound())
        max_bound = np.asarray(aabb.get_max_bound())
        extent = np.asarray(aabb.get_extent())
        center = np.asarray(aabb.get_center())

        cluster_infos.append({
            "raw_label": int(cluster_id),
            "indices": indices,
            "num_points": int(len(indices)),
            "min_bound": min_bound,
            "max_bound": max_bound,
            "extent": extent,
            "center": center
        })

    print(f"[INFO] Raw clusters kept for merging: {len(cluster_infos)}")
    return cluster_infos


def axis_gap(min_a, max_a, min_b, max_b):
    if max_a < min_b:
        return min_b - max_a
    if max_b < min_a:
        return min_a - max_b
    return 0.0


def should_merge_clusters(info_a, info_b, merge_xy_gap, merge_z_gap):
    min_a = info_a["min_bound"]
    max_a = info_a["max_bound"]
    min_b = info_b["min_bound"]
    max_b = info_b["max_bound"]

    gap_x = axis_gap(min_a[0], max_a[0], min_b[0], max_b[0])
    gap_y = axis_gap(min_a[1], max_a[1], min_b[1], max_b[1])
    gap_z = axis_gap(min_a[2], max_a[2], min_b[2], max_b[2])

    xy_gap = np.sqrt(gap_x ** 2 + gap_y ** 2)

    return xy_gap <= merge_xy_gap and gap_z <= merge_z_gap


def merge_raw_clusters(
    pcd,
    raw_labels,
    n_raw_clusters,
    output_dir,
    min_raw_cluster_points,
    merge_xy_gap,
    merge_z_gap,
    min_merged_cluster_points,
    min_merged_extent,
    max_merged_extent
):
    cluster_infos = get_cluster_infos(
        pcd=pcd,
        labels=raw_labels,
        n_clusters=n_raw_clusters,
        min_raw_cluster_points=min_raw_cluster_points
    )

    n = len(cluster_infos)

    if n == 0:
        print("[WARNING] No raw clusters available for merging.")
        return np.full(len(pcd.points), -1, dtype=int), []

    uf = UnionFind(n)

    print("[INFO] Merging nearby raw clusters using AABB gaps...")

    merge_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge_clusters(
                cluster_infos[i],
                cluster_infos[j],
                merge_xy_gap=merge_xy_gap,
                merge_z_gap=merge_z_gap
            ):
                uf.union(i, j)
                merge_count += 1

    print(f"[INFO] Pairwise merge links created: {merge_count}")

    groups = defaultdict(list)

    for i in range(n):
        root = uf.find(i)
        groups[root].append(i)

    print(f"[INFO] Merged groups before filtering: {len(groups)}")

    merged_labels = np.full(len(pcd.points), -1, dtype=int)
    merged_info = []

    saved_count = 0

    for _, group_indices in groups.items():
        point_indices_list = []

        raw_labels_in_group = []

        for local_cluster_index in group_indices:
            info = cluster_infos[local_cluster_index]
            point_indices_list.append(info["indices"])
            raw_labels_in_group.append(info["raw_label"])

        merged_indices = np.concatenate(point_indices_list)

        if len(merged_indices) < min_merged_cluster_points:
            continue

        cluster = pcd.select_by_index(merged_indices.tolist())

        aabb = cluster.get_axis_aligned_bounding_box()
        extent = np.asarray(aabb.get_extent())
        center = np.asarray(aabb.get_center())

        if np.any(extent < min_merged_extent):
            continue

        if max_merged_extent > 0 and np.any(extent > max_merged_extent):
            continue

        merged_labels[merged_indices] = saved_count

        cluster_name = f"merged_cluster_{saved_count:03d}.pcd"
        cluster_path = os.path.join(output_dir, cluster_name)

        o3d.io.write_point_cloud(cluster_path, cluster)

        info = {
            "merged_cluster_file": cluster_name,
            "merged_label": int(saved_count),
            "raw_labels_merged": [int(x) for x in raw_labels_in_group],
            "num_raw_fragments": int(len(raw_labels_in_group)),
            "num_points": int(len(merged_indices)),
            "center_xyz": center.tolist(),
            "extent_xyz": extent.tolist(),
            "volume_aabb": float(extent[0] * extent[1] * extent[2])
        }

        merged_info.append(info)

        print(
            f"[SAVED] {cluster_name} | "
            f"points={len(merged_indices)} | "
            f"fragments={len(raw_labels_in_group)} | "
            f"extent={extent}"
        )

        saved_count += 1

    print(f"[INFO] Final merged clusters saved: {saved_count}")

    summary_path = os.path.join(output_dir, "merged_clusters_summary.json")

    with open(summary_path, "w") as f:
        json.dump(merged_info, f, indent=4)

    print(f"[INFO] Saved merged cluster summary: {summary_path}")

    return merged_labels, merged_info


def main():
    parser = argparse.ArgumentParser(
        description="Code 10: local floor removal + DBSCAN fragments + AABB fragment merging."
    )

    parser.add_argument("--input", required=True, help="Input scene point cloud: .pcd/.ply/.xyz")
    parser.add_argument("--output", default="segmented_scene_trial10_merged_clusters", help="Output directory")

    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size in meters")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Statistical outlier removal neighbors")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Statistical outlier removal std ratio")

    parser.add_argument("--cell_size", type=float, default=0.60, help="XY grid cell size in meters")
    parser.add_argument("--local_floor_percentile", type=float, default=7.0, help="Local low-Z percentile per cell")
    parser.add_argument("--global_floor_percentile", type=float, default=1.0, help="Global low-Z floor reference percentile")
    parser.add_argument("--floor_thickness", type=float, default=0.09, help="Remove points up to local floor Z + this value")
    parser.add_argument("--max_floor_height_above_global", type=float, default=1.50, help="Reject local floor cells higher than global floor reference + this value")
    parser.add_argument("--min_cell_points", type=int, default=5, help="Minimum points in an XY cell")
    parser.add_argument("--min_floor_points_per_cell", type=int, default=2, help="Minimum near-floor points required in a cell")

    parser.add_argument("--eps", type=float, default=0.18, help="Raw DBSCAN radius in meters")
    parser.add_argument("--min_points", type=int, default=20, help="Raw DBSCAN minimum points")

    parser.add_argument("--min_raw_cluster_points", type=int, default=50, help="Minimum raw fragment points kept for merging")
    parser.add_argument("--merge_xy_gap", type=float, default=0.25, help="Merge raw clusters if XY AABB gap is below this")
    parser.add_argument("--merge_z_gap", type=float, default=0.35, help="Merge raw clusters if Z AABB gap is below this")

    parser.add_argument("--min_merged_cluster_points", type=int, default=300, help="Minimum points per final merged cluster")
    parser.add_argument("--min_merged_extent", type=float, default=0.08, help="Minimum x/y/z extent for final cluster")
    parser.add_argument("--max_merged_extent", type=float, default=0.0, help="Maximum x/y/z extent. 0 disables this filter.")

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

    raw_labels, n_raw_clusters = cluster_dbscan(
        objects_cloud,
        eps=args.eps,
        min_points=args.min_points
    )

    raw_colored_path = os.path.join(args.output, "raw_dbscan_clusters_colored.pcd")
    save_labeled_cloud(objects_cloud, raw_labels, raw_colored_path)

    merged_labels, merged_info = merge_raw_clusters(
        pcd=objects_cloud,
        raw_labels=raw_labels,
        n_raw_clusters=n_raw_clusters,
        output_dir=args.output,
        min_raw_cluster_points=args.min_raw_cluster_points,
        merge_xy_gap=args.merge_xy_gap,
        merge_z_gap=args.merge_z_gap,
        min_merged_cluster_points=args.min_merged_cluster_points,
        min_merged_extent=args.min_merged_extent,
        max_merged_extent=args.max_merged_extent
    )

    merged_colored_path = os.path.join(args.output, "merged_clusters_colored.pcd")
    save_labeled_cloud(objects_cloud, merged_labels, merged_colored_path)

    print("")
    print("[DONE] Code 10 finished.")
    print(f"[DONE] Raw clusters colored: {raw_colored_path}")
    print(f"[DONE] Merged clusters colored: {merged_colored_path}")
    print(f"[DONE] Final merged clusters saved as merged_cluster_XXX.pcd")


if __name__ == "__main__":
    main()