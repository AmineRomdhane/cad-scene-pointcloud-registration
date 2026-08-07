# CAD-to-Scene Point Cloud Registration

This repository contains the point cloud registration pipeline used in the GMC6003 project to align CAD-derived point clouds with segmented scene point clouds.

The goal is to convert or subsample CAD geometry into a point cloud, register it to a segmented object extracted from a real scene, save the estimated transformation matrix, and export the aligned CAD point cloud.

## Project Context

In the GMC6003 project, real scenes were segmented into point cloud clusters. Each cluster represented a candidate object or scene part. CAD models were then converted or sampled into point clouds and aligned with these segmented scene clusters.

The workflow was used to:

- generate CAD proxy point clouds;
- segment real scene point clouds into clusters;
- register CAD point clouds to scene clusters;
- refine the alignment using ICP;
- save transformation matrices;
- export registered CAD point clouds;
- visually inspect the CAD-to-scene alignment.

## Main Pipeline

1. Load the CAD model or CAD-derived point cloud.
2. Convert or subsample the CAD geometry into a point cloud.
3. Load the segmented scene point cloud.
4. Apply optional scaling if CAD and scene units are different.
5. Apply an initial transform.
6. Refine the alignment using ICP.
7. Save the final transformation matrix.
8. Export the registered CAD point cloud.

## Main Scripts

```text
src/01_create_cad_proxy.py
src/02_register_cad_to_cluster.py
src/03_register_cad_yaw_search.py
src/04_refine_registration_from_matrix.py
src/05_apply_transform_to_cad_mesh.py
src/07_adjust_cad_dimensions_and_proxy.py
src/cad_mesh_surface_to_proxy_pcd.py
src/convert_dae_to_ply.py
src/convert_step_to_ply_freecad.py
src/segment_scene_v11_point_merge.py

## Scene Registration Example

The following figures show the scene point cloud before and after CAD-to-scene registration.

### Before CAD Registration

The scene initially contains only the segmented point cloud clusters extracted from the real scan.

![Scene point cloud before CAD registration](docs/images/scene_before_registration.png)

### After CAD Registration

After registration, the CAD models are aligned with the segmented scene clusters using the estimated transformation matrices.

![Scene point cloud after CAD registration](docs/images/scene_after_registration.png)
