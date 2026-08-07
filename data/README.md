# Data Folder

This folder is reserved for local input data used by the CAD-to-scene registration pipeline.

Typical local inputs include:

- CAD-derived point clouds;
- segmented scene point clouds;
- CAD meshes or converted CAD files;
- point cloud clusters extracted from a real scan.

Large raw files are not tracked by Git. Place them locally in this folder when running the scripts.

Example local structure:

```text
data/
├── cad/
│   └── cad_proxy.ply
├── scene/
│   └── full_scene.ply
└── clusters/
    └── segmented_cluster_001.ply
