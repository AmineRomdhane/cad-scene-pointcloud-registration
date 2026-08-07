import argparse
import os
import sys
import numpy as np

import bpy
from mathutils import Matrix


def parse_args():
    argv = sys.argv

    if "--" not in argv:
        raise RuntimeError("Use -- before script arguments.")

    argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(
        description="Apply Open3D 4x4 transform to a colored GLB CAD file and export a transformed GLB."
    )

    parser.add_argument("--input", required=True, help="Input colored CAD .glb")
    parser.add_argument("--transform", required=True, help="Open3D transform matrix .txt")
    parser.add_argument("--output", required=True, help="Output transformed .glb")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale before applying transform. Keep 1.0 if CAD is already in meters.")

    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)


def export_glb(path):
    output_dir = os.path.dirname(path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        export_apply=True
    )


def get_root_objects():
    return [obj for obj in bpy.context.scene.objects if obj.parent is None]


def main():
    args = parse_args()

    clear_scene()

    print(f"[INFO] Importing GLB: {args.input}")
    import_glb(args.input)

    T = np.loadtxt(args.transform)

    if T.shape != (4, 4):
        raise RuntimeError(f"Transform must be 4x4, got {T.shape}")

    print("[INFO] Loaded transform:")
    print(T)

    T_blender = Matrix(T.tolist())

    root_objects = get_root_objects()

    print(f"[INFO] Root objects: {len(root_objects)}")

    if args.scale != 1.0:
        print(f"[INFO] Applying scale: {args.scale}")
        S = Matrix((
            (args.scale, 0.0, 0.0, 0.0),
            (0.0, args.scale, 0.0, 0.0),
            (0.0, 0.0, args.scale, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ))
        for obj in root_objects:
            obj.matrix_world = S @ obj.matrix_world

    print("[INFO] Applying registration transform to GLB objects...")

    for obj in root_objects:
        obj.matrix_world = T_blender @ obj.matrix_world

    print(f"[INFO] Exporting transformed GLB: {args.output}")
    export_glb(args.output)

    print("[DONE] Saved transformed colored GLB.")


if __name__ == "__main__":
    main()
