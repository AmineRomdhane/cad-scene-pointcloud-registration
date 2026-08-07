import argparse
import os
import trimesh


def main():
    parser = argparse.ArgumentParser(description="Convert DAE/COLLADA file to PLY using trimesh.")
    parser.add_argument("--input", required=True, help="Input .dae file")
    parser.add_argument("--output", required=True, help="Output .ply file")
    args = parser.parse_args()

    print(f"[INFO] Loading: {args.input}")

    obj = trimesh.load(args.input, force="scene")

    if obj is None:
        raise RuntimeError("Failed to load DAE file.")

    if isinstance(obj, trimesh.Scene):
        print(f"[INFO] Loaded as scene with {len(obj.geometry)} geometries")

        if len(obj.geometry) == 0:
            raise RuntimeError("Scene contains no geometry.")

        mesh = trimesh.util.concatenate(tuple(obj.geometry.values()))
    elif isinstance(obj, trimesh.Trimesh):
        print("[INFO] Loaded as single mesh")
        mesh = obj
    else:
        raise RuntimeError(f"Unsupported loaded object type: {type(obj)}")

    print(f"[INFO] Vertices: {len(mesh.vertices)}")
    print(f"[INFO] Faces: {len(mesh.faces)}")
    print(f"[INFO] Bounds:\n{mesh.bounds}")
    print(f"[INFO] Extents: {mesh.extents}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    mesh.export(args.output)

    print(f"[DONE] Saved: {args.output}")


if __name__ == "__main__":
    main()