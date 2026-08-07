import sys
import os

import FreeCAD
import Import
import Mesh
import MeshPart


def convert_step_to_ply(input_path, output_path, linear_deflection=0.5, angular_deflection=0.3):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    doc = FreeCAD.newDocument("step_to_ply_doc")

    print(f"[INFO] Loading STEP/STP: {input_path}")
    Import.insert(input_path, doc.Name)
    doc.recompute()

    mesh_objects = []

    for obj in doc.Objects:
        if not hasattr(obj, "Shape"):
            continue

        shape = obj.Shape

        if shape.isNull():
            continue

        if len(shape.Faces) == 0:
            continue

        print(f"[INFO] Meshing object: {obj.Name}")

        mesh = MeshPart.meshFromShape(
            Shape=shape,
            LinearDeflection=linear_deflection,
            AngularDeflection=angular_deflection,
            Relative=False
        )

        mesh_obj = doc.addObject("Mesh::Feature", obj.Name + "_mesh")
        mesh_obj.Mesh = mesh
        mesh_objects.append(mesh_obj)

    if len(mesh_objects) == 0:
        raise RuntimeError("No valid mesh objects were generated from the STEP/STP file.")

    print(f"[INFO] Exporting PLY: {output_path}")
    Mesh.export(mesh_objects, output_path)

    print(f"[OK] Saved PLY: {output_path}")

    FreeCAD.closeDocument(doc.Name)


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  freecadcmd src/convert_step_to_ply_freecad.py input.stp output.ply [linear_deflection] [angular_deflection]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    linear_deflection = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    angular_deflection = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3

    convert_step_to_ply(
        input_path=input_path,
        output_path=output_path,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection
    )


if __name__ == "__main__":
    main()
