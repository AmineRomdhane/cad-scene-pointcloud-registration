import open3d as o3d
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "synthetic_cases"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_reference_cloud():
    # Create a simple 3D object: box + sphere
    box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.0, depth=1.0)
    box.translate([-1.0, -0.5, -0.5])

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
    sphere.translate([0.8, 0.0, 0.3])

    mesh = box + sphere
    cloud = mesh.sample_points_uniformly(number_of_points=5000)

    return cloud

def make_transform():
    angle = np.deg2rad(15)

    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])

    t = np.array([0.5, -0.2, 0.1])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def add_noise(cloud, sigma=0.01):
    points = np.asarray(cloud.points)
    noise = np.random.normal(0, sigma, points.shape)
    noisy_points = points + noise

    noisy_cloud = o3d.geometry.PointCloud()
    noisy_cloud.points = o3d.utility.Vector3dVector(noisy_points)

    return noisy_cloud

if __name__ == "__main__":
    np.random.seed(42)

    reference = create_reference_cloud()
    T_true = make_transform()

    observation = o3d.geometry.PointCloud(reference)
    observation.transform(T_true)
    observation = add_noise(observation, sigma=0.01)

    o3d.io.write_point_cloud(str(DATA_DIR / "reference_t0.ply"), reference)
    o3d.io.write_point_cloud(str(DATA_DIR / "observation_t1.ply"), observation)

    np.savetxt(DATA_DIR / "T_true.txt", T_true)

    print("Saved:")
    print(DATA_DIR / "reference_t0.ply")
    print(DATA_DIR / "observation_t1.ply")
    print(DATA_DIR / "T_true.txt")
