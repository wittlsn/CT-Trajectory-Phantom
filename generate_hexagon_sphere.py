import trimesh
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path


def calculate_theta_k(theta_k_prior: float, h_k: float, N: float) -> float:
    return (theta_k_prior + 3.6 / np.sqrt(N) * 1. / np.sqrt(1. - h_k ** 2)) % (2 * np.pi)

def calculate_single_vector(theta, sigma):
        vector = np.array([0, 0, 1]).reshape((3, 1))
        vector = Rotation.from_euler('z', theta).as_matrix() @ Rotation.from_euler('x', sigma).as_matrix() @ vector
        return vector

OUTER_RADIUS = 250
INNER_RADIUS = 240
HEXAGON_PARTS = 50 # number of Hexagon parts
HEXAGON_TOP_THICKNESS = 10 # top additional. The overall hexagon thickness is outer_r - innner_r + thickness
SIDE_LENGTH_FAKTOR = 0.99 # Fit in sphere holes
TOP_LENGTH_FACTOR = 1.25 # extended top hexagons
SAVE_FOLDER = Path('./meshes')


def main():
    SAVE_FOLDER.mkdir(exist_ok=True)
    hexagon_heigth = OUTER_RADIUS - INNER_RADIUS  
    outer_sphere = trimesh.creation.uv_sphere(OUTER_RADIUS, (180, 180))
    inner_sphere = trimesh.creation.uv_sphere(INNER_RADIUS, (180, 180))

    k = np.arange(HEXAGON_PARTS, dtype=np.float64)
    h_k = -1. + 2 * k / (HEXAGON_PARTS - 1)
    sigma_k = np.arccos(h_k)
    theta_k = np.zeros_like(k)

    for i in range(1, int(HEXAGON_PARTS)):
        theta_k[i] = calculate_theta_k(theta_k[i - 1], h_k[i], HEXAGON_PARTS)

    vectors = list(map(calculate_single_vector, theta_k.tolist(), sigma_k.tolist()))
    distance_centers = np.linalg.norm(vectors[1] - vectors[2])
    distance_mm = distance_centers * INNER_RADIUS / 2

    side_length = distance_mm * SIDE_LENGTH_FAKTOR
    angle_offset = np.pi / 6
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle_offset
    vertices = np.column_stack([side_length * np.cos(angles), side_length * np.sin(angles)])
    faces = [(0, i, (i + 1) % 8) for i in range(1, 5)]
    hexagon = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    hexagon = trimesh.creation.extrude_triangulation(hexagon.vertices, hexagon.faces, hexagon_heigth + HEXAGON_TOP_THICKNESS)

    side_length = distance_mm * SIDE_LENGTH_FAKTOR * TOP_LENGTH_FACTOR
    angle_offset = np.pi / 6
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle_offset
    vertices = np.column_stack([side_length * np.cos(angles), side_length * np.sin(angles)])
    faces = [(0, i, (i + 1) % 8) for i in range(1, 5)]
    hexagon_add = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    hexagon_add = trimesh.creation.extrude_triangulation(hexagon_add.vertices, hexagon_add.faces, HEXAGON_TOP_THICKNESS)


    hexagon = trimesh.boolean.union([hexagon, hexagon_add])
    hexagon.export(SAVE_FOLDER / 'hexagon.stl')

    side_length = distance_mm * SIDE_LENGTH_FAKTOR * 1.05
    angle_offset = np.pi / 6
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle_offset
    vertices = np.column_stack([side_length * np.cos(angles), side_length * np.sin(angles)])
    faces = [(0, i, (i + 1) % 8) for i in range(1, 5)]
    hexagon_cut = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    hexagon_cut = trimesh.creation.extrude_triangulation(hexagon_cut.vertices, hexagon_cut.faces, hexagon_heigth * 4)

    hull = trimesh.boolean.difference([outer_sphere, inner_sphere])

    for i in range(HEXAGON_PARTS // 10, HEXAGON_PARTS - 1):

        radius = INNER_RADIUS - (OUTER_RADIUS - INNER_RADIUS) / 2 - hexagon_heigth * 2
        rotation_matrix = Rotation.from_euler('ZX', np.array([theta_k[i], sigma_k[i]])).as_matrix()
        rotation_matrix_eye = np.eye(4)
        rotation_matrix_eye[:3, :3] = rotation_matrix
        hexagon_cut_part = hexagon_cut.copy()

        translation = np.matmul(rotation_matrix, np.array([0, 0, radius]))
        translation = trimesh.transformations.translation_matrix(translation)
        hexagon_cut_part.apply_transform(rotation_matrix_eye)
        hexagon_cut_part.apply_transform(translation)
        hull = trimesh.boolean.difference([hull, hexagon_cut_part])

        hexagon_save_part = hexagon.copy()

        rotation_matrix_y = Rotation.from_euler('Y', 180, degrees=True).as_matrix()
        rotation_matrix_eye_y = np.eye(4)
        rotation_matrix_eye_y[:3, :3] = rotation_matrix_y
        translation = np.matmul(rotation_matrix, np.array([0, 0, radius + (OUTER_RADIUS - INNER_RADIUS) *2 + HEXAGON_TOP_THICKNESS * 2]))
        translation = trimesh.transformations.translation_matrix(translation)
        hexagon_save_part.apply_transform(rotation_matrix_eye_y)
        hexagon_save_part.apply_transform(rotation_matrix_eye)
        hexagon_save_part.apply_transform(translation)
        hexagon_save_part.export(SAVE_FOLDER / f'part_{i}.stl')

    hull.export(SAVE_FOLDER / 'hull.stl')

    # TODO: add platform for mounting

if __name__ == '__main__':
    main()
