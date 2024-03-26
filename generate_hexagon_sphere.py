import trimesh
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import json



OUTER_RADIUS = 230
INNER_RADIUS = 220
HEXAGON_PARTS = 55 # number of Hexagon parts
HEXAGON_TOP_THICKNESS = 10 # top additional. The overall hexagon thickness is outer_r - innner_r + thickness
SIDE_LENGTH_FAKTOR = 0.99 # Fit in sphere holes
TOP_LENGTH_FACTOR = 1.25 # extended top hexagons
BOTTOM_HEIGHT = 40 # Heigth of the bottom sample holder
SAVE_FOLDER = Path('./meshes')
DATAFORMAT = '.obj'


def calculate_theta_k(theta_k_prior: float, h_k: float, N: float) -> float:
    return (theta_k_prior + 3.6 / np.sqrt(N) * 1. / np.sqrt(1. - h_k ** 2)) % (2 * np.pi)


def calculate_single_vector(theta, sigma):
        vector = np.array([0, 0, 1]).reshape((3, 1))
        vector = Rotation.from_euler('z', theta).as_matrix() @ Rotation.from_euler('x', sigma).as_matrix() @ vector
        return vector


def main():
    data_dict = dict()

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
    hexagon_path = SAVE_FOLDER / f'hexagon{DATAFORMAT}'
    hexagon.export(hexagon_path)

    side_length = distance_mm * SIDE_LENGTH_FAKTOR * 1.05
    angle_offset = np.pi / 6
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle_offset
    vertices = np.column_stack([side_length * np.cos(angles), side_length * np.sin(angles)])
    faces = [(0, i, (i + 1) % 8) for i in range(1, 5)]
    hexagon_cut = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    hexagon_cut = trimesh.creation.extrude_triangulation(hexagon_cut.vertices, hexagon_cut.faces, hexagon_heigth * 4)

    hull = trimesh.boolean.difference([outer_sphere, inner_sphere])

    hexagon_trafo = list()

    for i in range(HEXAGON_PARTS // 10, HEXAGON_PARTS-(HEXAGON_PARTS // 10)):

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

        hexagon_transformation = rotation_matrix_eye_y.copy()
        hexagon_transformation = np.matmul(rotation_matrix_eye, hexagon_transformation)
        hexagon_transformation: np.ndarray = np.matmul(translation, hexagon_transformation)
        hexagon_save_part.apply_transform(hexagon_transformation)

        hexagon_save_part_path = SAVE_FOLDER / f'part_{i}{DATAFORMAT}'
        hexagon_save_part.export(hexagon_save_part_path)

        hexagon_trafo.append({
            'matrix': hexagon_transformation.tolist(),
            'untransformed': str(hexagon_path),
            'part': str(hexagon_save_part_path)
        })

    data_dict['hexagon_poses'] = hexagon_trafo


    cut_out_top = trimesh.creation.cylinder(OUTER_RADIUS * 1 / 3, OUTER_RADIUS * 2)
    cut_out_top.apply_translation(np.array([0, 0, -OUTER_RADIUS]))
    hull = trimesh.boolean.difference([hull, cut_out_top])

    bottom = trimesh.creation.cylinder(OUTER_RADIUS * 1 / 3, BOTTOM_HEIGHT)
    bottom.apply_translation(np.array([0, 0, OUTER_RADIUS - (BOTTOM_HEIGHT * 0.5)]))


    bottom_hexagon: trimesh.Trimesh = hexagon.copy()
    
    
    bottom_transformation = trimesh.transformations.scale_and_translate(1, np.array([0, 0, OUTER_RADIUS - BOTTOM_HEIGHT/2 - HEXAGON_TOP_THICKNESS*2]))
    bottom_hexagon.apply_transform(bottom_transformation)

    bottom = trimesh.boolean.difference([bottom, bottom_hexagon])
    bottom_hexagon.apply_scale(SIDE_LENGTH_FAKTOR)

    hull = trimesh.boolean.union([hull, bottom])

    screws1 = trimesh.creation.cylinder(6, OUTER_RADIUS * 3)
    screws2 = screws1.copy()

    screws1.apply_translation([25, 0, 0])
    screws2.apply_translation([-25, 0, 0])

    hull = trimesh.boolean.difference([hull, screws1])
    hull = trimesh.boolean.difference([hull, screws2])
    bottom_hexagon = trimesh.boolean.difference([bottom_hexagon, screws1])
    bottom_hexagon = trimesh.boolean.difference([bottom_hexagon, screws2])

    bottom_hexagon_path = SAVE_FOLDER / f'sample_holder_mount{DATAFORMAT}'
    bottom_hexagon.export(bottom_hexagon_path)

    hull_path = SAVE_FOLDER / f'hull{DATAFORMAT}'
    hull.export(hull_path)

    data_dict['hull'] = {'path': str(hull_path),
                         'matrix': np.eye(4).tolist()}
    data_dict['sample_holder'] = {'path': str(bottom_hexagon_path),
                                  'matrix': np.eye(4).tolist()}


    with open(str(SAVE_FOLDER / 'data_dict.json'), 'w') as f:
        json.dump(data_dict, f, indent=4)

if __name__ == '__main__':
    main()
