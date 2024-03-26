from artistlib import API, SAVEMODES
from artistlib.trajectory import sphere_trajectory
from pathlib import Path
import json
import numpy as np


SCENE_PATH = Path.cwd() / 'experiments' / 'experiment_00_simple' / 'scene_experiment_00.aRTist'
MESEHES_DICT_PATH = Path.cwd() / 'meshes' / 'data_dict.json'
FOD_MM = 1000.
FDD_MM = 2000.
NUMBER_OF_PROJECTIONS = 500
SAVE_FOLDER = Path.cwd() / 'experiments' / 'experiment_00_simple' / 'data'
MIN_Z_POSITION_MM = -70
MAX_Z_POSITION_MM = 60



def main():
    # Make save folder, clear scene and load scene
    SAVE_FOLDER.mkdir(exist_ok=True)
    artist = API()
    artist.clear_scene()
    artist.load_project(SCENE_PATH)
    artist.load_part(SCENE_PATH.parent / 'Dragon.stl', 'Al', 'Balerion')
    
    # Load meshes into scene
    with open(str(MESEHES_DICT_PATH), 'r') as f:
        data_dict = json.load(f)

    artist.load_part(Path(data_dict['hull']['path']), 'PE (HDPE)')
    artist.load_part(Path(data_dict['sample_holder']['path']), 'W')
    for hexagon_dict in data_dict['hexagon_poses']:
        position = np.array(hexagon_dict['matrix'])[:3, 3]
        z_position = position[2]
        if MIN_Z_POSITION_MM > z_position or  z_position > MAX_Z_POSITION_MM:
            artist.load_part(Path(hexagon_dict['part']), 'W')

    # generate
    source_positions, detector_positions, orientation = sphere_trajectory(FOD_MM, FDD_MM, NUMBER_OF_PROJECTIONS)
    for i in range(100, NUMBER_OF_PROJECTIONS-100):
        artist.translate('S', *source_positions[i])
        artist.translate('D', *detector_positions[i])
        artist.rotate_from_rotation_matrix('S', orientation[i])
        artist.rotate_from_rotation_matrix('D', orientation[i])
        artist.save_image(SAVE_FOLDER / f'projection_{i:04}.tif', 
                          save_mode=SAVEMODES.UINT16, 
                          save_projection_geometry=True)


if __name__ == '__main__':
    main()
