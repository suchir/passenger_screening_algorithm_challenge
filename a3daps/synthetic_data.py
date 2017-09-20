from common.caching import read_input_dir, cached

from collections import Counter

import os
import subprocess
import glob



@cached(version=4)
def render_synthetic_body_zone_data(mode):
    with read_input_dir('makehuman/a3daps/meshes'):
        mesh_paths = ['%s/%s' % (os.getcwd(), x) for x in glob.glob('*.mhx2')]
    if mode == 'sample':
        mesh_paths = mesh_paths[:10]
    done = Counter([x.split('_')[0] for x in glob.glob('*.png')])
    done = set([x for x, y in done.items() if y == 128])
    mesh_paths = [x for x in mesh_paths if x.split('/')[-1].split('.')[0] not in done]

    if mesh_paths:
        with read_input_dir('makehuman/a3daps/textures'):
            texture_path = os.getcwd() + '/colors.png'
        with read_input_dir('scripts/a3daps/makehuman'):
            script_path = os.getcwd() + '/render_synthetic_body_zone_data.py'

        with open('filepaths.txt', 'w') as f:
            f.write('\n'.join([texture_path] + mesh_paths))
        subprocess.call(['blender', '--python', script_path, '--background'])