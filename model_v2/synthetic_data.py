from common.caching import read_input_dir, cached, read_log_dir
from common.dataio import get_aps_data_hdf5, get_passenger_clusters, get_data

from . import dataio

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import skimage.io
import skimage.color
import glob
import os
import tqdm
import h5py
import pickle
import imageio
import math
import time
import subprocess
import json


@cached(version=0)
def generate_random_models(n_models):
    with read_input_dir('makehuman/passengers'):
        ranges = defaultdict(lambda: [float('inf'), float('-inf')])
        for file in glob.glob('*.mhm'):
            with open(file, 'r') as f:
                modifiers = f.readlines()[4:-5]
                for modifier in modifiers:
                    _, m, x = modifier.split(' ')
                    x = float(x)
                    r = ranges[m]
                    r[0], r[1] = min(r[0], x), max(r[1], x)

    np.random.seed(0)
    for i in range(n_models):
        lines = ['version v1.1.1']
        for modifier in ranges:
            val = np.random.uniform(*ranges[modifier])
            lines.append('modifier %s %s' % (modifier, val))
        lines.append('skeleton game_engine.mhskel')
        with open('%s.mhm' % i, 'w') as f:
            f.write('\n'.join(lines))


BODY_ZONE_COLORS = np.array([
    [255, 255, 255],
    [255, 115, 35],
    [55, 64, 197],
    [32, 168, 67],
    [116, 116, 116],
    [255, 193, 17],
    [255, 164, 194],
    [172, 226, 28],
    [193, 183, 227],
    [142, 212, 231],
    [255, 240, 3],
    [234, 25, 33],
    [176, 110, 77],
    [232, 219, 164],
    [101, 135, 182],
    [255, 3, 255],
    [125, 0, 21],
    [153, 64, 154]
])


def _convert_colors_to_label(image):
    highlight = lambda color: np.sum(np.abs(image-color), axis=-1)
    dist = np.stack([highlight(color) for color in BODY_ZONE_COLORS], axis=-1)
    return np.argmin(dist, axis=-1)


@cached(generate_random_models, subdir='ssd', version=0)
def render_synthetic_zone_data(mode):
    assert mode in ('all', 'sample_large', 'sample')
    if not os.path.exists('done'):
        with read_input_dir('makehuman/generated'):
            mesh_paths = sorted(['%s/%s' % (os.getcwd(), x) for x in glob.glob('*.mhx2')])
        if mode == 'sample_large':
            mesh_paths = mesh_paths[:100]
        elif mode == 'sample':
            mesh_paths = mesh_paths[:10]

        with read_input_dir('hand_labeling/blender'):
            texture_path = os.getcwd() + '/zones.png'
        with read_input_dir('scripts/blender'):
            script_path = os.getcwd() + '/render_synthetic_data.py'

        angles = 16
        with open('config.json', 'w') as f:
            json.dump({
                'num_angles': angles,
                'texture_path': texture_path,
                'mesh_paths': mesh_paths
            }, f)
        subprocess.check_call(['blender', '--python', script_path, '--background'])

        f = h5py.File('data.hdf5', 'w')
        dset = f.create_dataset('dset', (len(mesh_paths), angles, 330, 256, 2))

        for i, file in enumerate(tqdm.tqdm(glob.glob('*_depth.png'))):
            zones_file = file.replace('depth', 'zones')
            angle = int(file.split('_')[-2])
            dset[i//angles, angle, ..., 0] = skimage.color.rgb2gray(skimage.io.imread(file))
            zones = skimage.io.imread(zones_file)
            labels = _convert_colors_to_label(zones[..., :3])
            dset[i//angles, angle, ..., 1] = labels

        open('done', 'w').close()
    else:
        f = h5py.File('data.hdf5', 'r')
        dset = f['dset']
    return dset