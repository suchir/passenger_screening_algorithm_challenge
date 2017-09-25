from common.caching import read_input_dir, cached

from . import dataio
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import glob
import json
import h5py
import skimage.io
import skimage.transform
import skimage.color
import tqdm
import random
import itertools
import skimage.transform


def _convert_colors_to_label(image):
    colors = np.array([
        [0, 0, 0],
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
    ]) / 255.0
    tol = 0.1

    highlight = lambda color: np.sum(np.abs(image[..., 0:3]-color), axis=-1)
    dist = np.stack([highlight(color) for color in colors], axis=-1)
    dist, ret = np.min(dist, axis=-1), np.argmin(dist, axis=-1)
    ret[dist > tol] = 0
    return ret


@cached(version=8)
def render_synthetic_body_zone_data(mode):
    assert mode in ('sample', 'sample_large', 'all')

    num_angles = 64
    image_size = 256
    with read_input_dir('makehuman/a3daps/meshes'):
        mesh_paths = sorted(['%s/%s' % (os.getcwd(), x) for x in glob.glob('*.mhx2')])
        random.seed(0)
        random.shuffle(mesh_paths)
    if mode == 'sample':
        mesh_paths = mesh_paths[:10]
        num_angles = 8
    elif mode == 'sample_large':
        mesh_paths = mesh_paths[:100]

    done = Counter([x.split('_')[0] for x in glob.glob('*.png')])
    done = set([x for x, y in done.items() if y == 2*num_angles])
    todo = [x for x in mesh_paths if x.split('/')[-1].split('.')[0] not in done]

    if todo:
        with read_input_dir('makehuman/a3daps/textures'):
            texture_path = os.getcwd() + '/colors.png'
        with read_input_dir('scripts/a3daps/makehuman'):
            script_path = os.getcwd() + '/render_synthetic_body_zone_data.py'

        with open('config.json', 'w') as f:
            json.dump({
                'num_angles': num_angles,
                'texture_path': texture_path,
                'mesh_paths': todo
            }, f)
        subprocess.call(['blender', '--python', script_path, '--background'])
    assert len(glob.glob('*.png')) == 2*num_angles*len(mesh_paths)

    if not os.path.exists('done'):
        f = h5py.File('data.hdf5', 'w')
        x = f.create_dataset('x', (num_angles*len(mesh_paths), 2, image_size, image_size))
        y = np.zeros(num_angles*len(mesh_paths), dtype='int32')

        for i, file in enumerate(tqdm.tqdm(glob.glob('*_skin.png'))):
            color_file = file.replace('skin', 'color')
            image = skimage.color.rgb2gray(skimage.io.imread(file))
            image = skimage.transform.resize(image, (image_size, image_size))
            color = skimage.io.imread(color_file)
            color = skimage.transform.resize(color, (image_size, image_size))
            labels = _convert_colors_to_label(color[..., 0:3])
            x[i, 0, ...] = image
            x[i, 1, ...] = labels
            y[i] = int(file.split('_')[1])

        np.save('y.npy', y)
        open('done', 'w').close()
    else:
        f = h5py.File('data.hdf5', 'r')
        x = f['x']
        y = np.load('y.npy')
    return x, y


@cached(render_synthetic_body_zone_data, version=0)
def get_real_and_fake_images(mode):
    if mode.startswith('sample'):
        chunk_size = 128
    else:
        chunk_size = 1024
    image_size = 256
    num_angles = 64

    if not os.path.exists('done'):
        def get_real_images_chunk():
            ret = []
            while True:
                for _, _, data in dataio.get_data(mode):
                    ret.append(data)
                    if len(ret) * num_angles >= chunk_size:
                        ret = np.concatenate(ret, axis=-1)[..., :chunk_size]
                        ret = skimage.transform.resize(ret, (image_size, image_size))
                        ret = np.swapaxes(ret, 0, 2)[:, ::-1, :]
                        yield ret
                        ret = []

        fake_images = render_synthetic_body_zone_data(mode)
        f = h5py.File('data.hdf5', 'w')
        x = f.create_dataset('x', fake_images.shape)

        for i, real_chunk in zip(range(0, len(fake_images), chunk_size), get_real_images_chunk()):
            rem = min(i+chunk_size, len(fake_images)) - i
            real, fake = real_chunk[:rem], fake_images[i:i+rem, 0, ...]
            np.random.shuffle(real)
            np.random.shuffle(fake)
            x[i:i+rem, 0, ...] = real
            x[i:i+rem, 1, ...] = fake

        open('done', 'w').close()
    else:
        f = h5py.File('data.hdf5', 'r')
        x = f['x']
    return x
