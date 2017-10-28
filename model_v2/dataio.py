from common.caching import read_input_dir, cached
from common.dataio import get_aps_data_hdf5

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import glob
import os
import tqdm
import h5py
import pickle
import imageio


SEGMENTATION_COLORS = np.array([[255, 0, 0], [255, 0, 255], [0, 0, 255]])


@cached(get_aps_data_hdf5, version=0, subdir='ssd')
def get_threat_heatmaps(mode):
    if not os.path.exists('done'):
        names, _, x = get_aps_data_hdf5(mode)
        f = h5py.File('data.hdf5', 'w')
        th = f.create_dataset('th', x.shape + (3,))

        with read_input_dir('hand_labeling/threat_segmentation/base'):
            for i, (name, data) in tqdm.tqdm(enumerate(zip(names, x)), total=len(x)):
                files = glob.glob(name + '*')
                assert files, 'missing hand segmentation for %s' % name
                image = imageio.imread(files[0])
                masks = []
                for ci in range(3):
                    mask = np.all(image[..., :3] == SEGMENTATION_COLORS[ci], axis=-1)
                    mask = np.stack(np.split(mask, 16, axis=1), axis=-1)
                    masks.append(mask)
                th[i] = np.stack(masks, axis=-1)

        open('done', 'w').close()
    else:
        f = h5py.File('data.hdf5')
        th = f['th']
    return th


@cached(get_aps_data_hdf5, get_threat_heatmaps, version=0, subdir='ssd')
def get_data_and_threat_heatmaps(mode):
    names, labels, x = get_aps_data_hdf5(mode)
    if not os.path.exists('done'):
        th = get_threat_heatmaps(mode)
        f = h5py.File('data.hdf5', 'w')
        dset = f.create_dataset('dset', x.shape + (4,))
        for i, (data, hmap) in tqdm.tqdm(enumerate(zip(x, th)), total=len(x)):
            dset[i] = np.concatenate([data[..., np.newaxis], hmap], axis=-1)
        open('done', 'w').close()
    else:
        f = h5py.File('data.hdf5')
        dset = f['dset']
    return names, labels, dset