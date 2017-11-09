from common.caching import read_input_dir, cached
from common.dataio import get_aps_data_hdf5, get_passenger_clusters

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


def _get_mask(image, color):
    mask = np.all(image[..., :3] == color, axis=-1)
    mask = np.stack(np.split(mask, 16, axis=1), axis=-1)
    return mask


@cached(get_aps_data_hdf5, version=2, subdir='ssd')
def get_threat_heatmaps(mode):
    if not os.path.exists('done'):
        names, labels, x = get_aps_data_hdf5(mode)
        f = h5py.File('data.hdf5', 'w')
        th = f.create_dataset('th', x.shape + (3,))

        with read_input_dir('hand_labeling/threat_segmentation/base'):
            for i, (name, label, data) in tqdm.tqdm(enumerate(zip(names, labels, x)), total=len(x)):
                files = glob.glob(name + '*')
                assert files, 'missing hand segmentation for %s' % name

                image = imageio.imread(files[0])
                masks = [_get_mask(image, SEGMENTATION_COLORS[ci]) for ci in range(3)]
                with read_input_dir('hand_labeling/threat_segmentation/revision_v0'):
                    for revision in glob.glob(name + '*'):
                        rlabel = int(revision.split('_')[1].split('.')[0])
                        rci = [i+1 for i in range(17) if label[i]].index(rlabel)
                        rimage = imageio.imread(revision)
                        masks[rci] = _get_mask(rimage, SEGMENTATION_COLORS[0])

                th[i] = np.stack(masks, axis=-1)

        open('done', 'w').close()
    else:
        f = h5py.File('data.hdf5', 'r')
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
        f = h5py.File('data.hdf5', 'r')
        dset = f['dset']
    return names, labels, dset


@cached(get_data_and_threat_heatmaps, version=0)
def sanity_check_threat_heatmaps(mode):
    names, labels, dset = get_data_and_threat_heatmaps(mode)
    for name, label, data in tqdm.tqdm(zip(names, labels, dset), total=len(dset)):
        th = data[..., 1:]
        has_t = np.any(th, axis=(0, 1, 2))
        if np.sum(has_t) != sum(label):
            print('heatmaps from %s does not match label' % name)
