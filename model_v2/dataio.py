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
import skimage.measure
import cv2


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


@cached(get_threat_heatmaps, version=7, subdir='ssd', cloud_cache=True)
def get_augmented_threat_heatmaps(mode):
    if not os.path.exists('done'):
        th_in = get_threat_heatmaps(mode)
        f = h5py.File('data.hdf5', 'w')
        th = f.create_dataset('th', (len(th_in), 16, 660, 512, 6))

        def segmentation_mask(masks):
            ret = np.zeros((16, 660, 512, 2))
            for i in range(16):
                for j in range(3):
                    cur = masks[..., i, j]
                    if not cur.any():
                        continue
                    ret[i, ..., 0] += cur / np.max(cur)
                    ret[i, ..., 1] += cur / np.sum(cur)
            return ret

        def com_mask(masks):
            ret = np.zeros((16, 660, 512, 2))
            for i in range(16):
                for j in range(3):
                    cur = masks[..., i, j]
                    if not cur.any():
                        continue
                    M = skimage.measure.moments(cur.astype('double'))
                    xb, yb = M[0, 1]/M[0, 0], M[1, 0]/M[0, 0]
                    u20 = M[2, 0]/M[0, 0] - yb**2
                    u02 = M[0, 2]/M[0, 0] - xb**2
                    u11 = M[1, 1]/M[0, 0] - xb*yb
                    cov = np.array([[u02, u11], [u11, u20]])
                    covinv = np.linalg.inv(cov)
                    mean = np.array([xb, yb])
                    gx, gy = np.meshgrid(np.arange(512), np.arange(660))
                    g = np.reshape(np.stack([gy, gx], axis=-1), (-1, 2))
                    g = np.exp(-2*np.sum((g-mean).dot(covinv)*(g-mean), axis=1))
                    g = np.reshape(g, (660, 512))
                    ret[i, ..., 0] += g / np.max(g)
                    ret[i, ..., 1] += g / np.sum(g)
            return ret

        def distance_mask(masks):
            ret = np.zeros((16, 660, 512, 2))
            for i in range(16):
                for j in range(3):
                    cur = (masks[..., i, j]*255).astype('uint8')
                    if not cur.any():
                        continue
                    g = cv2.distanceTransform(cur, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                    ret[i, ..., 0] += g / np.max(g)
                    ret[i, ..., 1] += g / np.sum(g)
            return ret

        mean = np.zeros(6)
        for i, data in enumerate(tqdm.tqdm(th_in)):
            th[i, ..., 0:2] = segmentation_mask(data)
            th[i, ..., 2:4] = com_mask(data)
            th[i, ..., 4:6] = distance_mask(data)
            mean += np.mean(th[i], axis=(0, 1, 2)) / len(th)

        np.save('mean.npy', mean)
        f.close()
        open('done', 'w').close()

    f = h5py.File('data.hdf5', 'r')
    th = f['th']
    mean = np.load('mean.npy')
    return th, mean


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
