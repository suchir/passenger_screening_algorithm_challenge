from common.caching import cached, read_input_dir
from common.dataio import get_aps_data_hdf5

from . import dataio

import numpy as np
import skimage.io
import tqdm
import imageio


@cached(get_aps_data_hdf5, version=1)
def write_aps_hand_labeling_images(mode):
    names, labels, x = get_aps_data_hdf5(mode)
    for name, label, data in tqdm.tqdm(zip(names, labels, x), total=len(x)):
        images = np.concatenate(np.rollaxis(data, 2), axis=1) / data.max()
        filename = '_'.join([name] + [str(i+1) for i in range(17) if label and label[i]])
        skimage.io.imsave('%s.png' % filename, np.repeat(images[..., np.newaxis], 3, axis=-1))


@cached(get_aps_data_hdf5, version=0)
def write_aps_hand_labeling_revision(mode, version):
    names, _, x = get_aps_data_hdf5(mode)
    todo = {}
    with read_input_dir('hand_labeling/threat_segmentation'):
        with open('revision_v%s.txt' % version, 'r') as f:
            for line in f:
                name, labels = line[:5], line[6:]
                assert name not in todo, "duplicate revision names"
                todo[name] = [int(x) for x in labels.split(', ')]
    for name, data in tqdm.tqdm(zip(names, x), total=len(x)):
        for label in todo.get(name[:5], []):
            images = np.concatenate(np.rollaxis(data, 2), axis=1) / data.max()
            filename = '%s_%s' % (name, label)
            skimage.io.imsave('%s.png' % filename, np.repeat(images[..., np.newaxis], 3, axis=-1))


@cached(dataio.get_data_and_threat_heatmaps, version=0)
def write_aps_hand_labeling_gifs(mode):
    names, labels, dset = dataio.get_data_and_threat_heatmaps(mode)
    for name, label, data in tqdm.tqdm(zip(names, labels, dset), total=len(dset)):
        frames = np.concatenate([data[..., 0], data[..., 0]], axis=1)
        frames /= np.max(frames)
        frames = np.stack([np.zeros(frames.shape), frames, np.zeros(frames.shape)], axis=-1)
        frames[:, 512:, :, 0] = np.sum(data[..., 1:], axis=-1)
        frames = np.rollaxis(frames, 2, 0)
        filename = '_'.join([name] + [str(i+1) for i in range(17) if label[i]])
        imageio.mimsave('%s.gif' % filename, frames)
