from common.caching import cached, read_input_dir
from common.dataio import get_aps_data_hdf5

from . import dataio
from . import threat_segmentation_models

import numpy as np
import skimage.io
import tqdm
import imageio
import sklearn.cluster
import sklearn.decomposition
import os


@cached(get_aps_data_hdf5, version=1)
def write_aps_hand_labeling_images(mode):
    names, labels, x = get_aps_data_hdf5(mode)
    for name, label, data in tqdm.tqdm(zip(names, labels, x), total=len(x)):
        images = np.concatenate(np.rollaxis(data, 2), axis=1) / data.max()
        filename = '_'.join([name] + [str(i+1) for i in range(17) if label and label[i]])
        skimage.io.imsave('%s.png' % filename, np.repeat(images[..., np.newaxis], 3, axis=-1))


@cached(get_aps_data_hdf5, version=0)
def write_aps_hand_labeling_revision_v0(mode):
    names, _, x = get_aps_data_hdf5(mode)
    todo = {}
    with read_input_dir('hand_labeling/threat_segmentation'):
        with open('revision_v0.txt', 'r') as f:
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


@cached(get_aps_data_hdf5, version=0)
def write_passenger_id_images(mode):
    names, _, x = get_aps_data_hdf5(mode)
    for name, data in tqdm.tqdm(zip(names, x), total=len(x)):
        imageio.imsave('%s.png' % name, data[..., 0] / np.max(data[..., 0]))


@cached(get_aps_data_hdf5, version=0)
def naive_cluster_passengers(mode, n_clusters):
    names, _, x = get_aps_data_hdf5(mode)
    images = x[:, ::8, ::8, 0].reshape((len(x), -1))
    reduced_data = sklearn.decomposition.PCA(n_components=128).fit_transform(images)
    kmeans = sklearn.cluster.KMeans(n_clusters).fit(reduced_data)
    clusters = kmeans.predict(reduced_data)

    for i in range(n_clusters):
        os.mkdir(str(i))
    for name, cluster, data in tqdm.tqdm(zip(names, clusters, x), total=len(x)):
        imageio.imsave('%s/%s.png' % (cluster, name), data[..., 0]/data[..., 0].max())


@cached(threat_segmentation_models.train_unet_cnn, version=1)
def write_unet_predicted_heatmaps(mode, batch_size, learning_rate, duration):
    predict = threat_segmentation_models.train_unet_cnn(mode, batch_size, learning_rate, duration)

    valid_mode = mode.replace('train', 'valid')
    names, _, dset_valid = dataio.get_data_and_threat_heatmaps(valid_mode)

    for name, data, preds in zip(names, dset_valid, predict(dset_valid)):
        for i in range(16):
            image = data[..., i, 0]
            image = np.concatenate([image, image, image], axis=-1) / np.max(image)
            image = np.stack([np.zeros(image.shape), image, np.zeros(image.shape)], axis=-1)
            image[:, 512:1024, 0] = np.sum(data[..., i, 1:], axis=-1)
            image[:, 1024:, 0] = preds[i, ...]
            imageio.imsave('%s_%s.png' % (name, i), image)
