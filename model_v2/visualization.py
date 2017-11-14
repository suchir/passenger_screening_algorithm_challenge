from common.caching import cached, read_input_dir
from common.dataio import get_aps_data_hdf5, get_passenger_clusters

from . import dataio
from . import threat_segmentation_models
from . import passenger_clustering

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import tqdm
import imageio
import sklearn.cluster
import sklearn.decomposition
import os
import common.pyelastix


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
def write_unet_predicted_heatmaps(mode, *args, **kwargs):
    predict = threat_segmentation_models.train_unet_cnn(mode, *args, **kwargs)

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


@cached(threat_segmentation_models.get_hourglass_cnn_predictions, version=0)
def write_hourglass_predicted_heatmaps(mode, *args, **kwargs):
    ranges, names, _, dset = passenger_clustering.get_clustered_data_and_threat_heatmaps(mode,
                                kwargs.get('cluster_type', 'groundtruth'))
    out = threat_segmentation_models.get_hourglass_cnn_predictions(mode, *args, **kwargs)
    for group in ranges:
        for i in range(*group):
            for j in range(16):
                image = dset[i, ..., j, 0]
                image = np.concatenate([image, image, image], axis=-1) / np.max(image)
                image = np.stack([np.zeros(image.shape), image, np.zeros(image.shape)], axis=-1)
                image[:, 512:1024, 0] = np.sum(dset[i, ..., j, 1:], axis=-1)
                image[:, 1024:, 0] = out[i, j, ...] / (group[1]-group[0]) * 2
                imageio.imsave('%s_%s.png' % (names[i], j), image)


@cached(threat_segmentation_models.train_unet_cnn, subdir='ssd', version=2)
def write_augmented_hourglass_predicted_heatmaps(mode, *args, **kwargs):
    predict = threat_segmentation_models.train_augmented_hourglass_cnn(mode, *args, **kwargs)

    valid_mode = mode.replace('train', 'valid')
    names, _, dset_valid = passenger_clustering.join_augmented_aps_segmentation_data(valid_mode, 6)

    for name, data, (preds, loss) in zip(names, dset_valid, predict(dset_valid)):
        for i in range(16):
            image = data[i, ..., 0]
            image = np.concatenate([image, image, image], axis=-1) / np.max(image)
            image = np.stack([np.zeros(image.shape), image, np.zeros(image.shape)], axis=-1)
            image[:, 512:1024, 0] = np.sum(data[i, ..., -3:], axis=-1) / 1000
            image[:, 1024:, 0] = preds[i, ...]
            imageio.imsave('%s_%s_%s.png' % (int(loss*1e6), name, i), image)


@cached(passenger_clustering.get_distance_matrix, version=0)
def plot_distance_matrix_accuracy(mode, max_near):
    dmat = passenger_clustering.get_distance_matrix(mode)
    perm = np.argsort(dmat, axis=1)
    group = passenger_clustering.get_passenger_groups()

    for res in range(dmat.shape[-1]):
        n_wrong = [0] * max_near
        for i in range(max_near):
            for j in range(len(dmat)):
                n_wrong[i] += group[j] != group[perm[j][i][res]]
            n_wrong[i] /= len(dmat)

        plt.plot(n_wrong)
        plt.savefig('%s.png' % res)
        plt.close()


@cached(passenger_clustering.get_nearest_neighbors, version=0)
def plot_nearest_neighbor_accuracy(mode, max_near):
    perm = passenger_clustering.get_nearest_neighbors(mode)
    group = passenger_clustering.get_passenger_groups()

    n_wrong = [0] * max_near
    for i in range(max_near):
        for j in range(len(perm)):
            n_wrong[i] += group[j] != group[perm[j][i]]
        n_wrong[i] /= len(perm)

    plt.plot(n_wrong)
    plt.savefig('out.png')


@cached(passenger_clustering.get_nearest_neighbors, get_aps_data_hdf5, version=0)
def plot_nearest_neighbors(mode, max_near):
    perm = passenger_clustering.get_nearest_neighbors(mode)
    group = passenger_clustering.get_passenger_groups(mode)
    names, _, dset = get_aps_data_hdf5(mode)

    for i, name in enumerate(names):
        n_wrong = sum(group[perm[i][j]] != group[i] for j in range(max_near))
        images = []
        for j in range(max_near):
            images.append(dset[perm[i][j], ::4, ::4, 0])
        rows = [np.concatenate(images[i:i+4], axis=1) for i in range(0, max_near, 4)]
        image = np.concatenate(rows, axis=0)
        imageio.imsave('%s_%s.png' % (n_wrong, name), image / image.max())


@cached(get_aps_data_hdf5, version=1)
def plot_image_registration_samples(mode, n_samples):
    names, _, dset = get_aps_data_hdf5(mode)
    group = passenger_clustering.get_passenger_groups(mode)
    for spacing in tqdm.tqdm([8, 16, 32, 64]):
        for num_res in tqdm.tqdm([2, 3, 4]):
            for num_iter in tqdm.tqdm([8, 16, 32, 64, 128]):
                np.random.seed(0)
                im1, im2 = [], []
                for i in range(n_samples):
                    while True:
                        i1, i2, angle = np.random.randint(len(dset)), np.random.randint(len(dset)), \
                                        np.random.randint(16)
                        if group[i1] == group[i2]:
                            break
                    d1, d2 = dset[i1, ..., angle], dset[i2, ..., angle]
                    d1 /= d1.max()
                    d2 /= d2.max()
                    im1.append(d1)
                    im2.append(d2)

                params = common.pyelastix.get_default_params()
                params.FinalGridSpacingInPhysicalUnits = spacing
                params.NumberOfResolutions = num_res
                params.MaximumNumberOfIterations = num_iter
                reg = passenger_clustering.register_images(im1, im2, params)

                for i, (d1, d2, im) in enumerate(zip(im1, im2, reg)):
                    im /= im.max()
                    image = np.concatenate([
                        np.concatenate([d1, d2], axis=1),
                        np.concatenate([im, np.zeros(d1.shape)], axis=1)
                    ], axis=0)
                    image = np.repeat(image[..., np.newaxis], 3, axis=-1)
                    image[660:, 512:, 0] = d2
                    image[660:, 512:, 1] = im

                    path = '%s/%s/%s' % (spacing, num_res, num_iter)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    imageio.imsave('%s/%s.png' % (path, i), image)
