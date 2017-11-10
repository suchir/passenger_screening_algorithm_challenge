from common.caching import read_input_dir, cached
from common.dataio import get_aps_data_hdf5, get_passenger_clusters

from . import dataio

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import glob
import os
import tqdm
import h5py
import pickle
import imageio
import time
import multiprocessing
import common.pyelastix
import heapq


@cached(get_passenger_clusters, dataio.get_data_and_threat_heatmaps, version=0, subdir='ssd')
def get_clustered_data_and_threat_heatmaps(mode, cluster_type):
    assert cluster_type in ('groundtruth')

    if not os.path.exists('done'):
        clusters = get_passenger_clusters()

        names_in, labels_in, dset_in = dataio.get_data_and_threat_heatmaps(mode)
        clusters = [[y for y in x if y in names_in] for x in clusters]
        clusters = [x for x in clusters if x]
        names = sum(clusters, [])
        names_in_idx = {name: i for i, name in enumerate(names_in)}
        perm = [names_in_idx[name] for name in names]
        labels = np.stack([labels_in[i] for i in perm])
        ranges = [(0, len(x)) for x in clusters]
        for i in range(1, len(ranges)):
            ranges[i] = (ranges[i][0]+ranges[i-1][1], ranges[i][1]+ranges[i-1][1])

        f = h5py.File('data.hdf5', 'w')
        dset = f.create_dataset('dset', dset_in.shape)
        for i in tqdm.trange(len(dset)):
            dset[i] = dset_in[perm[i]]

        with open('pkl', 'wb') as f:
            pickle.dump((ranges, names), f)
        np.save('labels.npy', labels)
        open('done', 'w').close()
    else:
        with open('pkl', 'rb') as f:
            ranges, names = pickle.load(f)
        labels = np.load('labels.npy')
        f = h5py.File('data.hdf5', 'r')
        dset = f['dset']
    return ranges, names, labels, dset


def get_passenger_groups(mode):
    assert not mode.startswith('test')

    clusters = get_passenger_clusters()
    names, _, _ = get_aps_data_hdf5(mode)
    group = [None] * len(names)
    for i in range(len(group)):
        for j, cluster in enumerate(clusters):
            if names[i] in cluster:
                group[i] = j
    return group


@cached(get_aps_data_hdf5, version=3)
def get_distance_matrix(mode):
    if not os.path.exists('done'):
        batch_size = 32

        tf.reset_default_graph()
        x1_in = tf.placeholder(tf.float32, [None, 660, 512, 16])
        x2_in = tf.placeholder(tf.float32, [None, 660, 512, 16])
        dist_mats = []

        for feat in range(3):
            res = 512
            if feat == 0:
                x1, x2 = x1_in, x2_in
            elif feat == 1:
                x1, x2 = x1_in[:, :330, :, :], x2_in[:, :330, :, :]
            else:
                x1, x2 = x1_in[:, :, 128:384, :], x2_in[:, :, 128:384, :]
            x1 = tf.image.resize_images(x1, [res, res])
            x2 = tf.image.resize_images(x2, [res, res])

            for _ in range(9):
                n = 16 * res**2
                x1_v = tf.reshape(x1, [-1, n])
                x2_v = tf.transpose(tf.reshape(x2, [-1, n]))
                dots = tf.matmul(x1_v, x2_v)
                diff = tf.reduce_sum(tf.square(x1_v), axis=1, keep_dims=True) - 2*dots + \
                       tf.reduce_sum(tf.square(x2_v), axis=0, keep_dims=True)

                dist = tf.sqrt(tf.maximum(diff/n, 0))
                dist_mats.append(dist)

                res //= 2
                x1 = tf.image.resize_images(x1, [res, res])
                x2 = tf.image.resize_images(x2, [res, res])

        dist_mat = tf.stack(dist_mats, axis=-1)

        _, _, dset = get_aps_data_hdf5(mode)
        dmat = np.zeros((len(dset), len(dset), 27))
        with tf.Session() as sess:
            for i in tqdm.trange(0, len(dset), batch_size):
                for j in tqdm.trange(0, len(dset), batch_size):
                    mat = sess.run(dist_mat, feed_dict={
                        x1_in: dset[i:i+batch_size],
                        x2_in: dset[j:j+batch_size]
                    })
                    dmat[i:i+batch_size, j:j+batch_size, :] = mat

        np.save('dmat.npy', dmat)
        open('done', 'w').close()
    else:
        dmat = np.load('dmat.npy')
    return dmat


@cached(get_aps_data_hdf5, get_distance_matrix, version=0)
def train_clustering_model(mode, duration):
    tf.reset_default_graph()

    dmat_in = tf.placeholder(tf.float32, [None, None, 27])
    labels_in = tf.placeholder(tf.float32, [None, None])

    dmat = tf.reshape(dmat_in, [-1, 27])
    mean, var = tf.nn.moments(dmat, [0, 1])
    dmat = (dmat - mean) / tf.sqrt(var)
    labels = tf.reshape(labels_in, [-1])
    logits = tf.squeeze(tf.layers.dense(dmat, 1))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_step = optimizer.minimize(loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def predict(x):
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            return sess.run(logits, feed_dict={dmat_in: x})

    if os.path.exists('done'):
        return predict

    dmat_train = get_distance_matrix(mode)
    clusters = get_passenger_clusters()
    names, _, _ = get_aps_data_hdf5(mode)
    name_idx = {x: i for i, x in enumerate(names)}

    labels_train = np.zeros(dmat_train.shape[:2])
    for cluster in clusters:
        for name1 in cluster:
            for name2 in cluster:
                i1, i2 = name_idx[name1], name_idx[name2]
                labels_train[i1, i2] = 1

    def train_model(sess, duration):
        t0 = time.time()
        while time.time() - t0 < duration * 3600:
            sess.run([train_step, logits, loss], feed_dict={
                dmat_in: dmat_train, labels_in: labels_train
            })
        saver.save(sess, model_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_model(sess, duration)
    open('done', 'w').close()

    return predict


@cached(train_clustering_model, version=0)
def get_nearest_neighbors(mode):
    if not os.path.exists('done'):
        dmat = get_distance_matrix(mode)
        predict = train_clustering_model('all', 0.25)
        pmat = np.reshape(predict(dmat), (len(dmat), -1))
        perm = np.argsort(-pmat, axis=1)

        np.save('perm.npy', perm)
        open('done', 'w').close()
    else:
        perm = np.load('perm.npy')
    return perm


@cached(get_distance_matrix, train_clustering_model, cloud_cache=True, version=0)
def get_candidate_neighbors(mode, min_neighbors):
    if not os.path.exists('done'):
        dmat = get_distance_matrix(mode)
        n = len(dmat)
        pmat = np.reshape(train_clustering_model('all', 0.25)(dmat), (n, -1))
        perm = np.argsort(-pmat, axis=1)

        conf = np.zeros(n)
        for i in range(n):
            conf[i] += np.sum(pmat[perm[i, :min_neighbors]])
        order = np.argsort(conf)

        cand = [[] for _ in range(n)]
        for i in range(n):
            n_cand = min_neighbors * int(-np.log2((i+1)/n) + 1)
            for j in range(n_cand):
                cand[order[i]].append(perm[order[i]][j])

        with open('pkl', 'wb') as f:
            pickle.dump(cand, f)
        open('done', 'w').close()
    else:
        with open('pkl', 'rb') as f:
            cand = pickle.load(f)
    return cand


def _register_images(args):
    return common.pyelastix.register(*args, verbose=0)[0]


def register_images(im1, im2, params=None):
    if params is None:
        params = common.pyelastix.get_default_params()
        params.FinalGridSpacingInPhysicalUnits = 32
        params.NumberOfResolutions = 4
        params.MaximumNumberOfIterations = 64
    if isinstance(im1, list):
        if not isinstance(im2, list):
            im2 = [im2 for _ in range(len(im1))]
        for _ in range(100):
            try:
                with multiprocessing.Pool(max(multiprocessing.cpu_count()//2, 1)) as p:
                    return p.map(_register_images, [(i1, i2, params) for i1, i2 in zip(im1, im2)])
            except:
                pass
        raise Exception('failed to register images')
    else:
        return _register_images((im1, im2, params))


@cached(get_aps_data_hdf5, get_candidate_neighbors, subdir='ssd', version=0)
def get_augmented_aps_segmentation_data(mode):
    if not os.path.exists('done'):
        names, labels, dset_in = dataio.get_data_and_threat_heatmaps(mode)
        n = len(dset_in)
        n = 32
        f = h5py.File('data.hdf5', 'w')
        dset = f.create_dataset('dset', (n, 16, 660, 512, 7))
        dset_in = dset_in[()]
        dset = dset[()]

        batch_size = 8
        for i in tqdm.trange(0, n, batch_size):
            im1, im2 = [], []
            for j in range(i, min(n, i+batch_size)):
                data = np.rollaxis(dset_in[j], 2, 0)
                dset[j, ..., 0] = data[..., 0]
                dset[j, ..., 4:] = data[..., 1:]
                rot = np.concatenate([data[0:1, :, ::-1, 0], data[-1::-1, :, ::-1, 0]])
                for k in range(16):
                    im1.append(rot[k] / rot[k].max())
                    im2.append(data[k, :, :, 0] / data[k, :, :, 0].max())

            reg = register_images(im1, im2)
            for j in range(i, min(n, i+batch_size)):
                for k in range(16):
                    dset[j, k, ..., 1] = reg[16*(j-i)+k] / reg[16*(j-i)+k].max()

        with open('pkl', 'wb') as f:
            pickle.dump((names, labels), f)
        open('done', 'w').close()
    else:
        with open('pkl', 'rb') as f:
            names, labels = pickle.load(f)
        f = h5py.File('data.hdf5')
        dset = f['dset']
    return names, labels, dset