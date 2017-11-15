from common.caching import read_input_dir, cached
from common.dataio import get_aps_data_hdf5, get_passenger_clusters, get_data

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
import math


@cached(get_data, subdir='ssd', cloud_cache=True, version=1)
def get_a3d_projection_data(mode, percentile):
    if not os.path.exists('done'):
        angles, width, height = 16, 512, 660
        tf.reset_default_graph()

        data_in = tf.placeholder(tf.float32, [width, width, height])
        angle = tf.placeholder(tf.float32, [])

        image = data_in[::2, ::2, ::2]
        image = tf.contrib.image.rotate(image, 2*math.pi*angle/angles)
        max_proj = tf.reduce_max(image, axis=1)
        mean_proj, var_proj = tf.nn.moments(image, axes=[1])
        std_proj = tf.sqrt(var_proj)

        surf = image > tf.contrib.distributions.percentile(image, percentile, axis=1,
                                                           keep_dims=True)
        dmap = tf.cast(tf.argmax(tf.cast(surf, tf.int32), axis=1) / width, tf.float32)
        proj = tf.image.rot90(tf.stack([dmap, max_proj, mean_proj, std_proj], axis=-1))
        proj = tf.image.resize_images(proj, [height, width])

        gen = get_data(mode, 'a3d')
        f = h5py.File('data.hdf5', 'w')
        dset = f.create_dataset('dset', (len(gen), angles, height, width, 4))
        names, labels = [], []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i, (name, label, data) in enumerate(get_data(mode, 'a3d')):
                for j in tqdm.trange(angles):
                    dset[i, j] = sess.run(proj, feed_dict={data_in: data, angle: j})
                names.append(name)
                labels.append(label)

        with open('pkl', 'wb') as f:
            pickle.dump((names, labels), f)
        open('done', 'w').close()
    else:
        with open('pkl', 'rb') as f:
            names, labels = pickle.load(f)
        f = h5py.File('data.hdf5', 'r')
        dset = f['dset']
    return names, labels, dset
