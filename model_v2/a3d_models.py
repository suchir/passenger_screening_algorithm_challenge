from common.caching import cached, read_log_dir
from common.math import sigmoid, log_loss
from common.dataio import get_train_idx, get_valid_idx, get_data

from . import tf_models
from . import dataio
from . import passenger_clustering

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import datetime
import time
import tqdm
import math
import random
import h5py
import skimage.transform


@cached(get_data, subdir='ssd', version=0)
def get_downsized_a3d_data(mode, downsize=4):
    if not os.path.exists('done'):
        gen = get_data(mode, 'a3d')
        f = h5py.File('data.hdf5', 'w')
        dset = f.create_dataset('dset', (len(gen), 512//downsize, 512//downsize, 660//downsize))

        for i, (_, _, data) in enumerate(tqdm.tqdm(gen)):
            dset[i] = data[::downsize, ::downsize, ::downsize]

        f.close()
        open('done', 'w').close()

    f = h5py.File('data.hdf5', 'r')
    dset = f['dset']
    return dset



@cached(get_downsized_a3d_data, dataio.get_augmented_threat_heatmaps, version=7)
def train_1d_cnn(mode, cvid, duration, learning_rate=1e-3):
    tf.reset_default_graph()
    width, depth, height = 128, 128, 165

    a3d_in = tf.placeholder(tf.float32, [width, depth, height])
    labels_in = tf.placeholder(tf.float32, [height, width])

    a3d = tf.transpose(a3d_in, [2, 0, 1])[::-1]
    a3d = tf.reshape(a3d, [-1, depth, 1]) * 1000
    logits = tf_models.cnn_1d(a3d, 64, 4)
    labels = tf.reshape(labels_in, [-1])

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    train_summary = tf.summary.scalar('train_loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    dset_all = get_downsized_a3d_data(mode)
    labels_train = dataio.get_threat_heatmaps('train-%s' % cvid)
    labels_valid = dataio.get_threat_heatmaps('valid-%s' % cvid)
    train_idx, valid_idx = get_train_idx(mode, cvid), get_valid_idx(mode, cvid)

    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def data_gen(dset, labels, idx):
        for i, label in zip(tqdm.tqdm(idx), labels):
            yield {
                a3d_in: dset[i],
                labels_in: np.sum(label[::4, ::4, 0], axis=-1)
            }

    def eval_model(sess):
        losses = []
        for data in data_gen(dset_all, labels_valid, valid_idx):
            cur_loss = sess.run(loss, feed_dict=data)
            losses.append(cur_loss)
        return np.mean(losses)

    def train_model(sess):
        it = 0
        t0 = time.time()
        best_valid_loss = None
        while time.time() - t0 < duration * 3600:
            for data in data_gen(dset_all, labels_train, train_idx):
                _, cur_summary = sess.run([train_step, train_summary], feed_dict=data)
                writer.add_summary(cur_summary, it)
                it += 1

            valid_loss = eval_model(sess)
            cur_valid_summary = tf.Summary()
            cur_valid_summary.value.add(tag='valid_loss', simple_value=valid_loss)
            writer.add_summary(cur_valid_summary, it)

            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                saver.save(sess, model_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_model(sess)

    open('done', 'w').close()
