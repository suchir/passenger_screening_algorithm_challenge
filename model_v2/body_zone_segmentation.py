from common.caching import read_input_dir, cached, read_log_dir
from common.dataio import get_aps_data_hdf5, get_passenger_clusters, get_data

from . import dataio
from . import tf_models

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
import time


@cached(get_data, get_aps_data_hdf5, subdir='ssd', version=4)
def get_a3d_projection_data(mode, percentile):
    if not os.path.exists('done'):
        angles, width, height = 16, 512, 660
        tf.reset_default_graph()

        data_in = tf.placeholder(tf.float32, [width//2, width//2, height//2])
        angle = tf.placeholder(tf.float32, [])

        with tf.device('/cpu:0'):
            image = tf.contrib.image.rotate(data_in, -2*math.pi*angle/angles)
        max_proj = tf.reduce_max(image, axis=1)
        mean_proj, var_proj = tf.nn.moments(image, axes=[1])
        std_proj = tf.sqrt(var_proj)

        surf = image > tf.contrib.distributions.percentile(image, percentile, axis=1,
                                                           keep_dims=True)
        dmap = tf.cast(tf.argmax(tf.cast(surf, tf.int32), axis=1) / width, tf.float32)
        proj = tf.image.rot90(tf.stack([dmap, max_proj, mean_proj, std_proj], axis=-1))

        gen = get_data(mode, 'a3d')
        f = h5py.File('data.hdf5', 'w')
        dset = f.create_dataset('dset', (len(gen), angles, height//2, width//2, 5))
        names, labels, dset_in = get_aps_data_hdf5(mode)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i, (_, _, data) in enumerate(get_data(mode, 'a3d')):
                data = (data[::2,::2,::2]+data[::2,::2,1::2]+data[::2,1::2,::2]+
                        data[::2,1::2,1::2]+data[1::2,::2,::2]+data[1::2,::2,1::2]+
                        data[1::2,1::2,::2]+data[1::2,1::2,1::2])/8
                for j in tqdm.trange(angles):
                    dset[i, j, ..., :-1] = sess.run(proj, feed_dict={data_in: data, angle: j})
                    dset[i, j, ..., -1] = (dset_in[i, ::2, ::2, j]+dset_in[i, ::2, 1::2, j]+
                                           dset_in[i, 1::2, ::2, j]+dset_in[i, 1::2, 1::2, j])


        with open('pkl', 'wb') as f:
            pickle.dump((names, labels), f)
        open('done', 'w').close()
    else:
        with open('pkl', 'rb') as f:
            names, labels = pickle.load(f)
        f = h5py.File('data.hdf5', 'r')
        dset = f['dset']
    return names, labels, dset


@cached(get_a3d_projection_data, subdir='ssd', version=2)
def get_mask_training_data():
    if not os.path.exists('done'):
        names, labels, dset_in = get_a3d_projection_data('sample_large', 97)
        f = h5py.File('data.hdf5', 'w')
        dset = f.create_dataset('dset', (len(dset_in), 330, 256, 6))
        name_idx = {x: i for i, x in enumerate(names)}

        with read_input_dir('hand_labeling/a3d_projections'):
            for file in tqdm.tqdm(glob.glob('*')):
                name, angle = file.replace('.png', '').split('_')
                angle = int(angle)
                angle = 0 if angle == 0 else 16 - angle

                image = imageio.imread(file)
                mask = np.all(image == [255, 0, 0], axis=-1)
                idx = name_idx[name]
                dset[idx, ..., :-1] = dset_in[idx, angle]
                dset[idx, ..., -1] = mask

        with open('pkl', 'wb') as f:
            pickle.dump((names, labels), f)
        open('done', 'w').close()
    else:
        with open('pkl', 'rb') as f:
            names, labels = pickle.load(f)
        f = h5py.File('data.hdf5', 'r')
        dset = f['dset']
    return names, labels, dset


@cached(get_mask_training_data, version=6)
def train_mask_segmentation_cnn(duration, learning_rate=1e-3, model='logistic', min_res=4,
                                num_filters=16):
    assert model in ('logistic', 'hourglass')
    angles, height, width, res, filters = 16, 330, 256, 256, 6

    tf.reset_default_graph()

    data_in = tf.placeholder(tf.float32, [None, height, width, filters])

    # random resize
    size = tf.random_uniform([2], minval=int(0.75*res), maxval=res, dtype=tf.int32)
    h_pad, w_pad = (res-size[0])//2, (res-size[1])//2
    padding = [[0, 0], [h_pad, res-size[0]-h_pad], [w_pad, res-size[1]-w_pad]]
    data = tf.image.resize_images(data_in, size)
    data = tf.stack([tf.pad(data[..., i], padding) for i in range(filters)], axis=-1)

    # random left-right flip
    flip_lr = tf.random_uniform([], maxval=2, dtype=tf.int32)
    data = tf.cond(flip_lr > 0, lambda: data[:, :, ::-1, :], lambda: data)

    # input normalization
    scales = [10, 1000, 10000, 10000, 1000, 1]
    data = tf.stack([data[..., i] * scales[i] for i in range(filters)], axis=-1)

    # get logits
    if model == 'logistic':
        logits = tf.layers.conv2d(data[..., :-1], 1, 1, 1, padding='same')
    else:
        _, logits = tf_models.hourglass_cnn(data[..., :-1], res, min_res, res, num_filters)
    # segmentation logloss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data[..., -1:],
                                                                  logits=logits))

    # actual predictions
    preds = tf.sigmoid(logits)
    preds = tf.cond(flip_lr > 0, lambda: preds[:, :, ::-1, :], lambda: preds)
    preds = preds[:, padding[1][0]:-padding[1][1]-1, padding[2][0]:-padding[2][0]-1, :]
    preds = tf.squeeze(tf.image.resize_images(preds, [height, width]))

    # optimization
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_step = optimizer.minimize(loss)
    train_summary = tf.summary.scalar('train_loss', loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    _, _, dset_all = get_mask_training_data()
    dset_train = dset_all[:80]
    dset_valid = dset_all[80:]

    def predict(dset, num_sample=16):
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            for cur_data in tqdm.tqdm(dset):
                cur_data = np.concatenate([cur_data, np.zeros(cur_data.shape[:-1] + (1,))], axis=-1)
                pred = np.zeros((angles, height, width))
                for _ in range(num_sample):
                    pred += sess.run(preds, feed_dict={data_in: cur_data})
                yield pred / num_sample

    if os.path.exists('done'):
        return predict

    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def batch_gen(x):
        batch_size = 1
        for i in tqdm.trange(0, len(x), batch_size):
            yield x[i:i+batch_size]

    def eval_model(sess):
        losses = []
        for cur_data in batch_gen(dset_valid):
            losses.append(sess.run(loss, feed_dict={data_in: cur_data}))
        return np.mean(losses)

    def train_model(sess):
        it = 0
        t0 = time.time()
        best_valid_loss = None
        while time.time() - t0 < duration * 3600:
            for cur_data in batch_gen(dset_train):
                _, cur_train_summary = sess.run([train_step, train_summary], feed_dict={
                    data_in: cur_data
                })
                writer.add_summary(cur_train_summary, it)
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

    return predict