from common.caching import cached, read_log_dir
from common.math import sigmoid, log_loss

from . import tf_models
from . import dataio

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import datetime
import time
import tqdm
import math
import random


@cached(dataio.get_clustered_data_and_threat_heatmaps, version=0)
def train_hourglass_cnn(mode, duration, cluster_type='groundtruth', learning_rate=1e-3,
                        single_pred=False):
    assert 'train' in mode
    height, width = 660, 512

    tf.reset_default_graph()
    images_in = tf.placeholder(tf.float32, [None, height, width, 2])
    thmap_in = tf.placeholder(tf.float32, [None, height, width, 2])

    size = tf.random_uniform([2], minval=int(0.75*width), maxval=width, dtype=tf.int32)
    h_pad, w_pad = (width-size[0])//2, (width-size[1])//2
    padding = [[0, 0], [h_pad, width-size[0]-h_pad], [w_pad, width-size[1]-w_pad]]
    images = tf.image.resize_images(images_in, size)
    images = tf.stack([tf.pad(images[..., i], padding) for i in range(2)], axis=-1)
    thmap = tf.image.resize_images(thmap_in, size)
    thmap = tf.stack([tf.pad(thmap[..., i], padding) for i in range(2)], axis=-1)

    flip_lr = tf.random_uniform([], maxval=2, dtype=tf.int32)
    images = tf.cond(flip_lr > 0, lambda: images[:, :, ::-1, :], lambda: images)
    thmap = tf.cond(flip_lr > 0, lambda: thmap[:, :, ::-1, :], lambda: thmap)

    _, logits = tf_models.hourglass_cnn(images, width, 4, width, 64, num_output=2)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=thmap, logits=logits)
    loss = tf.reduce_mean(loss[..., 0] if single_pred else loss)

    pred = tf.sigmoid(logits)
    pred = pred[:, h_pad:-(width-size[0]-h_pad), w_pad:-(width-size[1]-w_pad), :]
    pred = tf.image.resize_images(pred, (height, width))
    pred = tf.cond(flip_lr > 0, lambda: pred[:, :, ::-1, :], lambda: pred)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)
    train_summary = tf.summary.scalar('train_loss', loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def feed(data0, data1):
        data0, data1 = np.rollaxis(data0, 2, 0), np.rollaxis(data1, 2, 0)
        image0, image1 = data0[..., 0], data1[..., 0]
        thmap0, thmap1 = np.sum(data0[..., 1:], axis=-1), np.sum(data1[..., 1:], axis=-1)
        return {
            images_in: np.stack([image0, image1], axis=-1),
            thmap_in: np.stack([thmap0, thmap1], axis=-1) 
        }

    def random_pair(ranges):
        group = np.random.randint(len(ranges))
        random_i = lambda: np.random.randint(ranges[group][0], ranges[group][1])
        return random_i(), random_i()

    def random_data(ranges, dset):
        i1, i2 = random_pair(ranges)
        return dset[i1], dset[i2]

    valid_mode = mode.replace('train', 'valid')
    ranges_train, _, _, dset_train = dataio.get_clustered_data_and_threat_heatmaps(mode,
                                                                                   cluster_type)
    ranges_valid, _, _, dset_valid = dataio.get_clustered_data_and_threat_heatmaps(valid_mode,
                                                                                cluster_type)

    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def eval_model(sess):
        losses = []
        for _ in tqdm.trange(len(dset_valid)):
            cur_loss = sess.run(loss, feed_dict=feed(*random_data(ranges_valid, dset_valid)))
            losses.append(cur_loss)
        return np.mean(losses)

    def train_model(sess):
        it = 0
        t0 = time.time()
        best_valid_loss = None
        while time.time() - t0 < duration * 3600:
            for _ in tqdm.trange(len(dset_train)):
                _, cur_train_summary = sess.run([train_step, train_summary], feed_dict=
                                                feed(*random_data(ranges_train, dset_train)))
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

    return None


@cached(version=0)
def train_unet_cnn(mode, batch_size, learning_rate, duration, rotate_images=False,
                   include_reflection=False, conv3d=False, refine2d=False, refine3d=False,
                   model='unet', scale_images=False, stack_hourglass=False, pool_angles=False,
                   batchnorm=False):
    assert 'train' in mode
    assert batch_size <= 16
    assert model in ('unet', 'hourglass')
    height, width = 660, 512

    tf.reset_default_graph()

    training = tf.placeholder(tf.bool)
    images = tf.placeholder(tf.float32, [None, height, width])
    thmap = tf.placeholder(tf.float32, [None, height, width])
    resized_images = tf.image.resize_images(tf.expand_dims(images, -1), (width, width))
    resized_thmap = tf.image.resize_images(tf.expand_dims(thmap, -1), (width, width))

    if scale_images:
        size = tf.random_uniform([2], minval=int(0.75*width), maxval=width, dtype=tf.int32)
        h_pad, w_pad = (width-size[0])//2, (width-size[1])//2
        padding = [[0, 0], [h_pad, width-size[0]-h_pad], [w_pad, width-size[1]-w_pad]]
        resized_images = tf.image.resize_images(resized_images, size)
        resized_images = tf.expand_dims(tf.pad(tf.squeeze(resized_images), padding), -1)
        resized_thmap = tf.image.resize_images(resized_thmap, size)
        resized_thmap = tf.expand_dims(tf.pad(tf.squeeze(resized_thmap), padding), -1)
    if include_reflection:
        flipped_images = tf.concat([resized_images[0:1], resized_images[:0:-1]], axis=0)
        resized_images = tf.concat([resized_images, flipped_images[:, :, ::-1, :]], axis=-1)
    if rotate_images:
        angles = tf.random_uniform([batch_size], maxval=2*math.pi)
        resized_images = tf.contrib.image.rotate(resized_images, angles)
        resized_thmap = tf.contrib.image.rotate(resized_thmap, angles)

    imean, ivar = tf.nn.moments(resized_images, [0, 1, 2, 3])
    resized_images = (resized_images - imean) / tf.sqrt(ivar)

    if model == 'unet':
        logits = tf_models.unet_cnn(resized_images, width, 32, width, 64, conv3d=conv3d)
    else:
        feat, logits = tf_models.hourglass_cnn(resized_images, width, 4, width, 64,
                                               training=training, batchnorm=batchnorm)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=resized_thmap,
                                                                  logits=logits))

    if stack_hourglass:
        if pool_angles:
            feat = tf.concat([tf.concat([feat[-1:], feat[:-1]], axis=0),
                              feat,
                              tf.concat([feat[1:], feat[0:1]], axis=0)],
                              axis=-1)
        _, logits = tf_models.hourglass_cnn(feat, width//4, 4, width, 64, downsample=False,
                                            training=training, batchnorm=batchnorm)
        refined_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=resized_thmap,
                                                                              logits=logits))
    if refine2d or refine3d:
        if refine2d:
            logits = tf.concat([tf.concat([logits[-1:], logits[:-1]], axis=0),
                                logits,
                                tf.concat([logits[1:], logits[0:1]], axis=0)],
                               axis=-1)
        logits = tf_models.unet_cnn(logits, 4, width, 8, conv3d=refine3d)
        refined_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=resized_thmap,
                                                                              logits=logits))
    refined = stack_hourglass or refine2d or refine3d

    train_summary = tf.summary.scalar('train_loss', refined_loss if refined else loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss + refined_loss if refined else loss)
    pred_hmap = tf.sigmoid(logits)
    if scale_images:
        pred_hmap = pred_hmap[:, h_pad:-(width-size[0]-h_pad), w_pad:-(width-size[1]-w_pad), :]
    pred_hmap = tf.squeeze(tf.image.resize_images(pred_hmap, (height, width)))

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def feed(data, is_training=True):
        return {
            images: data[..., 0],
            thmap: np.sum(data[..., 1:], axis=-1),
            training: is_training
        }

    def batch_gen(x):
        for data in tqdm.tqdm(x):
            for i in range(0, 16, batch_size):
                yield np.rollaxis(data[..., i:i+batch_size, :], -2, 0)

    def predict(dset, n_sample=16):
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            for data in batch_gen(dset):
                pred = np.zeros((16, height, width))
                for _ in range(n_sample):
                    pred += sess.run(pred_hmap, feed_dict=feed(data, False))
                yield pred / n_sample

    if os.path.exists('done'):
        return predict

    valid_mode = mode.replace('train', 'valid')
    _, _, dset_train = dataio.get_data_and_threat_heatmaps(mode)
    _, _, dset_valid = dataio.get_data_and_threat_heatmaps(valid_mode)

    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def eval_model(sess):
        losses = []
        for data in batch_gen(dset_valid):
            cur_loss = sess.run(loss, feed_dict=feed(data))
            losses.append(cur_loss)
        return np.mean(losses)

    def train_model(sess):
        it = 0
        t0 = time.time()
        best_valid_loss = None
        while time.time() - t0 < duration * 3600:
            for data in batch_gen(dset_train):
                _, cur_train_summary = sess.run([train_step, train_summary], feed_dict=feed(data))
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