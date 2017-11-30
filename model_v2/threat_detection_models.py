from common.caching import cached, read_log_dir
from common.math import sigmoid, log_loss
from common.dataio import get_train_idx, get_valid_idx, get_train_labels

from . import tf_models
from . import body_zone_segmentation
from . import threat_segmentation_models

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



@cached(threat_segmentation_models.get_augmented_hourglass_predictions,
        body_zone_segmentation.get_body_zones, version=6)
def train_simple_segmentation_model(mode, duration, learning_rate=1e-3, num_filters=0,
                                    num_layers=0, blur_size=0, per_zone=None, use_hourglass=False):
    tf.reset_default_graph()

    zones_in = tf.placeholder(tf.float32, [16, 330, 256, 18])
    hmaps_in = tf.placeholder(tf.float32, [16, 330, 256, 2])
    labels_in = tf.placeholder(tf.float32, [17])
    confidence = tf.get_variable('confidence', [])

    if blur_size > 0:
        rx = tf.expand_dims(tf.pow(tf.range(blur_size, dtype=tf.float32)-(blur_size-1)/2, 2.0), -1)
        rmat = tf.tile(rx, [1, blur_size])
        rmat = rmat + tf.transpose(rmat)
        blur_amt = tf.get_variable('blur_amt', [])
        kernel = tf.exp(rmat * blur_amt)
        kernel /= tf.reduce_sum(kernel)
        kernel = tf.reshape(kernel, [blur_size, blur_size, 1, 1])
        zones = tf.concat([
                    tf.nn.conv2d(zones_in[..., i:i+1], kernel, [1]*4, padding='SAME')
                    for i in range(18)
                ], axis=-1)
    else:
        zones = zones_in

    zones = zones / tf.reduce_sum(zones, axis=-1, keep_dims=True)
    zones = tf.exp(tf.log(zones + 1e-6) * confidence)
    zones = zones / tf.reduce_sum(zones, axis=-1, keep_dims=True)

    hmaps = hmaps_in
    if use_hourglass:
        res = 256
        size = tf.random_uniform([2], minval=int(0.75*res), maxval=res, dtype=tf.int32)
        h_pad, w_pad = (res-size[0])//2, (res-size[1])//2
        padding = [[0, 0], [h_pad, res-size[0]-h_pad], [w_pad, res-size[1]-w_pad]]
        hmaps = tf.image.resize_images(hmaps, size)
        hmaps = tf.expand_dims(tf.pad(hmaps[..., 0], padding), axis=-1)

        hmaps, _ = tf_models.hourglass_cnn(hmaps, res, 32, res, num_filters, downsample=False)

        hmaps = hmaps[:, padding[1][0]:-padding[1][1]-1, padding[2][0]:-padding[2][0]-1, :]
        hmaps = tf.image.resize_images(hmaps, [330, 256])
    elif num_filters > 0:
        hmaps = tf.layers.conv2d(hmaps, num_filters, 1, activation=tf.nn.relu)

    zones = tf.reshape(tf.transpose(zones, [0, 3, 1, 2]), [16, 18, -1])
    hmaps = tf.reshape(hmaps, [16, -1, max(num_filters, 1)])
    prod = tf.transpose(tf.matmul(zones, hmaps), [1, 0, 2])[1:]

    if num_filters == 0:
        prod = tf.reduce_mean(prod, axis=(1, 2))
        bias = tf.get_variable('bias', [17], initializer=tf.constant_initializer(-2.24302))
        weights = tf.get_variable('weights', [17], initializer=tf.constant_initializer(0))
        logits = prod*weights + bias
    else:
        for _ in range(num_layers):
            prod = tf.concat([prod[:, 15:16, :], prod, prod[:, 0:1, :]], axis=1)
            prod = tf.layers.conv1d(prod, num_filters, 3, activation=tf.nn.relu)
        prod = tf.reduce_max(prod, axis=1)
        prod = tf.layers.dense(prod, 1, kernel_initializer=tf.zeros_initializer(),
                                        bias_initializer=tf.constant_initializer(-2.24302))
        logits = tf.squeeze(prod)

        if per_zone == 'bias':
            logits *= tf.get_variable('zone_weights', [17], initializer=tf.constant_initializer(1))
            logits += tf.get_variable('zone_bias', [17], initializer=tf.constant_initializer(0))
        elif per_zone == 'matmul':
            logits = tf.matmul(tf.get_variable('zone_mat', [17, 17], initializer=tf.constant_initializer(np.eye(17))),
                               tf.expand_dims(logits, -1))
            logits += tf.get_variable('zone_bias', [17], initializer=tf.constant_initializer(0))
            logits = tf.squeeze(logits)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_in, logits=logits))
    preds = tf.sigmoid(logits)

    train_summary = tf.summary.scalar('train_loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def predict(zones_all, hmaps, idx):
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            for i, hmap in zip(tqdm.tqdm(idx), hmaps):
                yield sess.run(preds, feed_dict={
                    zones_in: zones_all[i],
                    hmaps_in: hmap
                })

    if os.path.exists('done'):
        return predict

    valid_mode = mode.replace('train', 'valid')
    cvid = int(mode[-1])
    _, _, zones_all = body_zone_segmentation.get_body_zones('all')
    hmaps_train = threat_segmentation_models.get_augmented_hourglass_predictions(mode)
    hmaps_valid = threat_segmentation_models.get_augmented_hourglass_predictions(valid_mode)
    labels_all = [y for x, y in sorted(get_train_labels().items())]
    train_idx, valid_idx = get_train_idx('all', cvid), get_valid_idx('all', cvid)

    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def data_gen(zones_all, hmaps, labels_all, idx):
        for i, hmap in zip(tqdm.tqdm(idx), hmaps):
            yield {
                zones_in: zones_all[i],
                hmaps_in: hmap,
                labels_in: np.array(labels_all[i])
            }

    def eval_model(sess):
        losses = []
        for data in data_gen(zones_all, hmaps_valid, labels_all, valid_idx):
            cur_loss = sess.run(loss, feed_dict=data)
            losses.append(cur_loss)
        return np.mean(losses)

    def train_model(sess):
        it = 0
        t0 = time.time()
        best_valid_loss = None
        while time.time() - t0 < duration * 3600:
            for data in data_gen(zones_all, hmaps_train, labels_all, train_idx):
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

    return predict


@cached(train_simple_segmentation_model, version=0)
def write_simple_segmentation_model_errors(mode, *args, **kwargs):
    cvid = int(mode[-1])
    names, labels, zones_all = body_zone_segmentation.get_body_zones('all')
    hmaps = threat_segmentation_models.get_augmented_hourglass_predictions(mode)
    idx = get_train_idx('all', cvid) if mode.startswith('train') else get_valid_idx('all', cvid)
    predict = train_simple_segmentation_model(*args, **kwargs)

    errors = []
    total_loss = 0
    for i, pred in zip(idx, predict(zones_all, hmaps, idx)):
        name, label = names[i], labels[i]
        loss = log_loss(pred, label)
        for i in range(17):
            errors.append((loss[i], '%s_Zone%s' % (name, i+1), pred[i], label[i]))
        total_loss += np.mean(loss) / len(idx)
    errors.sort(reverse=True)

    with open('errors.txt', 'w') as f:
        lines = ['total loss: %s' % total_loss]
        lines += ['%.3f_%s_%.3f_%.3f' % i for i in errors]
        f.write('\n'.join(lines))