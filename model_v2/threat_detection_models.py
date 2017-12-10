from common.caching import cached, read_log_dir
from common.math import sigmoid, log_loss
from common.dataio import get_train_idx, get_valid_idx, get_train_labels, write_answer_csv, \
                          get_cv_splits

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
import pickle


@cached(threat_segmentation_models.get_all_multitask_cnn_predictions,
        body_zone_segmentation.get_body_zones, version=12, cloud_cache=True)
def train_simple_segmentation_model(mode, cvid, duration, learning_rate=1e-3, num_filters=0,
                                    num_layers=0, blur_size=0, per_zone=None, use_hourglass=False,
                                    use_rotation=False, log_scale=False, num_conv=1,
                                    num_conv_filters=0, init_conf=1, zones_bias=False):
    tf.reset_default_graph()

    zones_in = tf.placeholder(tf.float32, [16, 330, 256, 18])
    hmaps_in = tf.placeholder(tf.float32, [16, 330, 256, 6])
    labels_in = tf.placeholder(tf.float32, [17])
    confidence = tf.get_variable('confidence', [], initializer=tf.constant_initializer(init_conf))

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

    sym_zone = [1, 2, 1, 2, 5, 6, 6, 8, 9, 8, 11, 11, 13, 13, 15, 15, 17]

    zones = zones / tf.reduce_sum(zones, axis=-1, keep_dims=True)
    zones = tf.log(zones + 1e-6)
    if zones_bias:
        with tf.variable_scope('zones_bias', reuse=tf.AUTO_REUSE):
            weights = tf.stack([
                tf.get_variable('zone_weights_%s' % zone, [], initializer=tf.constant_initializer(1)) 
                for zone in [0] + sym_zone
            ], axis=0)
            bias = tf.stack([
                tf.get_variable('zones_bias_%s' % zone, [], initializer=tf.constant_initializer(0))
                for zone in [0] + sym_zone
            ], axis=0)
        zones = zones*tf.square(weights) + bias
    else:
        zones *= tf.square(confidence)
    zones = tf.exp(zones)
    zones = zones / tf.reduce_sum(zones, axis=-1, keep_dims=True)

    if log_scale:
        hmaps = tf.log(hmaps_in)
    else:
        scales = np.array([2, 2000, 8, 600, 3, 2000])
        hmaps = tf.stack([hmaps_in[..., i] * scales[i] for i in range(6)], axis=-1)
    if use_hourglass:
        res = 256
        size = tf.random_uniform([2], minval=int(0.75*res), maxval=res, dtype=tf.int32)
        h_pad, w_pad = (res-size[0])//2, (res-size[1])//2
        padding = [[0, 0], [h_pad, res-size[0]-h_pad], [w_pad, res-size[1]-w_pad]]
        hmaps = tf.image.resize_images(hmaps, size)
        hmaps = tf.expand_dims(tf.pad(hmaps[..., 0], padding), axis=-1)

        if use_rotation:
            angle = tf.random_uniform([], maxval=2*math.pi)
            hmaps = tf.contrib.image.rotate(hmaps, angle)

        hmaps, _ = tf_models.hourglass_cnn(hmaps, res, 32, res, num_filters, downsample=False)

        if use_rotation:
            hmaps = tf.contrib.image.rotate(hmaps, -angle)

        hmaps = hmaps[:, padding[1][0]:-padding[1][1]-1, padding[2][0]:-padding[2][0]-1, :]
        hmaps = tf.image.resize_images(hmaps, [330, 256])
    elif num_filters > 0:
        for _ in range(num_conv):
            hmaps = tf.layers.conv2d(hmaps, num_conv_filters or num_filters, 1,
                                     activation=tf.nn.relu)

    zones = tf.reshape(tf.transpose(zones, [0, 3, 1, 2]), [16, 18, -1])
    hmaps = tf.reshape(hmaps, [16, -1, max(num_conv_filters or num_filters, 1)])
    prod = tf.transpose(tf.matmul(zones, hmaps), [1, 0, 2])[1:]

    flip_lr = tf.random_uniform([], maxval=2, dtype=tf.int32)
    prod = tf.cond(flip_lr > 0, lambda: prod[:, ::-1], lambda: prod)

    if num_filters == 0:
        prod = tf.reduce_mean(prod, axis=(1, 2))
        bias = tf.get_variable('bias', [17], initializer=tf.constant_initializer(-2.24302))
        weights = tf.get_variable('weights', [17], initializer=tf.constant_initializer(0))
        logits = prod*weights + bias
    else:
        def circular_conv(x, num_layers, num_filters, reduce_dim=True, reduce_max=True):
            for _ in range(num_layers):
                x = tf.concat([x[:, 15:16, :], x, x[:, 0:1, :]], axis=1)
                x = tf.layers.conv1d(x, num_filters, 3, activation=tf.nn.relu)
            if reduce_dim:
                x = tf.layers.conv1d(x, 1, 1)
                if reduce_max:
                    x = tf.reduce_max(x, axis=(1, 2))
            return x

        if per_zone == 'bias':
            logits = circular_conv(prod, num_layers, num_filters)

            with tf.variable_scope('zones', reuse=tf.AUTO_REUSE):
                weights = tf.stack([
                    tf.get_variable('weights_%s' % zone, [], initializer=tf.constant_initializer(1)) for zone in sym_zone
                ], axis=0)
                bias = tf.stack([
                    tf.get_variable('bias_%s' % zone, [], initializer=tf.constant_initializer(0)) for zone in sym_zone
                ], axis=0)
            logits = logits*weights + bias
        elif per_zone == 'matmul':
            logits = circular_conv(prod, num_layers, num_filters, reduce_max=False)
            logits = tf.reduce_max(logits, axis=1)
            logits = tf.matmul(tf.get_variable('zone_mat', [17, 17], initializer=tf.constant_initializer(np.eye(17))),
                               logits)
            logits += tf.get_variable('zone_bias', [17], initializer=tf.constant_initializer(0))
            logits = tf.squeeze(logits)
        elif per_zone == 'graph':
            logits = circular_conv(prod, num_layers, num_filters, reduce_max=False)
            def graph_refinement(a1, a2, num_layers, num_filters):
                x = tf.expand_dims(tf.concat([a1, a2], axis=-1), 0)
                with tf.variable_scope('graph'):
                    x = circular_conv(x, num_layers, num_filters)
                return tf.reduce_max(x)

            adj = [
                [2],
                [1],
                [4],
                [3],
                [6, 7],
                [5, 7, 17],
                [5, 6, 17],
                [6, 9, 11],
                [8, 10],
                [7, 9, 12],
                [8, 13],
                [10, 14],
                [11, 15],
                [12, 16],
                [13],
                [14],
                [6, 7]
            ]

            logits_list = []
            with tf.variable_scope('apply_graph') as scope:
                for i in range(17):
                    cur_logits = []
                    for j in adj[i]:
                        cur_logits.append(graph_refinement(logits[i], logits[j-1], 1,
                                                           1))
                        scope.reuse_variables()
                    logits_list.append(tf.reduce_min(tf.stack(cur_logits)))

            logits = tf.stack(logits_list)
        elif per_zone == 'dense':
            logits = circular_conv(prod, num_layers, num_filters, reduce_dim=False)
            logits = tf.reduce_max(logits, axis=1)

            with tf.variable_scope('zones', reuse=tf.AUTO_REUSE):
                weights = tf.stack([
                    tf.get_variable('weights_%s' % zone, [num_filters]) for zone in sym_zone
                ], axis=0)
                bias = tf.stack([
                    tf.get_variable('bias_%s' % zone, []) for zone in sym_zone
                ], axis=0)
            logits = tf.squeeze(tf.reduce_sum(logits*weights, axis=1) + bias)
        else:
            logits = circular_conv(prod, num_layers, num_filters)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_in, logits=logits))
    preds = tf.sigmoid(logits)

    train_summary = tf.summary.scalar('train_loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def predict(zones_all, hmaps_all, idx=None, n_sample=1):
        if idx is None:
            idx = range(len(zones_all))
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            for i in tqdm.tqdm(idx):
                ret = np.zeros(17)
                for _ in range(n_sample):
                    ret += sess.run(preds, feed_dict={
                        zones_in: zones_all[i],
                        hmaps_in: hmaps_all[i]
                    })
                yield ret / n_sample

    if os.path.exists('done'):
        return predict

    _, _, zones_all = body_zone_segmentation.get_body_zones(mode)
    hmaps_all = threat_segmentation_models.get_all_multitask_cnn_predictions(mode)
    labels_all = [y for x, y in sorted(get_train_labels().items())]
    train_idx, valid_idx = get_train_idx(mode, cvid), get_valid_idx(mode, cvid)

    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def data_gen(zones_all, hmaps_all, labels_all, idx):
        for i in tqdm.tqdm(idx):
            yield {
                zones_in: zones_all[i],
                hmaps_in: hmaps_all[i],
                labels_in: np.array(labels_all[i])
            }

    def eval_model(sess):
        losses = []
        for data in data_gen(zones_all, hmaps_all, labels_all, valid_idx):
            cur_loss = sess.run(loss, feed_dict=data)
            losses.append(cur_loss)
        return np.mean(losses)

    def train_model(sess):
        it = 0
        t0 = time.time()
        best_valid_loss = None
        while time.time() - t0 < duration * 3600:
            for data in data_gen(zones_all, hmaps_all, labels_all, train_idx):
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
def get_simple_segmentation_model_predictions(mode, *args, **kwargs):
    if not os.path.exists('done'):
        names, _, zones_all = body_zone_segmentation.get_body_zones(mode)
        hmaps_all = threat_segmentation_models.get_all_multitask_cnn_predictions(mode)
        predict = train_simple_segmentation_model(*args, **kwargs)

        preds = np.zeros((len(names), 17))
        for i, pred in enumerate(predict(zones_all, hmaps_all, n_sample=4)):
            preds[i] = pred

        np.save('preds.npy', preds)
        with open('pkl', 'wb') as f:
            pickle.dump(names, f)
        open('done', 'w').close()

    preds = np.load('preds.npy')
    with open('pkl', 'rb') as f:
        names = pickle.load(f)
    return names, preds


@cached(get_simple_segmentation_model_predictions, version=0)
def get_ensembled_model_predictions(mode):
    if not os.path.exists('done'):
        preds = []
        for i in range(5):
            names, pred = get_simple_segmentation_model_predictions(mode, 'all', i, 2,
                                                                    num_filters=16, num_layers=16,
                                                                    per_zone='bias')
            preds.append(pred)
        preds = np.mean(np.stack(preds, axis=-1), axis=-1)

        np.save('preds.npy', preds)
        with open('pkl', 'wb') as f:
            pickle.dump(names, f)
        open('done', 'w').close()

    preds = np.load('preds.npy')
    with open('pkl', 'rb') as f:
        names = pickle.load(f)
    return names, preds


@cached(get_simple_segmentation_model_predictions, version=0)
def get_out_of_fold_model_predictions(mode):
    if not os.path.exists('done'):
        preds_all = []
        for i in range(5):
            names, pred = get_simple_segmentation_model_predictions(mode, 'all', i, 2,
                                                                    num_filters=16, num_layers=16,
                                                                    per_zone='bias')
            preds_all.append(pred)

        preds = np.zeros((len(names), 17))
        cv = get_cv_splits(5)
        for i, name in enumerate(names):
            preds[i] =  preds_all[cv[name]][i]

        np.save('preds.npy', preds)
        open('done', 'w').close()

    preds = np.load('preds.npy')
    return preds


@cached(get_ensembled_model_predictions, version=0)
def train_zone_bias_model(mode, duration, learning_rate=1e-3):
    tf.reset_default_graph()

    preds_in = tf.placeholder(tf.float32, [None, 17])
    labels_in = tf.placeholder(tf.float32, [None, 17])

    sym_zone = [1, 2, 1, 2, 5, 6, 6, 8, 9, 8, 11, 11, 13, 13, 15, 15, 17]
    with tf.variable_scope('zones', reuse=tf.AUTO_REUSE):
        weights = tf.stack([
            tf.get_variable('weights_%s' % zone, [],
                            initializer=tf.constant_initializer(1))
            for zone in sym_zone
        ], axis=0)
        bias = tf.stack([
            tf.get_variable('bias_%s' % zone, [], initializer=tf.constant_initializer(0))
            for zone in sym_zone
        ], axis=0)

    logits = tf.log(preds_in) * weights + bias
    preds_out = tf.sigmoid(logits)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_in, logits=logits))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def predict(preds):
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            return sess.run(preds_out, feed_dict={preds_in: preds})

    if os.path.exists('done'):
        return predict

    preds_all = get_out_of_fold_model_predictions(mode)
    labels_all = [y for x, y in sorted(get_train_labels().items())]

    t0 = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while time.time() - t0 < duration * 3600:
            for _ in range(1000):
                _, cur_loss = sess.run([train_step, loss], feed_dict={
                    preds_in: preds_all,
                    labels_in: labels_all
                })
            
            print('loss = %s' % cur_loss)
            saver.save(sess, model_path)

    open('done', 'w').close()
    return predict


@cached(get_simple_segmentation_model_predictions, train_zone_bias_model, version=0)
def get_biased_ensembled_predictions(mode):
    if not os.path.exists('done'):
        preds = []
        for i in range(5):
            names, pred = get_simple_segmentation_model_predictions(mode, 'all', i, 2,
                                                                    num_filters=16, num_layers=16,
                                                                    per_zone='bias')
            bias = train_zone_bias_model('all', 0.05)
            preds.append(bias(pred))
        preds = np.mean(np.stack(preds, axis=-1), axis=-1)

        np.save('preds.npy', preds)
        with open('pkl', 'wb') as f:
            pickle.dump(names, f)
        open('done', 'w').close()

    preds = np.load('preds.npy')
    with open('pkl', 'rb') as f:
        names = pickle.load(f)
    return names, preds


def unconfident_preds(preds):
    n = 1147*17
    p_swap, p_fliplr, p_true, p_xor = 6/n, 4/n, 1871/n, 1442/n

    ret = []
    sym_zone = [3, 4, 1, 2, 5, 7, 6, 10, 9, 8, 12, 11, 14, 13, 16, 15, 17]
    for pred in preds:
        pred_new = []
        for i, p in enumerate(pred):
            p_s_t, p_s_f = p_swap/(2*p_true), p_swap/(2*(1-p_true))
            if sym_zone[i]-1 == i:
                p_new = p*(1-p_s_t) + (1-p)*p_s_f
            else:
                p_z = pred[sym_zone[i]-1]
                p_f_t, p_f_f = (1-p_z)*p_fliplr/p_xor, p_z*p_fliplr/p_xor
                p_new = p*(1-(p_s_t+p_f_t-p_s_t*p_f_t)) + (1-p)*(p_s_f+p_f_f-p_s_f*p_f_f)
            pred_new.append(p_new)
        ret.append(np.array(pred_new))
    ret = np.stack(ret)

    return ret


@cached(get_biased_ensembled_predictions, version=1)
def get_final_answer_csv(mode):
    names, conf_preds = get_biased_ensembled_predictions(mode)
    unconf_preds = unconfident_preds(conf_preds)
    conf_ans_dict = {k: v for k, v in zip(names, conf_preds)}
    unconf_ans_dict = {k: v for k, v in zip(names, unconf_preds)}
    write_answer_csv(conf_ans_dict, 'ans1.txt')
    write_answer_csv(unconf_ans_dict, 'ans2.txt')


@cached(get_biased_ensembled_predictions, version=0)
def get_train_logloss(unconfident=False):
    names, preds = get_biased_ensembled_predictions('all')
    labels_all = [y for x, y in sorted(get_train_labels().items())]

    if unconfident:
        preds = unconfident_preds(preds)

    losses = []
    total_loss = 0
    for name, pred, label in zip(names, preds, labels_all):
        loss = log_loss(pred, np.array(label))
        total_loss += np.mean(loss)/len(names)
        for i in range(17):
            losses.append((loss[i], '%s_Zone%s' % (name, i+1)))
    losses.sort()

    with open('loss.txt', 'w') as f:
        f.write('%s\n' % total_loss)
        for loss in losses:
            f.write('%s_%s\n' % (loss[1], loss[0]))


@cached(get_simple_segmentation_model_predictions, version=0)
def write_simple_segmentation_model_answer_csv(mode, clip, *args, **kwargs):
    names, preds = get_simple_segmentation_model_predictions(mode, *args, **kwargs)
    preds = np.clip(preds, clip, 1-clip)
    ans_dict = {k: v for k, v in zip(names, preds)}
    write_answer_csv(ans_dict)


@cached(train_simple_segmentation_model, version=0)
def write_simple_segmentation_model_errors(mode, *args, **kwargs):
    cvid = int(mode[-1])
    names, _, zones_all = body_zone_segmentation.get_body_zones('all')
    hmaps_all = threat_segmentation_models.get_all_multitask_cnn_predictions('all')
    idx = get_train_idx('all', cvid) if mode.startswith('train') else get_valid_idx('all', cvid)
    predict = train_simple_segmentation_model(*args, **kwargs)
    labels = get_train_labels()

    errors = []
    total_loss = 0
    for i, pred in zip(idx, predict(zones_all, hmaps_all, idx)):
        name = names[i]
        label = np.array(labels[name])
        loss = log_loss(pred, label)
        for i in range(17):
            errors.append((loss[i], '%s_Zone%s' % (name, i+1), pred[i], label[i]))
        total_loss += np.mean(loss) / len(idx)
    errors.sort(reverse=True)

    with open('errors.txt', 'w') as f:
        lines = ['total loss: %s' % total_loss]
        lines += ['%.3f_%s_%.3f_%.3f' % i for i in errors]
        f.write('\n'.join(lines))