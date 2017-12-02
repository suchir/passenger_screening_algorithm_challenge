from common.caching import cached, read_log_dir
from common.math import sigmoid, log_loss
from common.dataio import get_train_idx, get_valid_idx

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
import keras
import skimage.transform


@cached(passenger_clustering.join_augmented_aps_segmentation_data, cloud_cache=True, version=0)
def train_resnet50_fcn(mode, epochs, learning_rate=1e-3, num_layers=3, data_idx=0, downsize=2,
                       scale=1):
    layer_idxs = [4, 37, 79, 141, 173]
    height, width = 660//downsize, 512//downsize

    input_tensor = keras.layers.Input(shape=(height, width, 3))
    base_model = keras.applications.ResNet50(include_top=False, weights='imagenet',
                                             input_tensor=input_tensor,
                                             input_shape=(height, width, 3))

    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [height, width])

    hmaps = []
    for i in layer_idxs[-num_layers:]:
        output = base_model.layers[i].output
        hmap = keras.layers.Convolution2D(1, (1, 1))(output)
        hmap = keras.layers.Lambda(resize_bilinear)(hmap)
        hmaps.append(hmap)

    merged = keras.layers.Add()(hmaps)
    preds = keras.layers.Activation('sigmoid')(merged)
    model = keras.models.Model(inputs=input_tensor, outputs=preds)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='binary_crossentropy')

    def random_resize(images, amount=0.25):
        _, w, h, _ = images.shape
        pw, ph = np.random.randint(1, int(w*amount/2)), np.random.randint(1, int(h*amount/2))
        images = np.stack([skimage.transform.resize(image, [w-2*pw, h-2*ph])
                           for image in images / 10], axis=0) * 10
        images = np.pad(images, [(0, 0), (pw, pw), (ph, ph), (0, 0)], 'edge')
        return images

    def data_generator(dset):
        while True:
            for data in dset:
                if data_idx == 0:
                    images = data[:, ::downsize, ::downsize, 0]
                elif data_idx == 1:
                    images = data[:, ::downsize, ::downsize, 0] - data[:, ::downsize, ::downsize, 1]
                elif data_idx == 2:
                    images = data[:, ::downsize, ::downsize, 0] - data[:, ::downsize, ::downsize, 2]

                images = np.stack([images, images, images], axis=-1)
                labels = np.sum(data[:, ::downsize, ::downsize, -3:], axis=-1, keepdims=True) / 1000
                ret = random_resize(np.concatenate([images, labels], axis=-1))
                images, labels = ret[..., :3], ret[..., 3:]

                images *= 256
                if data_idx == 1 or data_idx == 2:
                    images += 128
                images = keras.applications.imagenet_utils.preprocess_input(images)

                yield images, labels

    valid_mode = mode.replace('train', 'valid')
    _, _, dset_train = passenger_clustering.join_augmented_aps_segmentation_data(mode, 6)
    _, _, dset_valid = passenger_clustering.join_augmented_aps_segmentation_data(valid_mode, 6)

    hist = model.fit_generator(data_generator(dset_train), steps_per_epoch=len(dset_train),
                               epochs=epochs, validation_data=data_generator(dset_valid),
                               validation_steps=len(dset_valid))

    plt.plot(hist.history['loss'])
    plt.savefig('loss.png')
    plt.plot(hist.history['val_loss'])
    plt.savefig('val_loss.png')

    with open('loss.txt', 'w') as f:
        f.write(str(min(hist.history['loss'])))
    with open('val_loss.txt', 'w') as f:
        f.write(str(min(hist.history['val_loss'])))


@cached(passenger_clustering.get_augmented_segmentation_data, dataio.get_augmented_threat_heatmaps,
        version=1)
def train_multitask_cnn(mode, cvid, duration, weights, sanity_check=False, normalize_data=True,
                        scale_data=1, num_filters=64):
    angles, height, width, res, filters = 16, 660, 512, 512, 14

    tf.reset_default_graph()

    data_in = tf.placeholder(tf.float32, [angles, height, width, filters])
    moments_in = tf.placeholder(tf.float32, [8, 2])
    means_in = tf.placeholder(tf.float32, [6])

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
    labels = data[..., 8:]
    if sanity_check:
        data = data[..., :4] * sanity_check
    elif normalize_data:
        data_list = []
        for i in range(8):
            data_list.append((data[..., i] - moments_in[i, 0]) / moments_in[i, 1])
        data = tf.stack(data_list, axis=-1)
    else:
        data = data[..., :8] * scale_data

    # get logits
    _, logits = tf_models.hourglass_cnn(data, res, 4, res, num_filters, num_output=6)

    # loss on segmentations
    losses, summaries = [], []
    for i in range(6):
        cur_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[..., i],
                                                                          logits=logits[..., i]))
        cur_summary = tf.summary.scalar('loss_%s' % i, cur_loss)
        default_loss = -(means_in[i]*tf.log(means_in[i]) + (1-means_in[i])*tf.log(1-means_in[i]))
        losses.append(cur_loss / default_loss * weights[i])
        summaries.append(cur_summary)
    loss = tf.add_n(losses)
    summaries.append(tf.summary.scalar('loss', loss))

    # actual predictions
    preds = tf.sigmoid(logits)
    preds = tf.cond(flip_lr > 0, lambda: preds[:, :, ::-1, :], lambda: preds)
    preds = preds[:, padding[1][0]:-padding[1][1]-1, padding[2][0]:-padding[2][0]-1, :]
    preds = tf.squeeze(tf.image.resize_images(preds, [height, width]))

    # optimization
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    dset_all, moments_all = passenger_clustering.get_augmented_segmentation_data(mode, 10)
    labels_all, means_all = dataio.get_augmented_threat_heatmaps(mode)
    train_idx, valid_idx = get_train_idx(mode, cvid), get_valid_idx(mode, cvid)

    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def data_gen(dset, moments, labels, means, idx):
        for i in tqdm.tqdm(idx):
            data = np.concatenate([dset[i], labels[i]], axis=-1)
            yield {
                data_in: data,
                moments_in: moments,
                means_in: means
            }

    def eval_model(sess):
        losses = []
        for data in data_gen(dset_all, moments_all, labels_all, means_all, valid_idx):
            cur_loss = sess.run(loss, feed_dict=data)
            losses.append(cur_loss)
        return np.mean(losses)

    def train_model(sess):
        it = 0
        t0 = time.time()
        best_valid_loss = None
        while time.time() - t0 < duration * 3600:
            for data in data_gen(dset_all, moments_all, labels_all, means_all, train_idx):
                cur_summaries = sess.run(summaries + [train_step], feed_dict=data)
                cur_summaries.pop()
                for summary in cur_summaries:
                    writer.add_summary(summary, it)
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



@cached(passenger_clustering.join_augmented_aps_segmentation_data, cloud_cache=True, version=4)
def train_augmented_hourglass_cnn(mode, duration, learning_rate=1e-3, random_scale=False,
                                  drop_loss=0, downsample=True, num_filters=64,
                                  loss_type='logloss', global_scale=1, scale_amount=0.25,
                                  batch_size=16, gaussian_noise=0, restore_model=False,
                                  random_shuffle=False):
    angles, height, width, res, filters = 16, 660, 512, 512, 7

    tf.reset_default_graph()

    data_in = tf.placeholder(tf.float32, [None, height, width, filters])
    data, labels = data_in[..., :-3], data_in[..., -3:]

    # random resize
    size = tf.random_uniform([2], minval=int((1-scale_amount)*res), maxval=res, dtype=tf.int32)
    h_pad, w_pad = (res-size[0])//2, (res-size[1])//2
    padding = [[0, 0], [h_pad, res-size[0]-h_pad], [w_pad, res-size[1]-w_pad]]
    data = tf.image.resize_images(data, size)
    data = tf.stack([tf.pad(data[..., i], padding) for i in range(4)], axis=-1)

    # random left-right flip
    flip_lr = tf.random_uniform([], maxval=2, dtype=tf.int32)
    data = tf.cond(flip_lr > 0, lambda: data[:, :, ::-1, :], lambda: data)

    # noise
    if gaussian_noise > 0:
        data = tf.concat([
            data[..., :-3] + tf.random_normal(tf.shape(data[..., :-3]), 0, gaussian_noise),
            data[..., -3:]
        ], axis=-1)

    # input normalization
    if random_scale:
        scale = 10 * tf.random_uniform([], minval=0.9, maxval=1.1)
    else:
        scale = 10 * global_scale
    label_fix = 1/1000  # screw-up
    data *= scale
    labels *= label_fix

    # hourglass network on first four filters
    _, logits = tf_models.hourglass_cnn(data, res, 4, res, num_filters,
                                        downsample=downsample)

    logits = tf.cond(flip_lr > 0, lambda: logits[:, :, ::-1, :], lambda: logits)
    logits = logits[:, padding[1][0]:-padding[1][1]-1, padding[2][0]:-padding[2][0]-1, :]
    logits = tf.image.resize_images(logits, [height, width])

    # loss on segmentations
    if drop_loss:
        pos_loss = []
        for i in range(3):
            cur_labels = tf.expand_dims(data[..., -i-1], -1)
            cur_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=cur_labels, logits=logits)
            angle_loss = tf.reduce_sum(tf.reshape(cur_loss*cur_labels, [angles, -1]), axis=-1)
            top_n, _ = tf.nn.top_k(angle_loss, k=drop_loss)
            pos_loss.append(tf.reduce_sum(angle_loss) - tf.reduce_sum(top_n))
        labels = tf.reduce_sum(data[..., -3:], axis=-1, keep_dims=True)
        neg_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                         logits=logits)*(1-labels))
        loss = (tf.add_n(pos_loss) + neg_loss) / tf.cast(tf.size(logits), tf.float32)
    else:
        if loss_type == 'density':
            hmaps = data[..., -3:]
            labels = tf.reduce_sum(hmaps / (tf.reduce_sum(hmaps, axis=(1, 2), keep_dims=True)+1e-3),
                                   axis=-1, keep_dims=True)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                          logits=logits))
        elif loss_type.startswith('normalized'):
            hmaps = data[..., -3:]
            all_hmaps = tf.reduce_sum(hmaps, axis=-1, keep_dims=True)
            loss_map = tf.nn.sigmoid_cross_entropy_with_logits(labels=all_hmaps, logits=logits)
            hmaps = tf.concat([hmaps, 1-all_hmaps], axis=-1)
            losses = []
            weights = [1, 1, 1, int(loss_type.split('-')[-1])]
            for i in range(4):
                losses.append(weights[i] * tf.reduce_sum((hmaps[..., i:i+1] * loss_map) / 
                              (tf.reduce_sum(hmaps[..., i:i+1]) + 1e-3)))
            loss = tf.add_n(losses)
        else:
            labels = tf.reduce_sum(labels, axis=-1, keep_dims=True)
            if loss_type == 'logloss':
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                              logits=logits))
            elif loss_type == 'l2':
                loss = tf.losses.mean_squared_error(labels, logits)
            elif loss_type == 'linear':
                preds = tf.sigmoid(logits)
                loss = tf.reduce_mean(-preds*(2*labels - 1))
            elif loss_type == 'dice':
                preds = tf.sigmoid(logits)
                loss = -(tf.reduce_sum(preds*labels) + 1)/\
                        (tf.reduce_sum(labels)+tf.reduce_sum(preds) + 1)

    # actual predictions
    if loss_type != 'l2':
        preds = tf.sigmoid(logits)
    else:
        preds = logits
    preds = tf.squeeze(preds)

    # optimization
    train_summary = tf.summary.scalar('train_loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def predict(dset, n_sample=16):
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            for data in tqdm.tqdm(dset):
                pred = np.zeros((angles, height, width))
                mean_loss = 0
                for _ in range(n_sample):
                    cur_loss, cur_pred = sess.run([loss, preds], feed_dict={
                        data_in: data
                    })
                    pred += cur_pred
                    mean_loss += cur_loss
                yield pred / n_sample, mean_loss / n_sample

    if os.path.exists('done'):
        return predict

    valid_mode = mode.replace('train', 'valid')
    _, _, dset_train = passenger_clustering.join_augmented_aps_segmentation_data(mode, 6)
    _, _, dset_valid = passenger_clustering.join_augmented_aps_segmentation_data(valid_mode, 6)

    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def eval_model(sess):
        losses = []
        for data in tqdm.tqdm(dset_valid):
            cur_loss = sess.run(loss, feed_dict={
                data_in: data
            })
            losses.append(cur_loss)
        return np.mean(losses)

    def train_model(sess):
        it = 0
        t0 = time.time()
        best_valid_loss = None
        while time.time() - t0 < duration * 3600:
            perm = [(i, j) for i in range(len(dset_train)) for j in range(0, 16, batch_size)]
            if random_shuffle:
                random.shuffle(perm)
            for choice in tqdm.tqdm(perm):
                data = dset_train[choice[0]][choice[1]:choice[1]+batch_size]
                _, cur_train_summary = sess.run([train_step, train_summary], feed_dict={
                    data_in: data
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
        if restore_model:
            saver.restore(sess, model_path)
        else:
            sess.run(tf.global_variables_initializer())
        train_model(sess)

    open('done', 'w').close()

    return predict


@cached(train_augmented_hourglass_cnn, subdir='ssd', cloud_cache=True, version=1)
def get_augmented_hourglass_predictions(mode):
    if not os.path.exists('done'):
        _, _, dset_in = passenger_clustering.join_augmented_aps_segmentation_data(mode, 6)
        f = h5py.File('data.hdf5', 'w')
        dset = f.create_dataset('dset', (len(dset_in), 16, 330, 256, 2))

        predict = train_augmented_hourglass_cnn('train-0', 8)
        for i, (pred, _) in enumerate(predict(dset_in)):
            dset[i, ..., 0] = pred[:, ::2, ::2]
        predict = train_augmented_hourglass_cnn('train-0', 12, loss_type='density')
        for i, (pred, _) in enumerate(predict(dset_in)):
            dset[i, ..., 1] = pred[:, ::2, ::2]


        f.close()
        open('done', 'w').close()

    f = h5py.File('data.hdf5', 'r')
    dset = f['dset']
    return dset


@cached(passenger_clustering.get_clustered_data_and_threat_heatmaps, version=0)
def train_hourglass_cnn(mode, duration, cluster_type='groundtruth', learning_rate=1e-3,
                        predict_one=False, lr_decay_tolerance=999, include_reflection=False,
                        random_shift=False, random_crop=False):
    assert 'train' in mode
    height, width = 660, 512
    res = 512

    tf.reset_default_graph()
    images_in = tf.placeholder(tf.float32, [None, height, width, 2])
    thmap_in = tf.placeholder(tf.float32, [None, height, width, 2])

    size = tf.random_uniform([2], minval=int(0.75*res), maxval=res-10, dtype=tf.int32)
    if random_shift:
        shift = tf.random_uniform([2], minval=-5, maxval=6, dtype=tf.int32)
    else:
        shift = [0, 0]
    h_pad, w_pad = (res-size[0])//2, (res-size[1])//2
    padding = [
        [[0, 0], [h_pad, res-size[0]-h_pad], [w_pad, res-size[1]-w_pad]],
        [[0, 0], [h_pad-shift[0], res-size[0]-h_pad+shift[0]],
         [w_pad-shift[1], res-size[1]-w_pad+shift[1]]],
    ]
    images = tf.image.resize_images(images_in, size)
    images = tf.stack([tf.pad(images[..., i], padding[i]) for i in range(2)], axis=-1)
    thmap = tf.image.resize_images(thmap_in, size)
    thmap = tf.stack([tf.pad(thmap[..., i], padding[i]) for i in range(2)], axis=-1)
    if random_crop:
        res = 256
        crop = tf.random_uniform([2], maxval=256, dtype=tf.int32)
        images = images[:, crop[0]:crop[0]+res, crop[1]:crop[1]+res, :]
        thmap = thmap[:, crop[0]:crop[0]+res, crop[1]:crop[1]+res, :]

    flip_lr = tf.random_uniform([], maxval=2, dtype=tf.int32)
    images = tf.cond(flip_lr > 0, lambda: images[:, :, ::-1, :], lambda: images)
    thmap = tf.cond(flip_lr > 0, lambda: thmap[:, :, ::-1, :], lambda: thmap)

    if include_reflection:
        flipped_images = tf.concat([images[0:1], images[:0:-1]], axis=0)
        images = tf.concat([images, images[:, :, ::-1, :]], axis=-1)
        flipped_thmap = tf.concat([thmap[0:1], thmap[:0:-1]], axis=0)
        thmap = tf.concat([thmap, thmap[:, :, ::-1, :]], axis=-1)

    imean, ivar = tf.nn.moments(images, [0, 1, 2, 3])
    images = (images - imean) / tf.sqrt(ivar)

    _, logits = tf_models.hourglass_cnn(images, res, 4, res, 64, num_output=thmap.shape[-1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=thmap, logits=logits)
    loss = tf.reduce_mean(loss[..., 0] if predict_one else loss)

    learning_rate_placeholder = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
    train_step = optimizer.minimize(loss)
    loss_summary = tf.summary.scalar('train_loss', loss)
    lr_summary = tf.summary.scalar('learning_rate', learning_rate_placeholder)

    pred_hmap = tf.sigmoid(logits)
    pred_hmap = tf.cond(flip_lr > 0, lambda: pred_hmap[:, :, ::-1, :], lambda: pred_hmap)
    pred_hmap = pred_hmap[:, h_pad:-(res-size[0]-h_pad), w_pad:-(res-size[1]-w_pad), :]
    pred_hmap = tf.squeeze(tf.image.resize_images(pred_hmap, (height, width)))
    if predict_one:
        pred_hmap = pred_hmap[..., 0]


    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def feed(data0, data1):
        data0, data1 = np.rollaxis(data0, 2, 0), np.rollaxis(data1, 2, 0)
        image0, image1 = data0[..., 0], data1[..., 0]
        thmap0, thmap1 = np.sum(data0[..., 1:], axis=-1), np.sum(data1[..., 1:], axis=-1)
        return {
            images_in: np.stack([image0, image1], axis=-1),
            thmap_in: np.stack([thmap0, thmap1], axis=-1),
            learning_rate_placeholder: learning_rate
        }

    def random_pair(ranges):
        group = np.random.randint(len(ranges))
        random_i = lambda: np.random.randint(ranges[group][0], ranges[group][1])
        return random_i(), random_i()

    def random_data(ranges, dset):
        i1, i2 = random_pair(ranges)
        return dset[i1], dset[i2]

    class Model:
        def __enter__(self):
            self.sess = tf.Session()
            saver.restore(self.sess, model_path)

        def predict(self, data0, data1):
            return self.sess.run(pred_hmap, feed_dict=feed(data0, data1))

        def __exit__(self, *args):
            self.sess.close()

    if os.path.exists('done'):
        return Model()

    valid_mode = mode.replace('train', 'valid')
    ranges_train, _, _, dset_train = passenger_clustering.get_clustered_data_and_threat_heatmaps(
                                        mode, cluster_type)
    ranges_valid, _, _, dset_valid = passenger_clustering.get_clustered_data_and_threat_heatmaps(
                                        valid_mode, cluster_type)

    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def eval_model(sess):
        losses = []
        for _ in tqdm.trange(len(dset_valid)):
            cur_loss = sess.run(loss, feed_dict=feed(*random_data(ranges_valid, dset_valid)))
            losses.append(cur_loss)
        return np.mean(losses)

    def train_model(sess):
        nonlocal learning_rate
        it, epoch = 0, 0
        t0 = time.time()
        best_valid_loss, best_valid_epoch = None, 0
        while time.time() - t0 < duration * 3600:
            for _ in tqdm.trange(len(dset_train)):
                _, cur_loss_summary, cur_lr_summary = \
                    sess.run([train_step, loss_summary, lr_summary],
                             feed_dict=feed(*random_data(ranges_train, dset_train)))
                writer.add_summary(cur_loss_summary, it)
                writer.add_summary(cur_lr_summary, it)
                it += 1

            valid_loss = eval_model(sess)
            cur_valid_summary = tf.Summary()
            cur_valid_summary.value.add(tag='valid_loss', simple_value=valid_loss)
            writer.add_summary(cur_valid_summary, it)

            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss, best_valid_epoch = valid_loss, epoch
                saver.save(sess, model_path)
            elif epoch - best_valid_epoch >= lr_decay_tolerance:
                learning_rate /= math.sqrt(10)
                best_valid_epoch = epoch
            epoch += 1


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_model(sess)

    open('done', 'w').close()

    return None


@cached(train_hourglass_cnn, passenger_clustering.get_clustered_data_and_threat_heatmaps, version=1)
def get_hourglass_cnn_predictions(mode, *args, **kwargs):
    if not os.path.exists('done'):
        model = train_hourglass_cnn(*args, **kwargs)
        ranges, _, _, dset = passenger_clustering.get_clustered_data_and_threat_heatmaps(mode,
                                kwargs.get('cluster_type', 'groundtruth'))

        f = h5py.File('data.hdf5', 'w')
        out = f.create_dataset('out', (len(dset), 16, 660, 512))
        with model:
            for group in tqdm.tqdm(ranges):
                for i in tqdm.trange(*group):
                    for j in tqdm.trange(*group):
                        out[i] += model.predict(dset[i], dset[j])
                    out[i] /= group[1] - group[0]
        open('done', 'w').close()
    else:
        f = h5py.File('data.hdf5', 'r')
        out = f['out']
    return out


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