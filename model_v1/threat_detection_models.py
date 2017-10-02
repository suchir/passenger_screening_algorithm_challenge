from common.caching import cached, read_log_dir
from common.math import sigmoid, log_loss

from . import body_zone_models
from . import tf_models
from . import dataio

import tensorflow as tf
import numpy as np
import os
import datetime
import time
import tqdm


def _heatmap(z):
    return np.stack([z == i for i in range(1, 18)], axis=-1).astype('float32')


def _train_data_generator(x, y, z, num_angles, random_angles, preproc, batch_size, chunk_size):
    x_batch, y_batch, z_batch = [], [], []
    for i in range(len(x)):
        num_slices = x.shape[1]//num_angles
        for j in range(num_slices):
            if random_angles:
                angles = sorted(np.random.choice(x.shape[1], num_angles, replace=False))
                x_batch.append(x[i, angles])
                z_batch.append(_heatmap(z[i, angles]))
            else:
                x_batch.append(x[i, j::num_slices])
                z_batch.append(_heatmap(z[i, j::num_slices]))
            y_batch.append(y[i])

        if len(x_batch) >= chunk_size or (len(x_batch) > 0 and i+1 == len(x)):
            x_batch, y_batch, z_batch = np.stack(x_batch), np.stack(y_batch), np.stack(z_batch)
            perm = np.random.permutation(len(x_batch))
            x_batch, y_batch, z_batch = x_batch[perm], y_batch[perm], z_batch[perm]
            x_batch, z_batch = preproc(x_batch, z_batch)

            for j in range(0, len(x_batch), batch_size):
                yield x_batch[j:j+batch_size], y_batch[j:j+batch_size], z_batch[j:j+batch_size]
            x_batch, y_batch, z_batch = [], [], []


def _test_data_generator(x, z, num_angles, random_angles, preproc):
    for i in range(len(x)):
        x_batch, z_batch = [], []
        num_slices = x.shape[1]//num_angles
        for j in range(num_slices):
            if random_angles:
                angles = sorted(np.random.choice(x.shape[1], num_angles, replace=False))
                x_batch.append(x[i, angles])
                z_batch.append(_heatmap(z[i, angles]))
            else:
                x_batch.append(x[i, j::num_slices])
                z_batch.append(_heatmap(z[i, j::num_slices]))

        x_batch, z_batch = np.stack(x_batch), np.stack(z_batch)
        x_batch, z_batch = preproc(x_batch, z_batch)
        yield x_batch, z_batch


def _train_basic_multiview_cnn(mode, model, train_preproc, test_preproc, num_angles, model_mode,
                               batch_size, learning_rate, duration, random_angles):
    assert mode in ('sample_train', 'train')
    image_size = 256
    output_size = 64
    chunk_size = 256

    tf.reset_default_graph()

    images = tf.placeholder(tf.float32, [None, num_angles, image_size, image_size])
    labels = tf.placeholder(tf.float32, [None, 17])
    zones = tf.placeholder(tf.float32, [None, num_angles, output_size, output_size, 17])

    logits = tf_models.simple_multiview_cnn(images, zones, model, model_mode)

    train_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                        logits=logits))
    train_summary = tf.summary.scalar('train_loss', train_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(train_loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def predict_batch(sess, x_batch, z_batch):
        preds = []
        for i in range(0, len(x_batch), batch_size):
            feed_dict = {
                images: x_batch[i:i+batch_size],
                zones: z_batch[i:i+batch_size]
            }
            cur_preds = sess.run([logits], feed_dict=feed_dict)[0]
            preds.append(cur_preds)
        preds = np.mean(np.concatenate(preds), axis=0)
        return preds

    def predict(x, z):
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            preds = []
            gen = _test_data_generator(x, z, num_angles, random_angles, test_preproc)
            for x_batch, z_batch in tqdm.tqdm(gen, total=len(x)):
                preds.append(predict_batch(sess, x_batch, z_batch))
            preds = np.stack(preds)
            return preds

    if os.path.exists('done'):
        return predict

    valid_mode = mode.replace('train', 'valid')
    _, x_train, y_train = dataio.get_data_hdf5(mode)
    z_train = body_zone_models.get_body_zone_heatmaps(mode)
    _, x_valid, y_valid = dataio.get_data_hdf5(valid_mode)
    z_valid = body_zone_models.get_body_zone_heatmaps(valid_mode)
    train_gen = lambda: _train_data_generator(x_train, y_train, z_train, num_angles, random_angles,
                                              train_preproc, batch_size, chunk_size)
    valid_gen = lambda: _test_data_generator(x_valid, z_valid, num_angles, random_angles, 
                                             test_preproc)
    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def eval_model(sess):
        losses = []
        for (x_batch, z_batch), y_batch in zip(valid_gen(), y_valid):
            preds = predict_batch(sess, x_batch, z_batch)
            loss = log_loss(sigmoid(preds), y_batch)
            losses.append(loss)
        return np.mean(losses)

    def train_model(sess, duration):
        it = 0
        t0 = time.time()
        best_valid_loss = None
        while time.time() - t0 < duration:
            num_batches = len(x_train)*x_train.shape[1]//(num_angles*batch_size)
            for x_batch, y_batch, z_batch in tqdm.tqdm(train_gen(), total=num_batches):
                feed_dict = {
                    images: x_batch,
                    labels: y_batch,
                    zones: z_batch
                }

                _, cur_train_summary = sess.run([train_step, train_summary], feed_dict=feed_dict)
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
        train_model(sess, duration)

    open('done', 'w').close()

    return predict


@cached(version=2)
def train_simple_cnn(mode, num_angles=4, num_features=64, model_mode='default', train_hours=10,
                     random_angles=False):
    def train_preproc(x_batch, z_batch):
        x_batch = x_batch + 0.025 * np.random.randn(*x_batch.shape)
        if np.random.randint(2):
            return x_batch[..., ::-1], z_batch[..., ::-1, :]
        return x_batch, z_batch

    def test_preproc(x_batch, z_batch):
        x_batch = x_batch + 0.025 * np.random.randn(*x_batch.shape)
        x_batch = np.concatenate([x_batch, x_batch[..., ::-1]])
        z_batch = np.concatenate([z_batch, z_batch[..., ::-1, :]])
        return x_batch, z_batch

    model = lambda x: tf_models.simple_cnn(x, num_features, [1, 3, 3], tf_models.leaky_relu)
    duration = 10 if mode.startswith('sample') else train_hours * 3600
    batch_size = 16*4*64 // (num_angles*num_features)
    return _train_basic_multiview_cnn(mode, model, train_preproc, test_preproc,
                                      num_angles=num_angles, model_mode=model_mode,
                                      batch_size=batch_size, learning_rate=1e-4, duration=duration,
                                      random_angles=random_angles)


@cached(version=0)
def get_simple_cnn_predictions(mode):
    if not os.path.exists('done'):
        _, x, y = dataio.get_data_hdf5(mode)
        z = body_zone_models.get_body_zone_heatmaps(mode)
        predict = train_simple_cnn('train')
        preds = predict(x, z)
        np.save('preds.npy', preds)
        open('done', 'w').close()
    else:
        preds = np.load('preds.npy')
    return preds


@cached(version=2)
def train_simple_meta_model(mode, train_minutes=1, reg_amt=1):
    assert mode in ('sample_train', 'train')

    valid_mode = mode.replace('train', 'valid')
    _, _, y_train = dataio.get_data_hdf5(mode)
    _, _, y_valid = dataio.get_data_hdf5(valid_mode)
    train_logits = get_simple_cnn_predictions(mode)
    valid_logits = get_simple_cnn_predictions(valid_mode)

    tf.reset_default_graph()

    logits_in = tf.placeholder(tf.float32, [None, 17])
    labels = tf.placeholder(tf.float32, [None, 17])
    W = tf.get_variable('W', [17, 17])
    logits_out = logits_in + tf.matmul(logits_in, W)
    reg = tf.reduce_mean(tf.abs(W)) * reg_amt
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits_out))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss + reg)
    train_summary = tf.summary.scalar('train_loss', loss)

    model_path = os.getcwd() + '/model.ckpt'
    saver = tf.train.Saver()

    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def train_model(sess, duration):
        it = 0
        t0 = time.time()
        best_valid_loss = None
        while time.time() - t0 < duration:
            for _ in tqdm.trange(100):
                feed_dict = {
                    logits_in: train_logits,
                    labels: y_train
                }
                _, cur_train_summary = sess.run([train_step, train_summary], feed_dict=feed_dict)
                writer.add_summary(cur_train_summary, it)
                it += 1

            feed_dict = {
                logits_in: valid_logits,
                labels: y_valid
            }
            cur_valid_loss = sess.run([loss], feed_dict=feed_dict)[0]
            cur_valid_summary = tf.Summary()
            cur_valid_summary.value.add(tag='valid_loss', simple_value=cur_valid_loss)
            writer.add_summary(cur_valid_summary, it)

            if best_valid_loss is None or cur_valid_loss < best_valid_loss:
                best_valid_loss = cur_valid_loss
                saver.save(sess, model_path)

    duration = 10 if mode.startswith('sample') else train_minutes * 60
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_model(sess, duration)


@cached(version=0)
def write_simple_cnn_predictions(mode):
    preds = sigmoid(get_simple_cnn_predictions(mode))
    ans_dict = {name: pred for name, pred in zip(names, preds)}
    dataio.write_answer_csv(ans_dict)