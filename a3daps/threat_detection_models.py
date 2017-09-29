from common.caching import cached, read_log_dir

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


def _train_data_generator(x, y, z, num_angles, preproc, batch_size, chunk_size):
    x_batch, y_batch, z_batch = [], [], []
    for i in range(len(x)):
        num_slices = x.shape[1]//num_angles
        for j in range(num_slices):
            x_batch.append(x[i, j::num_slices])
            y_batch.append(y[i])
            z_batch.append(_heatmap(z[i, j::num_slices]))

        if len(x_batch) >= chunk_size or (len(x_batch) > 0 and i+1 == len(x)):
            x_batch, y_batch, z_batch = np.stack(x_batch), np.stack(y_batch), np.stack(z_batch)
            perm = np.random.permutation(len(x_batch))
            x_batch, y_batch, z_batch = x_batch[perm], y_batch[perm], z_batch[perm]
            x_batch, z_batch = preproc(x_batch, z_batch)

            for j in range(0, len(x_batch), batch_size):
                yield x_batch[j:j+batch_size], y_batch[j:j+batch_size], z_batch[j:j+batch_size]
            x_batch, y_batch, z_batch = [], [], []


def _test_data_generator(x, z, num_angles, preproc):
    for i in range(len(x)):
        x_batch, z_batch = [], []
        num_slices = x.shape[1]//num_angles
        for j in range(num_slices):
            x_batch.append(x[i, j::num_slices])
            z_batch.append(_heatmap(z[i, j::num_slices]))

        x_batch, z_batch = np.stack(x_batch), np.stack(z_batch)
        x_batch, z_batch = preproc(x_batch, z_batch)
        yield x_batch, z_batch


def _train_basic_multiview_cnn(mode, model, train_preproc, test_preproc, num_angles, use_dense,
                               batch_size, learning_rate, duration):
    assert mode in ('sample_train', 'train')
    image_size = 256
    output_size = 64
    chunk_size = 256

    tf.reset_default_graph()

    images = tf.placeholder(tf.float32, [None, num_angles, image_size, image_size])
    labels = tf.placeholder(tf.float32, [None, 17])
    zones = tf.placeholder(tf.float32, [None, num_angles, output_size, output_size, 17])

    logits = tf_models.simple_multiview_cnn(images, zones, model, use_dense)

    train_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                        logits=logits))
    test_logits = tf.reduce_mean(logits, axis=0)
    test_preds = tf.sigmoid(test_logits)
    test_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[0], logits=test_logits)
    train_summary = tf.summary.scalar('train_loss', train_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(train_loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def predict(x, z):
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            preds = []
            gen = _test_data_generator(x, z, num_angles, test_preproc)
            for x_batch, z_batch in tqdm.tqdm(gen, total=len(x)):
                feed_dict = {
                    images: x_batch,
                    zones: z_batch
                }
                cur_preds = sess.run([test_preds], feed_dict=feed_dict)[0]
                preds.append(cur_preds)
            preds = np.stack(preds)
            return preds

    if os.path.exists('done'):
        return predict

    valid_mode = mode.replace('train', 'valid')
    _, x_train, y_train = dataio.get_data_hdf5(mode)
    z_train = body_zone_models.get_body_zone_heatmaps(mode)
    _, x_valid, y_valid = dataio.get_data_hdf5(valid_mode)
    z_valid = body_zone_models.get_body_zone_heatmaps(valid_mode)
    train_gen = lambda: _train_data_generator(x_train, y_train, z_train, num_angles, train_preproc,
                                              batch_size, chunk_size)
    valid_gen = lambda: _test_data_generator(x_valid, z_valid, num_angles, test_preproc)
    with read_log_dir():
        writer = tf.summary.FileWriter(os.getcwd())

    def eval_model(sess):
        losses = []
        for (x_batch, z_batch), y_batch in zip(valid_gen(), y_valid):
            feed_dict = {
                images: x_batch,
                labels: np.repeat(y_batch[np.newaxis], len(x_batch), axis=0),
                zones: z_batch
            }
            losses.append(sess.run([test_loss], feed_dict=feed_dict)[0])
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
def train_simple_cnn(mode, use_dense=False):
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

    model = lambda x: tf_models.simple_cnn(x, 64, [1, 3, 3], tf_models.leaky_relu)
    duration = 10 if mode.startswith('sample') else 8 * 3600
    return _train_basic_multiview_cnn(mode, model, train_preproc, test_preproc, num_angles=4,
                                      use_dense=use_dense, batch_size=24, learning_rate=1e-4,
                                      duration=duration)


@cached(version=0)
def write_simple_cnn_predictions(mode):
    assert mode in ('test', 'sample_test')

    names, x, _ = dataio.get_data_hdf5(mode)
    z = body_zone_models.get_body_zone_heatmaps(mode)

    train_mode = 'sample_train' if mode.startswith('sample') else 'train'
    predict = train_simple_cnn(train_mode)
    preds = predict(x, z)

    ans_dict = {name: pred for name, pred in zip(names, preds)}
    dataio.write_answer_csv(ans_dict)