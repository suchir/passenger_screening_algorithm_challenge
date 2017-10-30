from common.caching import cached, read_log_dir
from common.math import sigmoid, log_loss

from . import tf_models
from . import dataio

import tensorflow as tf
import numpy as np
import os
import datetime
import time
import tqdm


@cached(version=0)
def train_unet_cnn(mode, batch_size, learning_rate, duration):
    assert 'train' in mode
    assert batch_size <= 16
    height, width = 660, 512

    tf.reset_default_graph()

    images = tf.placeholder(tf.float32, [None, height, width])
    thmap = tf.placeholder(tf.float32, [None, height, width])
    resized_images = tf.image.resize_images(tf.expand_dims(images, -1), (width, width))
    resized_thmap = tf.image.resize_images(tf.expand_dims(thmap, -1), (width, width))

    logits = tf_models.unet_cnn(resized_images, width, 32, width, 64)
    pred_hmap = tf.squeeze(tf.image.resize_images(tf.sigmoid(logits), (height, width)))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=resized_thmap,
                                                                  logits=logits))
    train_summary = tf.summary.scalar('train_loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

    saver = tf.train.Saver()
    model_path = os.getcwd() + '/model.ckpt'

    def feed(data):
        image = data[..., 0]
        image -= np.mean(image)
        image /= np.std(image)
        return {
            images: image,
            thmap: np.sum(data[..., 1:], axis=-1)
        }

    def batch_gen(x):
        for data in tqdm.tqdm(x):
            for i in range(0, 16, batch_size):
                yield np.rollaxis(data[..., i:i+batch_size, :], -2, 0)

    def predict(dset):
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            preds = []
            for data in batch_gen(dset):
                preds.append(sess.run(pred_hmap, feed_dict=feed(data)))
                if len(preds)*batch_size == 16:
                    yield np.concatenate(preds)

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