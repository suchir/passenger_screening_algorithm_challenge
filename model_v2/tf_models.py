import tensorflow as tf
import numpy as np


def unet_cnn(x, in_res, min_res, out_res, init_filters, conv3d=False):
    def block(x, n_filters):
        for _ in range(2):
            if conv3d:
                x = tf.concat([x[:, -1:, :, :, :], x, x[:, 0:1, :, :, :]], axis=1)
                x = tf.layers.conv3d(x, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
                x = x[:, 1:-1, :, :, :]
            else:
                x = tf.layers.conv2d(x, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
        return x

    if conv3d:
        x = tf.expand_dims(x, 0)

    blocks = []
    while in_res > min_res:
        x = block(x, 2*int(x.shape[-1]))
        blocks.append(x)
        if conv3d:
            x = tf.layers.max_pooling3d(x, [1, 2, 2], [1, 2, 2])
        else:
            x = tf.layers.max_pooling2d(x, 2, 2)
        in_res //= 2;

    x = block(x, 2*int(x.shape[-1]))

    while in_res < out_res:
        if conv3d:
            x = tf.layers.conv3d_transpose(x, x.shape[-1], [1, 2, 2], [1, 2, 2])
        else:
            x = tf.layers.conv2d_transpose(x, x.shape[-1], 2, 2)
        x = tf.concat([blocks.pop(), x], axis=-1)
        x = block(x, int(x.shape[-1])//2)
        in_res *= 2

    if conv3d:
        x = tf.layers.conv3d(x, 1, 1, 1)
        x = tf.squeeze(x, axis=0)
    else:
        x = tf.layers.conv2d(x, 1, 1, 1)
    return x


def hourglass_cnn(x, in_res, min_res, out_res, num_filters, num_output=1, downsample=True,
                  training=False, batchnorm=False, skip_connection=False):
    def block(x):
        y = tf.layers.conv2d(x, num_filters//2, 1, 1, padding='same', activation=tf.nn.relu)
        if batchnorm:
            y = tf.layers.batch_normalization(y, training=training)
        y = tf.layers.conv2d(y, num_filters//2, 3, 1, padding='same', activation=tf.nn.relu)
        if batchnorm:
            y = tf.layers.batch_normalization(y, training=training)
        y = tf.layers.conv2d(y, num_filters, 1, 1, padding='same', activation=tf.nn.relu)
        if batchnorm:
            y = tf.layers.batch_normalization(y, training=training)
        return x + y

    if downsample:
        x = tf.layers.conv2d(x, num_filters, 7, 2, padding='same', activation=tf.nn.relu)
        if batchnorm:
            x = tf.layers.batch_normalization(x, training=training)
        x = tf.layers.max_pooling2d(x, 2, 2)
        in_res //= 4
    else:
        x = tf.layers.conv2d(x, num_filters, 1, 1, padding='same', activation=tf.nn.relu)
        if batchnorm:
            x = tf.layers.batch_normalization(x, training=training)

    blocks = []
    while in_res > min_res:
        x = block(x)
        blocks.append(x if skip_connection else block(x))
        x = tf.layers.max_pooling2d(x, 2, 2)
        in_res //= 2

    x = block(block(block(x)))

    while blocks:
        x = tf.image.resize_images(x, (2*in_res, 2*in_res))
        x += blocks[-1]
        x = block(x)
        blocks.pop()
        in_res *= 2

    y = tf.layers.conv2d(x, num_output, 1, 1)
    y = tf.image.resize_images(y, [out_res, out_res])
    return x, y


def cnn_1d(x, num_filters, num_layers):
    for layer in range(num_layers):
        x = tf.layers.conv1d(x, num_filters, 3, padding='same', activation=tf.nn.relu)
        x = tf.layers.max_pooling1d(x, 2, 2)

    x = tf.reduce_max(x, axis=1, keep_dims=True)
    x = tf.layers.conv1d(x, 1, 1)

    x = tf.squeeze(x)
    return x


def random_uniform_noise(res, z, default):
    noise = tf.Variable(np.zeros((res, res)), dtype=tf.float32)
    cur_res = res
    while cur_res > 0:
        cur_noise = tf.random_uniform([cur_res, cur_res, 1], 0, 1, tf.float32)
        noise += tf.squeeze(tf.image.resize_images(cur_noise, (res, res)))
        cur_res //= 2
    mean, var = tf.nn.moments(noise, axes=(0, 1))
    noise = (noise - mean)/tf.sqrt(var)
    mask = tf.cast(noise > z, tf.float32)
    noise =  mask*tf.random_uniform([], 0, 1) + (1-mask)*default
    return noise