import tensorflow as tf


def unet_cnn(x, in_res, min_res, out_res, init_filters):
    def block(x, n_filters):
        x = tf.layers.conv2d(x, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
        return x

    blocks = []
    while x.shape[1] > min_res:
        x = block(x, 2*int(x.shape[-1]))
        blocks.append(x)
        x = tf.layers.max_pooling2d(x, 2, 2)

    x = block(x, 2*int(x.shape[-1]))

    while x.shape[1] < out_res:
        x = tf.layers.conv2d_transpose(x, x.shape[-1], 2, 2)
        x = tf.concat([blocks.pop(), x], axis=-1)
        x = block(x, int(x.shape[-1])//2)

    x = tf.layers.conv2d(x, 1, 1, 1)
    return x