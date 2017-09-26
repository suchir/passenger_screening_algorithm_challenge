import tensorflow as tf


def hourglass_model(x, bottleneck, num_features, num_output):
    def block(x):
        y = tf.layers.conv2d(x, num_features//2, 1, 1, padding='same', activation=tf.nn.relu)
        y = tf.layers.conv2d(y, num_features//2, 3, 1, padding='same', activation=tf.nn.relu)
        y = tf.layers.conv2d(y, num_features, 1, 1, padding='same', activation=tf.nn.relu)
        return x + y

    x = tf.reshape(x, [-1, 256, 256, 1])
    x = tf.layers.conv2d(x, num_features, 7, 2, padding='same', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)

    blocks = []
    while not blocks or blocks[-1].shape[1] > bottleneck:
        x = block(x)
        blocks.append(block(x))
        x = tf.layers.max_pooling2d(x, 2, 2)
    
    x = block(block(block(x)))
    
    while blocks:
        size = int(blocks[-1].shape[1])
        x = tf.image.resize_images(x, (size, size))
        x += blocks[-1]
        x = block(x)
        blocks.pop()
    
    x = tf.layers.conv2d(x, num_output, 1, 1)
    return x
