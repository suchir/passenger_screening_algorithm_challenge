import tensorflow as tf


def hourglass_model(x, bottleneck, num_features, num_output, downsample=True):
    image_size = 256

    def block(x):
        y = tf.layers.conv2d(x, num_features//2, 1, 1, padding='same', activation=tf.nn.relu)
        y = tf.layers.conv2d(y, num_features//2, 3, 1, padding='same', activation=tf.nn.relu)
        y = tf.layers.conv2d(y, num_features, 1, 1, padding='same', activation=tf.nn.relu)
        return x + y

    x = tf.reshape(x, [-1, image_size, image_size, 1])
    if downsample:
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


def simple_cnn(x, num_features, num_conv, activation):
    for i, count in enumerate(num_conv):
        for _ in range(count):
            x = tf.layers.conv2d(x, num_features, 3, padding='same', activation=activation)
        if i+1 != len(num_conv):
            x = tf.layers.max_pooling2d(x, 2, 2)
    return x


def simple_multiview_cnn(images, zones, model, model_mode):
    image_size, num_angles = int(images.shape[-1]), int(images.shape[1])

    heatmaps = []
    with tf.variable_scope('heatmaps') as scope:
        for i in range(num_angles):
            with tf.variable_scope('model'):
                image = tf.reshape(images[:, i], [-1, image_size, image_size, 1])
                heatmaps.append(model(image))
            scope.reuse_variables()

    output_size, num_features = int(heatmaps[0].shape[1]), int(heatmaps[0].shape[-1])
    zones_flat = tf.reshape(zones, [-1, num_angles, output_size*output_size, 17])
    heatmaps = tf.stack(heatmaps, axis=1)
    heatmaps_flat = tf.reshape(heatmaps, [-1, num_angles, output_size*output_size, num_features])
    zone_features = tf.matmul(tf.transpose(zones_flat, [0, 1, 3, 2]), heatmaps_flat)
    pooled_features = tf.reduce_max(zone_features, axis=1)

    if model_mode == 'hybrid':
        flat_features = tf.reshape(pooled_features, [-1, 17*num_features])
        W = tf.get_variable('W', [17, num_features])
        b = tf.get_variable('b', [17])
        logits = tf.layers.dense(flat_features, 17) + tf.reduce_sum(pooled_features*W, axis=-1) + b
    elif model_mode == 'dense':
        flat_features = tf.reshape(pooled_features, [-1, 17*num_features])
        logits = tf.layers.dense(flat_features, 17)
    else:
        W = tf.get_variable('W', [17, num_features])
        b = tf.get_variable('b', [17])
        logits = tf.reduce_sum(pooled_features * W, axis=-1) + b
    return logits


def leaky_relu(x, alpha=0.5):
    return tf.maximum(x, alpha*x)
