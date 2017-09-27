from common.caching import cached


from . import dataio
from . import synthetic_data
from . import tf_models


import tensorflow as tf
import numpy as np
import skimage.transform
import tqdm
import os
import h5py


@cached(synthetic_data.render_synthetic_body_zone_data, version=0)
def train_body_zone_segmenter(mode):
    image_size = 256
    output_size = image_size//4
    num_angles = 64
    num_splits = 16
    num_noise = 128


    get_noise = lambda size: skimage.transform.resize(np.random.randn(size, size, num_noise),
                                                      (image_size, image_size))
    noise = sum(get_noise(image_size//2**i) for i in range(9))
    noise = np.swapaxes(noise, 0, 2)

    def data_generator(x, angles, batch_size):
        for i in range(0, len(x), batch_size):
            images = x[i:i+batch_size, 0, ...]
            choice = np.random.choice(num_noise, len(images))
            images = np.clip(images + noise[choice] * 0.05, 0, 1)
            images = (images > 0.25).astype('float32')

            labels = np.repeat(x[i:i+batch_size, 1, ::4, ::4][..., np.newaxis], num_splits, axis=-1)

            yield images, labels, angles[i:i+batch_size]


    tf.reset_default_graph()
    images = tf.placeholder(tf.float32, [None, image_size, image_size])
    zones = tf.placeholder(tf.int32, [None, output_size, output_size, num_splits])
    mask = tf.placeholder(tf.float32, [None, output_size, output_size, num_splits])

    heatmaps = tf_models.hourglass_model(images, 4, 256, 18*num_splits)
    heatmaps = tf.reshape(heatmaps, [-1, output_size, output_size, num_splits, 18])
    all_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=zones, logits=heatmaps)
    loss = num_splits*tf.reduce_mean(mask*all_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_step = optimizer.minimize(loss)


    def train_model(sess, epochs):
        x, angles = synthetic_data.render_synthetic_body_zone_data(mode)
        batch_size = 32
        for epoch in tqdm.trange(epochs, desc='epochs'):
            for image_batch, zone_batch, angle_batch in tqdm.tqdm(data_generator(x, angles,
                                                                                 batch_size),
                                                                  desc='step',
                                                                  total=len(x)//batch_size):
                mask_batch = np.zeros((len(angle_batch), output_size, output_size, num_splits))
                for j, angle in enumerate(angle_batch):
                    mask_batch[j, ..., angle*num_splits//num_angles] = 1

                feed_dict = {
                    images: image_batch,
                    zones: zone_batch,
                    mask: mask_batch
                }
                sess.run([train_step], feed_dict=feed_dict)

    epochs = 1 if mode.startswith('sample') else 5
    saver = tf.train.Saver()
    path = os.getcwd() + '/model.ckpt'
    if not os.path.exists('done'):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_model(sess, epochs)
            saver.save(sess, path)
        open('done', 'w').close()

    def predict_generator(image_generator):
        with tf.Session() as sess:
            saver.restore(sess, path)
            for image_batch, angle_batch in image_generator:
                input_batch = (image_batch > 0.1).astype('float32')
                heatmap_batch = sess.run([heatmaps], feed_dict={images: input_batch})[0]
                get_pred = lambda i, angle: np.argmax(heatmap_batch[i, ..., angle, :], axis=-1)
                preds = np.stack([get_pred(i, angle*num_splits//num_angles)
                                  for i, angle in enumerate(angle_batch)])
                yield preds

    return predict_generator


@cached(dataio.get_data_hdf5, train_body_zone_segmenter, version=1)
def get_body_zone_heatmaps(mode):
    output_size = 64
    num_angles = 64

    if not os.path.exists('done'):
        _, x, _ = dataio.get_data_hdf5(mode)
        z = np.zeros((len(x), num_angles, output_size, output_size), dtype='uint8')

        def gen():
            for images in x:
                yield images, np.arange(num_angles)

        predict_generator = train_body_zone_segmenter('all')
        for i, preds in tqdm.tqdm(enumerate(predict_generator(gen())), total=len(x)):
            z[i] = preds

        np.save('z.npy', z)
        open('done', 'w').close()
    else:
        z = np.load('z.npy')
    return z