from common.caching import cached

from . import dataio
from . import synthetic_data
from . import body_zone_models

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import tqdm


@cached(version=0)
def plot_a3daps_images(mode):
    for file, _, images in dataio.get_data(mode):
        skimage.io.imsave('%s.png' % file, np.rot90(images[..., np.random.randint(64)]))


@cached(dataio.get_data_hdf5, body_zone_models.train_body_zone_segmenter, version=0)
def plot_body_zone_predictions(mode):
    image_size = 256
    _, x, _ = dataio.get_data_hdf5(mode)
    batch_size = 32

    def image_generator():
        for i in range(0, len(x), batch_size):
            np.random.seed(i)
            angles = np.random.choice(x.shape[1], min(len(x), i+batch_size)-i)
            images = np.stack(x[i+j, angles[j]] for j in range(len(angles)))
            yield images, angles

    predict_generator = body_zone_models.train_body_zone_segmenter(mode)

    gen = zip(image_generator(), predict_generator(image_generator()))
    for i, ((images, angles), preds) in tqdm.tqdm(enumerate(gen), total=len(x)//batch_size):
        colors = synthetic_data.BODY_ZONE_COLORS[preds]
        colors = np.stack([skimage.transform.resize(image, (image_size, image_size))
                           for image in colors])
        images = np.repeat(images[..., np.newaxis], 3, axis=-1)
        ret = np.clip(images + 0.25*colors, 0, 1)

        for j, image in enumerate(ret):
            skimage.io.imsave('%s.png' % (i*batch_size+j), image)
