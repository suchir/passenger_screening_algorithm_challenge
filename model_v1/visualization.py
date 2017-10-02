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


@cached(dataio.get_data_hdf5, body_zone_models.get_body_zone_heatmaps, version=1)
def plot_body_zone_predictions(mode):
    _, x, _ = dataio.get_data_hdf5(mode)
    z = body_zone_models.get_body_zone_heatmaps(mode)
    for i, (images, heatmaps) in tqdm.tqdm(enumerate(zip(x, z))):
        angle = np.random.randint(len(images))
        colors = synthetic_data.BODY_ZONE_COLORS[heatmaps[angle]]
        colors = skimage.transform.resize(colors, images.shape[-2:])
        image = np.repeat(images[angle][..., np.newaxis], 3, axis=-1)
        viz = np.clip(image + 0.25*colors, 0, 1)
        skimage.io.imsave('%s.png' % i, viz)
