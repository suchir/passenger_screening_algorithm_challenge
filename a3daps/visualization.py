from common.caching import cached

from . import dataio

import numpy as np
import skimage.io


@cached(version=0)
def plot_a3daps_images(mode):
    for file, _, images in dataio.get_data(mode):
        skimage.io.imsave('%s.png' % file, np.rot90(images[..., np.random.randint(64)]))