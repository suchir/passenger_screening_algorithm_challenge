from common.caching import cached
from common.dataio import get_aps_data_hdf5


import numpy as np
import skimage.io
import tqdm


@cached(get_aps_data_hdf5, version=1)
def write_aps_hand_labeling_images(mode):
    names, labels, x = get_aps_data_hdf5(mode)
    for name, label, data in tqdm.tqdm(zip(names, labels, x), total=len(x)):
        images = np.concatenate(np.rollaxis(data, 2), axis=1) / data.max()
        filename = '_'.join([name] + [str(i+1) for i in range(17) if label and label[i]])
        skimage.io.imsave('%s.png' % filename, np.repeat(images[..., np.newaxis], 3, axis=-1))


