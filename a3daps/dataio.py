from common.caching import read_input_dir, cached
from common.dataio import read_data, get_train_labels, write_answer_csv, get_data

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import glob
import os
import tqdm
import h5py
import pickle


@cached(version=3, subdir='ssd')
def get_data_hdf5(mode):
    num_angles = 64
    image_size = 256

    if not os.path.exists('done'):
        gen = get_data(mode)
        f = h5py.File('data.hdf5', 'w')
        x = f.create_dataset('x', (len(gen), num_angles, image_size, image_size))
        names = []
        y = np.zeros((len(gen), 17))

        for i, (name, label, data) in enumerate(gen):
            images = skimage.transform.resize(data, (image_size, image_size))
            x[i] = np.swapaxes(images, 0, 2)[:, ::-1, :]
            names.append(name)
            y[i] = label

        np.save('y.npy', y)
        with open('names.pickle', 'wb') as f:
            pickle.dump(names, f)
        open('done', 'w').close()
    else:
        f = h5py.File('data.hdf5', 'r')
        x = f['x']
        y = np.load('y.npy')
        with open('names.pickle', 'rb') as f:
            names = pickle.load(f)

    return names, x, y