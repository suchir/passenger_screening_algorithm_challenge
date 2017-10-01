from common.caching import read_input_dir, cached
from common.dataio import read_data, get_train_labels, write_answer_csv

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import glob
import os
import tqdm
import h5py
import pickle


def get_data(mode):
    assert mode in ('sample', 'sample_large', 'all', 'sample_train', 'train', 'sample_valid',
                    'valid', 'sample_test', 'test')
    with read_input_dir('competition_data/a3daps'):
        files = glob.glob('*.a3daps')

    labels = get_train_labels()
    has_label = lambda file: file.split('.')[0] in labels
    if mode.endswith('test'):
        files = [file for file in files if not has_label(file)]
    else:
        files = [file for file in files if has_label(file)]
        if mode.endswith('train'):
            files = files[100:]
        elif mode.endswith('valid'):
            files = files[:100]
    if mode.startswith('sample'):
        if mode.endswith('large'):
            files = files[:100]
        else:
            files = files[:10]

    with read_input_dir('competition_data/a3daps'):
        files = [os.path.abspath(file) for file in files]

    def generator():
        for file in tqdm.tqdm(files):
            file = file.replace('\\', '/')
            name = file.split('/')[-1].split('.')[0]
            yield name, labels.get(name), read_data(file)

    class DataGenerator(object):
        def __init__(self):
            self.gen = generator()
            self.len = len(files)

        def __len__(self):
            return self.len

        def __iter__(self):
            return self.gen

    return DataGenerator()


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