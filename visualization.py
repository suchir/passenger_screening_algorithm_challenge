from caching import read_input_dir, cached

import numpy as np
import preprocessing
import pyevtk
import tqdm
import os


@cached(preprocessing.get_a3d_data_generator, version=2)
def convert_a3d_to_vtk(mode):
    assert mode in ('train', 'sample')
    dir = f'{mode}/a3d'

    if os.path.exists('done'):
        return

    for file, data in preprocessing.get_a3d_data_generator(mode)():
        data /= np.max(data)
        data[data < 0.1] = 0
        x = np.arange(data.shape[0] + 1)
        y = np.arange(data.shape[1] + 1)
        z = np.arange(data.shape[2] + 1)
        pyevtk.hl.gridToVTK(file.split('.')[0], x, y, z, cellData={'data': data.copy()})

    open('done', 'a').close()


if __name__ == '__main__':
    convert_a3d_to_vtk('sample')