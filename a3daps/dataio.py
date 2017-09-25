from common.caching import read_input_dir, cached
from common.dataio import read_data, get_train_labels

import glob
import os
import tqdm


def get_data(mode):
    assert mode in ('sample', 'all', 'sample_train', 'train', 'sample_valid', 'valid')
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
        files = files[:10]

    with read_input_dir('competition_data/a3daps'):
        files = [os.path.abspath(file) for file in files]
    for file in tqdm.tqdm(files):
        file = file.replace('\\', '/')
        name = file.split('/')[-1].split('.')[0]
        yield name, labels.get(name), read_data(file)