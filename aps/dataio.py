from common.caching import read_input_dir, cached
from common.dataio import read_header, read_data, get_train_labels, write_answer_csv

import numpy as np
import os
import random
import glob
import tqdm
import pickle


@cached(version=1)
def get_train_headers(filetype):
    assert filetype in ('a3d', 'aps')
    if os.path.exists('headers.pickle'):
        with open('headers.pickle', 'rb') as f:
            return pickle.load(f)
    else:
        with read_input_dir('competition_data/%s' % filetype):
            headers = {file.split('.')[0]: read_header(file) for file in glob.glob('*')}
        with open('headers.pickle', 'wb') as f:
            pickle.dump(headers, f)
        return headers


def _get_data_generator(filetype, keep):
    assert filetype in ('a3d', 'aps')

    loc = 'competition_data/%s' % filetype

    with read_input_dir(loc):
        files = sorted(glob.glob('*'))
        random.seed(0)
        random.shuffle(files)
    files = [file for i, file in enumerate(files) if keep(i, file.split('.')[0])]

    def gen():
        for file in tqdm.tqdm(files):
            with read_input_dir(loc):
                data = read_data(file)
            yield file.split('.')[0], data

    return gen


@cached(version=5)
def get_all_data_generator(mode, filetype):
    assert mode in ('sample', 'all')

    labels = get_train_labels()
    keep = lambda i, x: x in labels and (mode != 'sample' or i < 10)
    return _get_data_generator(filetype, keep)


@cached(version=5)
def get_train_data_generator(mode, filetype):
    assert mode in ('train', 'valid', 'sample_train', 'sample_valid')

    labels = get_train_labels()
    if mode.startswith('sample'):
        keep = lambda i, x: x in labels and i < 10 and ((i < 8) == (mode == 'sample_train'))
    else:
        keep = lambda i, x: x in labels and ((i < 800) == (mode == 'train'))
    return _get_data_generator(filetype, keep)


@cached(version=5)
def get_test_data_generator(mode, filetype):
    assert mode in ('test', 'sample_test')

    labels = get_train_labels()
    keep = lambda i, x: x not in labels and (mode != 'sample_test' or i < 100)
    return _get_data_generator(filetype, keep)
