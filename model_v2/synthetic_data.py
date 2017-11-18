from common.caching import read_input_dir, cached, read_log_dir
from common.dataio import get_aps_data_hdf5, get_passenger_clusters, get_data

from . import dataio

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import glob
import os
import tqdm
import h5py
import pickle
import imageio
import math
import time


@cached(version=0)
def generate_random_models(n_models):
    with read_input_dir('makehuman/passengers'):
        ranges = defaultdict(lambda: [float('inf'), float('-inf')])
        for file in glob.glob('*.mhm'):
            with open(file, 'r') as f:
                modifiers = f.readlines()[4:-6]
                for modifier in modifiers:
                    _, m, x = modifier.split(' ')
                    x = float(x)
                    r = ranges[m]
                    r[0], r[1] = min(r[0], x), max(r[1], x)

    np.random.seed(0)
    for i in range(n_models):
        lines = ['version v1.1.1']
        for modifier in ranges:
            val = np.random.uniform(*ranges[modifier])
            lines.append('modifier %s %s' % (modifier, val))
        with open('%s.mhm' % i, 'w') as f:
            f.write('\n'.join(lines))