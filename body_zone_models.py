from caching import cached, read_input_dir

import numpy as np
import dataio
import hand_labeling
import skimage.io
import os
import glob
import random
import time

@cached(hand_labeling.get_body_part_labels, version=0)
def get_naive_body_part_labels(mode):
    _, side_labels, _, front_labels = hand_labeling.get_body_part_labels(mode)

    rows = np.mean(np.concatenate([side_labels[:, :8], front_labels[:, :8]]), axis=0).astype(int)
    cols = np.concatenate([np.mean(front_labels[:, 8:], axis=0),
                           np.mean(side_labels[:, 8:], axis=0)]).astype(int)

    return rows, cols


def _get_body_part_partition_labels():
    labels = [
        [(0, (2, 0), (2, 1)), (2, (2, 0), (2, 1)), (3, (2, 0), (2, 2))],  # zone 1
        [(0, (0, 0), (1, 1)), (2, (0, 0), (1, 1)), (3, (0, 0), (1, 2))],  # zone 2
        None,                                                             # zone 3
        None,                                                             # zone 4
        [(0, (3, 1), (3, 4)), (1, (3, 2), (3, 2)), (3, (3, 2), (3, 2))],  # zone 5
        [(0, (4, 1), (4, 2)), (2, (4, 1), (4, 2)), (3, (4, 0), (4, 2))],  # zone 6
        None,                                                             # zone 7
        [(0, (5, 0), (5, 1)), (2, (5, 0), (5, 1)), (3, (5, 0), (5, 2))],  # zone 8
        [(0, (5, 2), (6, 3)), (2, (5, 2), (6, 3))],                       # zone 9
        None,                                                             # zone 10
        [(0, (6, 0), (6, 1)), (2, (6, 0), (6, 1)), (3, (6, 0), (6, 2))],  # zone 11
        None,                                                             # zone 12
        [(0, (7, 0), (7, 2)), (2, (7, 0), (7, 2)), (3, (7, 0), (7, 2))],  # zone 13
        None,                                                             # zone 14
        [(0, (8, 0), (8, 2)), (2, (8, 0), (8, 2)), (3, (8, 0), (8, 2))],  # zone 15
        None,                                                             # zone 16
        [(1, (3, 0), (3, 0)), (2, (3, 1), (3, 4)), (3, (3, 0), (3, 0))],  # zone 17
    ]

    def mirror(label):
        ret = []
        for x in label:
            if x[0] in (0, 2):
                r1, c1 = x[1]
                r2, c2 = x[2]
                ret.append((x[0], (r1, 5-c2), (r2, 5-c1)))
            else:
                ret.append((4-x[0], x[1], x[2]))
        return ret

    for x, y in [(3, 1), (4, 2), (7, 6), (10, 8), (12, 11), (14, 13), (16, 15)]:
        labels[x-1] = mirror(labels[y-1])

    return labels


def _hcrop(image):
    image1d = np.max(image, axis=0)
    start = np.argmax(image1d > 0.1)
    end = np.argmax(image1d[::-1] > 0.1) + 1
    ret = image[:, start:-end]
    if ret.size == 0:
        return np.zeros((image.shape[0], 1))
    return ret


def _vcrop(image):
    return _hcrop(image.T).T


def _get_body_part_partitions(image, rows, cols):
    front_cols = [2*cols[0]-cols[2], 2*cols[0]-cols[1], cols[0], cols[1], cols[2]]
    side_cols = cols[-2:]

    pad = lambda x, y: np.append(np.insert(x, 0, 0), y)
    rows = pad(rows, 660)
    front_cols = pad(front_cols, 512)
    side_cols = pad(side_cols, 512)

    image = image.copy()
    image = [np.rot90(image[:, :, i])/np.max(image[:, :, i]) for i in range(0, 16, 4)]
    image[1] = np.fliplr(image[1])
    image[2] = np.fliplr(image[2])

    ret = []
    for label in _get_body_part_partition_labels():
        images = []
        for angle, p1, p2 in label:
            r1, c1 = p1
            r2, c2 = p2
            cols = front_cols if angle in (0, 2) else side_cols
            images.append(_hcrop(image[angle][rows[r1]:rows[r2+1], cols[c1]:cols[c2+1]]))
        ret.append(_vcrop(np.concatenate(images, axis=1)))

    return ret


@cached(dataio.get_train_data_generator, get_naive_body_part_labels, version=9)
def get_naive_partitioned_body_part_train_data(mode):
    if not os.path.exists('done'):
        for i in '01':
            if not os.path.exists(i):
                os.mkdir(i)
        labels = dataio.get_train_labels()
        rows, cols = get_naive_body_part_labels('all')

        for file, data in dataio.get_train_data_generator(mode, 'aps')():
            images = _get_body_part_partitions(data, rows, cols)
            for i, image in enumerate(images):
                label = labels[file][i]
                skimage.io.imsave('%s/%s-%s.png' % (label, file, i+1), image)

        open('done', 'w').close()

