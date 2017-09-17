from common.caching import cached, read_input_dir

from . import dataio
from . import hand_labeling

import numpy as np
import skimage.io
import skimage.transform
import os
import glob
import random
import time
import keras.preprocessing.image
import pickle


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


def _crop_image(img, tol=0.1):
    img = img / np.max(img)
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def _concat_images(images):
    ih = max(image.shape[0] for image in images)
    iw = max(image.shape[1] for image in images)

    best_nrows, best_stretch = -1, 1e9
    for nrows in range(1, len(images) + 1):
        h, w = ih * nrows, iw * (len(images)+nrows-1) // nrows
        stretch = max(h / w, w / h)
        if stretch < best_stretch:
            best_stretch, best_nrows = stretch, nrows

    best_ncols = (len(images)+best_nrows-1) // best_nrows
    ret = np.zeros((ih * best_nrows, iw * best_ncols))
    for i, image in enumerate(images):
        r, c = i // best_ncols, i % best_ncols
        h, w = images[i].shape
        ret[r*ih:r*ih+h, c*iw:c*iw+w] = images[i]

    return ret


def _get_grid_axes(rows, cols):
    front_cols = [2*cols[0]-cols[2], 2*cols[0]-cols[1], cols[0], cols[1], cols[2]]
    side_cols = cols[-2:]

    pad = lambda x, y: np.append(np.insert(x, 0, 0), y)
    rows = pad(rows, 660)
    front_cols = pad(front_cols, 512)
    side_cols = pad(side_cols, 512)

    return front_cols, side_cols, rows


def _get_body_part_partitions(image, rows, cols):
    front_cols, side_cols, rows = _get_grid_axes(rows, cols)

    image = image.copy()
    image = [np.rot90(image[:, :, i]) for i in range(0, 16, 4)]
    image[1] = np.fliplr(image[1])
    image[2] = np.fliplr(image[2])

    ret = []
    for label in _get_body_part_partition_labels():
        images = []
        for angle, p1, p2 in label:
            r1, c1 = p1
            r2, c2 = p2
            cols = front_cols if angle in (0, 2) else side_cols
            images.append(_crop_image(image[angle][rows[r1]:rows[r2+1], cols[c1]:cols[c2+1]]))
        ret.append(skimage.transform.resize(_concat_images(images), (256, 256)))
    return ret


@cached(dataio.get_train_data_generator, get_naive_body_part_labels, version=13)
def get_naive_partitioned_body_part_train_data(mode):
    if not os.path.exists('done'):
        labels = dataio.get_train_labels()
        rows, cols = get_naive_body_part_labels('all')
        x, y = [], []
        for file, data in dataio.get_train_data_generator(mode, 'aps')():
            images = _get_body_part_partitions(data, rows, cols)
            x += images
            y += labels[file]

        x, y = np.stack(x), np.array(y)
        np.save('x.npy', x)
        np.save('y.npy', y)

        open('done', 'w').close()
    else:
        x, y = np.load('x.npy'), np.load('y.npy')

    return x, y


@cached(get_naive_partitioned_body_part_train_data, version=0)
def get_naive_partitioned_symmetric_body_part_train_data(mode):
    # this function is broken
    if not os.path.exists('done'):
        x_in, y_in = get_naive_partitioned_body_part_train_data(mode)

        swap_idx = [2, 3, 0, 1, -4, 6, 5, 9, -8, 7, 11, 10, 13, 12, 15, 14, -16]
        def get_symmetric_image(i, j):
            j = swap_idx[j]
            ret = x_in[i+abs(j)]
            if j < 0:
                ret = np.fliplr(ret)
            return ret

        x, y = [], []
        for i in range(0, len(x_in), 17):
            for j in range(17):
                x.append(np.stack([x_in[i+j], get_symmetric_image(i, j)], axis=-1))
                y.append(np.array([y_in[i+j], y_in[i+abs(swap_idx[j])]]))

        x, y = np.stack(x), np.stack(y)
        np.save('x.npy', x)
        np.save('y.npy', y)

        open('done', 'w').close()
    else:
        x, y = np.load('x.npy'), np.load('y.npy')

    return x, y


def get_global_image_masks():
    rows, cols = get_naive_body_part_labels('all')
    front_cols, side_cols, rows = _get_grid_axes(rows, cols)
    masks = np.zeros((4, 660, 512, 17))

    labels = _get_body_part_partition_labels()
    for i, label in enumerate(labels):
        for angle, (r1, c1), (r2, c2) in label:
            cols = front_cols if angle in (0, 2) else side_cols
            masks[angle, rows[r1]:rows[r2+1], cols[c1]:cols[c2+1], i] = 1

    return masks


def _get_images_and_masks(data, masks, size, symmetric):
    images = []
    for i in range(4):
        image = np.rot90(data[:, :, 4*i])
        if i in (1, 2):
            image = np.fliplr(image)
        image = np.concatenate([image[:, :, np.newaxis], masks[i]], axis=2)
        images.append(skimage.transform.resize(image, (size, size)))
    images = np.stack(images)
    if symmetric:
        images = np.concatenate([np.stack([
            np.fliplr(images[0, :, :, 0:1]),
            images[3, :, :, 0:1],
            np.fliplr(images[2, :, :, 0:1]),
            images[1, :, :, 0:1]
        ]), images], axis=3)
    return images


@cached(dataio.get_train_data_generator, version=0)
def get_global_image_train_data(mode, size, symmetric):
    if not os.path.exists('done'):
        labels = dataio.get_train_labels()
        masks = get_global_image_masks()
        x, y = [], []
        for file, data in dataio.get_train_data_generator(mode, 'aps')():
            x.append(_get_images_and_masks(data, masks, size, symmetric))
            y.append(np.array(labels[file]))

        x, y = np.stack(x), np.stack(y)
        np.save('x.npy', x)
        np.save('y.npy', y)

        open('done', 'w').close()
    else:
        x, y = np.load('x.npy'), np.load('y.npy')

    return x, y


@cached(dataio.get_test_data_generator, version=0)
def get_global_image_test_data(mode, size):
    if not os.path.exists('done'):
        masks = get_global_image_masks()

        files, x = [], []
        for file, data in dataio.get_test_data_generator(mode, 'aps')():
            x.append(_get_images_and_masks(data, masks, size))
            files.append(file)

        x = np.stack(x)
        np.save('x.npy', x)
        with open('files.txt', 'w') as f:
            f.write('\n'.join(files))

        open('done', 'w').close()
    else:
        x = np.load('x.npy')
        with open('files.txt', 'r') as f:
            files = f.read().splitlines()

    return x, files


@cached(dataio.get_test_data_generator, get_naive_body_part_labels, version=0)
def get_naive_partitioned_body_part_test_data(mode):
    if not os.path.exists('ret.pickle'):
        ret = {}
        rows, cols = get_naive_body_part_labels('all')

        for file, data in dataio.get_test_data_generator(mode, 'aps')():
            images = _get_body_part_partitions(data, rows, cols)
            ret[file] = images

        with open('ret.pickle', 'wb') as f:
            pickle.dump(ret, f)
    else:
        with open('ret.pickle', 'rb') as f:
            ret = pickle.load(f)
    return ret