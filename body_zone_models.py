from caching import cached, read_input_dir

import numpy as np
import dataio
import hand_labeling
import skimage.io
import skimage.transform
import os
import glob
import random
import time
import keras.preprocessing.image


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
    return img[np.ix_(mask.any(1),mask.any(0))]


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


def _get_body_part_partitions(image, rows, cols):
    front_cols = [2*cols[0]-cols[2], 2*cols[0]-cols[1], cols[0], cols[1], cols[2]]
    side_cols = cols[-2:]

    pad = lambda x, y: np.append(np.insert(x, 0, 0), y)
    rows = pad(rows, 660)
    front_cols = pad(front_cols, 512)
    side_cols = pad(side_cols, 512)

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


def get_data_generator(x, y, batch_size, proportion_true):
    x = x[:, :, :, np.newaxis]
    true_indexes = np.where(y == 1)[0]
    false_indexes = np.where(y == 0)[0]

    while True:
        num_true = int((random.random() * 2 * proportion_true) * batch_size)
        true_choice = np.random.choice(true_indexes, num_true)
        false_choice = np.random.choice(false_indexes, batch_size-num_true)

        yield np.concatenate([x[true_choice], x[false_choice]]), \
              np.concatenate([y[true_choice], y[false_choice]])

