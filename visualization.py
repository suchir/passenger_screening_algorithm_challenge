from caching import read_input_dir, cached

import numpy as np
import matplotlib.pyplot as plt
import dataio
import hand_labeling
import pyevtk
import os
import matplotlib.animation


@cached(dataio.get_all_data_generator, version=2)
def convert_a3d_to_vtk(mode):
    if os.path.exists('done'):
        return

    for file, data in dataio.get_all_data_generator(mode, 'a3d')():
        data /= np.max(data)
        data[data < 0.1] = 0
        x = np.arange(data.shape[0] + 1)
        y = np.arange(data.shape[1] + 1)
        z = np.arange(data.shape[2] + 1)
        pyevtk.hl.gridToVTK(file.split('.')[0], x, y, z, cellData={'data': data.copy()})

    open('done', 'a').close()


@cached(dataio.get_all_data_generator, version=4)
def convert_aps_to_gif(mode):
    def animate(i):
        im = ax.imshow(np.flipud(data[:,:,i].transpose()), cmap = 'viridis')
        return [im]

    for file, data in dataio.get_all_data_generator(mode, 'aps')():
        fig = plt.figure(figsize = (16,16))
        ax = fig.add_subplot(111)
        anim =  matplotlib.animation.FuncAnimation(fig, animate, frames=range(0, data.shape[2]),
                                                   interval=200, blit=True)
        anim.save(file.replace('.aps', '.gif'))


@cached(dataio.get_train_headers, version=1)
def plot_a3d_density_distribution():
    headers = dataio.get_train_headers('a3d')

    plt.hist([x['avg_data_value'][0] for _, x in headers.items()], bins=100)
    plt.savefig('densities.png')


@cached(hand_labeling.get_body_part_labels, version=1)
def plot_zone_boundary_distributions(mode):
    _, _, _, front_labels = hand_labeling.get_body_part_labels('all')
    for i in range(front_labels.shape[1]):
        plt.hist(front_labels[:, i], bins=100, range=(0, 1))
    plt.savefig('boundaries.png')