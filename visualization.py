from caching import read_input_dir, cached

import numpy as np
import preprocessing
import pyevtk
import os
import matplotlib.pyplot
import matplotlib.animation


@cached(preprocessing.get_data_generator, version=2)
def convert_a3d_to_vtk(mode):
    assert mode in ('train', 'sample')

    if os.path.exists('done'):
        return

    for file, data in preprocessing.get_data_generator(mode, 'a3d')():
        data /= np.max(data)
        data[data < 0.1] = 0
        x = np.arange(data.shape[0] + 1)
        y = np.arange(data.shape[1] + 1)
        z = np.arange(data.shape[2] + 1)
        pyevtk.hl.gridToVTK(file.split('.')[0], x, y, z, cellData={'data': data.copy()})

    open('done', 'a').close()



@cached(preprocessing.get_data_generator, version=0)
def convert_aps_to_gif(mode):
    assert mode in ('train', 'sample')

    def animate(i):
        im = ax.imshow(np.flipud(data[:,:,i].transpose()), cmap = 'viridis')
        return [im]

    for file, data in preprocessing.get_data_generator(mode, 'aps')():
        fig = matplotlib.pyplot.figure(figsize = (16,16))
        ax = fig.add_subplot(111)
        anim =  matplotlib.animation.FuncAnimation(fig, animate, frames=range(0, data.shape[2]),
                                                   interval=200, blit=True)
        anim.save(file.replace('.aps', '.gif'))


if __name__ == '__main__':
    convert_aps_to_gif('train')