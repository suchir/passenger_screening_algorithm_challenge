from common.caching import read_input_dir, cached

import numpy as np
import os
import glob
import tqdm
import h5py
import pickle


def read_header(infile):
    """Read image header (first 512 bytes)
    """
    h = dict()
    fid = open(infile, 'r+b')
    h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
    h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
    h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
    h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
    return h


def read_data(infile):
    """Read any of the 4 types of image files, returns a numpy array of the image contents
    """
    extension = os.path.splitext(infile)[1]
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    if extension == '.aps' or extension == '.a3daps':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
    elif extension == '.a3d':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image
    elif extension == '.ahi':
        data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
        data = data.reshape(2, ny, nx, nt, order='F').copy()
        real = data[0,:,:,:].copy()
        imag = data[1,:,:,:].copy()
    fid.close()
    if extension != '.ahi':
        return data
    else:
        return real, imag


@cached(version=0)
def get_passenger_clusters():
    n_clusters = 24
    clusters = [None] * n_clusters
    for i in range(n_clusters):
        with read_input_dir('hand_labeling/passenger_id/%s' % i):
            clusters[i] = [x.split('.')[0] for x in glob.glob('*')]
    return clusters


@cached(version=0)
def get_cv_splits(n_split):
    if not os.path.exists('cv.pkl'):
        id_names = get_passenger_clusters()
        n_id = len(id_names)
        labels = get_train_labels()
        id_labels = np.array([np.sum([labels[x] for x in id_names[i]], axis=0)
                              for i in range(n_id)])
        all_labels = np.sum(id_labels, axis=0)

        np.random.seed(0)
        bdist, bsplit = 0, None
        for _ in range(10000):
            split = np.random.randint(n_split, size=n_id)
            freq = [sum(len(id_names[i]) for i in range(n_id) if split[i] == x) for x in range(n_split)]
            if min(freq)/len(labels) < 0.15 or max(freq)/len(labels) > 0.25:
                continue

            split_labels = np.array([np.sum(id_labels[split == x], axis=0) for x in range(n_split)])
            rem_labels = all_labels - split_labels
            dist = [np.dot(split_labels[i]/np.linalg.norm(split_labels[i]),
                           rem_labels[i]/np.linalg.norm(rem_labels[i])) for i in range(n_split)]
            if min(dist) > bdist:
                bdist = min(dist)
                bsplit = split

        cv = {}
        for i in range(n_id):
            for name in id_names[i]:
                cv[name] = bsplit[i]

        with open('cv.pkl', 'wb') as f:
            pickle.dump(cv, f)
    else:
        with open('cv.pkl', 'rb') as f:
            cv = pickle.load(f)
    return cv


@cached(version=1)
def get_data(mode, dtype):
    assert mode in ('sample', 'sample_large', 'all', 'sample_train', 'train', 'sample_valid',
                    'valid', 'sample_test', 'test', 'train-0', 'train-1', 'train-2', 'train-3',
                    'train-4', 'valid-0', 'valid-1', 'valid-2', 'valid-3', 'valid-4')
    assert dtype in ('aps', 'a3daps', 'a3d')
    with read_input_dir('competition_data/%s' % dtype):
        files = glob.glob('*')

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
        else:
            split = int(mode[-1])
            cv = get_cv_splits(5)
            in_valid = lambda file: cv[file.split('.')[0]] == split
            files = [file for file in files if mode.startswith('valid') == in_valid(file)]
    if mode.startswith('sample'):
        if mode.endswith('large'):
            files = files[:100]
        else:
            files = files[:10]

    with read_input_dir('competition_data/%s' % dtype):
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


@cached(version=1)
def get_train_labels():
    with read_input_dir('competition_data'):
        lines = open('revised_stage1_labels.csv').readlines()[1:]

    ret = {}
    for line in lines:
        file, label = line.split(',')
        file, zone = file.split('_')
        zone = int(zone.replace('Zone', ''))
        label = int(label)

        if file not in ret:
            ret[file] = [0] * 17
        ret[file][zone-1] = label

    return ret


@cached(get_data, get_train_labels, version=1, subdir='ssd')
def get_aps_data_hdf5(mode):
    if not os.path.exists('done'):
        names = []
        labels = []
        f = h5py.File('data.hdf5', 'w')
        gen = get_data(mode, 'aps')
        x = f.create_dataset('x', (len(gen), 660, 512, 16))
        for i, (name, label, data) in enumerate(gen):
            names.append(name)
            labels.append(label)
            x[i] = np.rot90(data)

        labels = np.stack(labels)
        np.save('labels.npy', labels)
        with open('names.txt', 'w') as f:
            f.write('\n'.join(names))
        open('done', 'w').close()
    else:
        f = h5py.File('data.hdf5', 'r')
        x = f['x']
        labels = np.load('labels.npy')
        with open('names.txt') as f:
            names = f.read().split('\n')
    return names, labels, x


def write_answer_csv(ans_dict):
    with open('ans.csv', 'w') as f:
        f.write('Id,Probability\n')
        for label, ret in ans_dict.items():
            for i in range(17):
                f.write('%s_Zone%s,%s\n' % (label, i+1, ret[i]))
