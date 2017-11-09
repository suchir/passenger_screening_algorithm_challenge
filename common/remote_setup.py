import subprocess
import sys
import os


def download_bucket(path, bucket):
    if not os.path.exists(path):
        os.makedirs(path)
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', 'gs://%s' % bucket, path])


if __name__ == '__main__':
    inputs = set(sys.argv[1:])
    assert inputs <= {'aps', 'a3daps', 'a3d', 'hand_labeling', 'pyelastix'}, "unrecognized input"

    for dtype in ('aps', 'a3daps', 'a3d'):
        if dtype in inputs:
            download_bucket('input/competition_data', 'kaggle-tsa-stage1/stage1/%s' % dtype)
    if 'hand_labeling' in inputs:
        download_bucket('input', 'psac_hand_labeling/hand_labeling')
    if inputs:
        download_bucket('input/competition_data', 'psac_competition_data/*')
    if 'pyelastix' in inputs:
        os.mkdir('pyelastix')
        url = 'http://elastix.isi.uu.nl/download/elastix_linux64_v4.8.tar.bz2'
        subprocess.check_call(['wget', '-P', 'pyelastix/', url])
        subprocess.check_call(['tar', 'xvjf', 'pyelastix/elastix_linux64_v4.8.tar.bz2', '-C',
                               'pyelastix/'])
        subprocess.check_call(['sudo', 'cp', 'pyelastix/bin/*', '/usr/local/bin'])
        subprocess.check_call(['sudo', 'cp', 'pyelastix/lib/*', '/usr/local/lib'])
        subprocess.check_call(['rm', '-r', 'pyelastix'])

    subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
