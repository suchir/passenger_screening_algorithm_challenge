import subprocess
import sys
import os


def download_bucket(path, bucket):
    if not os.path.exists(path):
        os.makedirs(path)
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', 'gs://%s' % bucket, path])


if __name__ == '__main__':
    subprocess.check_call(['source', '/home/cs231n/myVE35/bin/activate'])

    inputs = set(sys.argv[1:])
    assert inputs <= {'aps', 'a3daps', 'a3d', 'hand_labeling'}, "unrecognized input"

    for dtype in ('aps', 'a3daps', 'a3d'):
        if dtype in inputs:
            download_bucket('input/competition_data', 'kaggle-tsa-stage1/stage1/%s' % dtype)
    if 'hand_labeling' in inputs:
        download_bucket('input', 'psac_hand_labeling/hand_labeling')
    if inputs:
        download_bucket('input/competition_data', 'psac_competition_data/*')

    subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
