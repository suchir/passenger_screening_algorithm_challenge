## Requirements
- those in requirements.txt
	- run `pip install -r requirements.txt`
- tensorflow v1.4.0
- [elastix v4.8](http://elastix.isi.uu.nl/download/elastix_linux64_v4.8.tar.bz2)
- blender v2.79

## Setup
- build `input/scripts/spatial_pooling.cpp`
	- run `g++ spatial_pooling.cpp -o spatial_pooling --std=c++11` in the appropriate folder
- if on unix, remove the `.exe` from `model_v2/body_zone_segmentation.py:424`
- place stage1 a3d, a3daps, aps files in `input/competition_data/{a3d,a3daps,aps}`
  respectively
- place stage2 a3d, a3daps, aps files in `input/competition_data/stage2/{a3d,a3daps,aps}`
  respectively

## Reproducing on a single machine
- machine with an NVIDIA P100 or 1080ti, 16 cores, and 2TB SSD required
- run `python run.py private_test`
- output files are `cache/get_final_answer_csv/122369/'private_test'/ans1.txt`, `cache/get_final_answer_csv/122369/'private_test'/ans2.txt`
- estimated time to compute is about two weeks

## Reproducing on multiple machines
### Step 1
- change `CLOUD_CACHE_ENABLED` in `common/caching.py:12` to `True`
- create a Google Cloud storage bucket, and change `CACHE_BUCKET` in `common/caching.py:11` to the corresponding name
### Step 2
- in parallel, run the following
- create 20 VM instances with 16 cores, 60GB memory, and 1TB SSD
- on the first 10 VMs, do:
	- `python -c "from model_v2.passenger_clustering import get_augmented_segmentation_data_split as f; f('all', 10, <VM id from 0 to 9>)"`
- on the remaining 10 VMs, do:
	- `python -c "from model_v2.passenger_clustering import get_augmented_segmentation_data_split as f; f('private_test', 10, <VM id from 0 to 9>)"`
- create a VM with 16 cores, 60GB memory, 1TB SSD, and an NVIDIA P100
- run `python -c "from model_v2.body_zone_segmentation import get_body_zones as f; f('all'); f('private_test')"`
- wait for all the steps to complete and terminate VMs (should be within 24 hours)
### Step 3
- create 6 VMs with 16 cores, 60GB memory, 1TB SSD, and an NVIDIA P100
- for each of the VMs, do:
	- `python -c "from model_v2.threat_segmentation_models import get_multitask_cnn_predictions as f; f('all', 10, <vVM id from 0 to 5>); f('private_test', 10, <vVM id from 0 to 5>)"`
- wait for all the steps to complete and terminate VMs (should be within 24 hours)
### Step 4
- create a VM with 16 cores, 60GB memory, 2TB SSD, and an NVIDIA P100
- run `python run.py private_test`
- wait for all the steps to complete (should be within 24 hours)
- output files are `cache/get_final_answer_csv/122369/'private_test'/ans1.txt`, `cache/get_final_answer_csv/122369/'private_test'/ans2.txt`
