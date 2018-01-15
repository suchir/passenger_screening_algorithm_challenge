## Instance setup
- create a VM with Ubuntu 16.04
- build the docker image
- start a shell session with `sudo nvidia-docker run -it <image name> /bin/bash` for GPU-enabled machines, or `sudo docker run -it <image name> /bin/bash` for machines without a GPU
- place stage1 a3d, a3daps, aps files in `input/competition_data/{a3d,a3daps,aps}`
  respectively
- place stage2 a3d, a3daps, aps files in `input/competition_data/stage2/{a3d,a3daps,aps}`
  respectively

## Inference / training + inference on a single machine
- create a VM with 16 cores, 60GB memory, 2TB SSD, and an NVIDIA P100
- for running training + inference, delete the `cache` directory
    - for running just inference, keep it as is
- run `python run.py private_test`
- estimated time to compute:
    - for inference, about four days
    - for training + inference, about two weeks
- output files are `cache/get_final_answer_csv/122369/'private_test'/ans1.txt`, `cache/get_final_answer_csv/122369/'private_test'/ans2.txt`

## Training + inference on multiple machines
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
	- `python -c "from model_v2.threat_segmentation_models import get_multitask_cnn_predictions as f; f('all', 10, <VM id from 0 to 5>); f('private_test', 10, <VM id from 0 to 5>)"`
- wait for all the steps to complete and terminate VMs (should be within 24 hours)
### Step 4
- create a VM with 16 cores, 60GB memory, 2TB SSD, and an NVIDIA P100
- run `python run.py private_test`
- wait for all the steps to complete (should be within 24 hours)
- output files are `cache/get_final_answer_csv/122369/'private_test'/ans1.txt`, `cache/get_final_answer_csv/122369/'private_test'/ans2.txt`
