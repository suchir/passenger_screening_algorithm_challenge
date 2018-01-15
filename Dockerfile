FROM tensorflow/tensorflow:1.4.0-gpu-py3

WORKDIR /app
ADD . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN apt-get update && apt-get install -y blender
RUN cp -r dependencies/blender/import_runtime_mhx2 /usr/share/blender/scripts/addons

RUN cp dependencies/elastix/elastix /usr/local/bin
RUN chmod +x /usr/local/bin/elastix
RUN cp dependencies/elastix/transformix /usr/local/bin
RUN chmod +x /usr/local/bin/transformix
RUN cp dependencies/elastix/libANNlib.so /usr/local/lib
RUN apt-get update && apt-get install -y ocl-icd-opencl-dev

RUN apt-get update && apt-get install -y g++
RUN g++ --std=c++11 input/scripts/spatial_pooling.cpp -o input/scripts/spatial_pooling
