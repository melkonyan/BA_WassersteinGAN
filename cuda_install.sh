#!/bin/bash
tar -xvzf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/

sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
