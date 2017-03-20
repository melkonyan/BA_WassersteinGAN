#!/bin/bash
sudo apt-get install python3-pip
sudo apt-get install libcupti-dev
pip3 install tensorflow-gpu
sudo apt-get install git
git clone https://github.com/melkonyan/BA_WassersteinGAN.git
export PATH="/usr/local/cuda-8.0/bin:/home/ubuntu/bin:/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"
sudo apt-get install python3-tk
sudo apt-get install unzip
