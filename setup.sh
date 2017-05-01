#!/bin/bash

./install_sys_libs.sh
pip3 install -r requirements.txt
wget https://www.dropbox.com/s/9ljw0v3ycsqavjr/cudnn-8.0-linux-x64-v5.1.tgz
./cuda_install.sh
./download_celeba.sh
source create_vars.sh

