#!/bin/bash
wget https://www.dropbox.com/s/vash6zkwm04yfvq/img_align_celeba.zip
unzip img_align_celeba.zip
mkdir Data_zoo
mkdir Data_zoo/CelebA_faces
mv img_align_celeba Data_zoo/CelebA_faces/
