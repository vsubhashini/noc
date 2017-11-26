#!/bin/sh

echo "Downloading VGG model [~530MB] ..."

wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

echo "Downloading NOC Imagenet captioning model [~1.3GB] ..."
wget --no-check-certificate https://www.dropbox.com/s/.caffemodel.h5

echo "Organizing..."

DIR="./models"
if [ ! -d "$DIR" ]; then
    mkdir $DIR
fi

mv VGG_ILSVRC_16_layers.caffemodel $DIR"/"
mv imgnetcoco_3loss_voc72klabel_inglove_prelm75k_sgd_lr4e5_iter_80000.caffemodel.h5 $DIR"/"
echo "Done."

