#!/usr/bin/env bash

GPU_ID=0
# WEIGHTS=\

#export PYTHONPATH=.

../../build/tools/caffe train \
    -solver solver_3loss_coco_fc7_voc72klabel.shared_glove72k.prototxt \
    -gpu $GPU_ID &> logfile_3loss_cocofc7_label72k_glove_input_prelm75k_sgd_lr4e5.log

# ../../build/tools/caffe train \
#     -solver solver_3loss_coco_fc7_voc72klabel.shared_glove72k.prototxt \
#     -weights $WEIGHTS \
#     -gpu $GPU_ID &> logfile_new8objs_3loss_cocofc7_label72k_glove_input_prelm75k_sgd_lr4e5.log
