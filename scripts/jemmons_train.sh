#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

./vfeedbacknet_train /dev/shm/20bn-datasets/jester-v1-labels.csv \
                     /dev/shm/20bn-datasets/jester-v1-validation.csv \
                     /dev/shm/20bn-datasets/jester-v1-train.csv \
                     /dev/shm/20bn-datasets/20bn-jester-v1
                     
