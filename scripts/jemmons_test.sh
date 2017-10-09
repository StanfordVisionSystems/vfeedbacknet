#!/bin/bash

if [[ -z $1 || -z $2 ]]; then
    echo "usage: ./jemmons_test.sh <GPU_NUM> <CHKT_PREFIX>"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$1

./vfeedbacknet_test /dev/shm/20bn-datasets/jester-v1-labels.csv \
                    /dev/shm/20bn-datasets/jester-v1-validation.csv \
                    /dev/shm/20bn-datasets/20bn-jester-v1 \
                    $2

# python3 -u vfeedbacknet_test /dev/shm/20bn-datasets/jester-v1-labels.csv \
#         /dev/shm/20bn-datasets/jester-v1-train.csv \
#         /dev/shm/20bn-datasets/20bn-jester-v1 \
#         $2
