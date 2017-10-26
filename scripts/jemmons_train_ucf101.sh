#!/bin/bash

if [[ -z $1 || -z $2 || -z $3 ]]; then
    echo "usage: ./jemmons_train.sh <GPU_NUM> <MODEL_NAME> <CHKT_PREFIX>"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$1
mkdir -p $3

UCF101_ROOT=/mnt/scratch/jemmons/UCF-101-dumpjpg

python3 -u vfeedbacknet_train $UCF101_ROOT/classInd.txt \
        $UCF101_ROOT/testlist01.txt \
        $UCF101_ROOT/trainlist01.txt \
        $UCF101_ROOT/ \
        $2 \
        $3 \
        $3/training_log.csv --ucf101 ${*:4} 2>&1 | tee -a $3/training_log.log 
