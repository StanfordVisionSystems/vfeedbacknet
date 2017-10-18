#!/bin/bash

if [[ -z $1 || -z $2 || -z $3 ]]; then
    echo "usage: ./jemmons_train.sh <GPU_NUM> <MODEL_NAME> <CHKT_PREFIX>"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$1
mkdir -p $3

python3 -u vfeedbacknet_train /dev/shm/UCF-101-dumpjpg/classInd.txt \
        /dev/shm/UCF-101-dumpjpg/testlist01.txt \
        /dev/shm/UCF-101-dumpjpg/trainlist01.txt \
        /dev/shm/UCF-101-dumpjpg \
        $2 \
        $3 \
        $3/training_log.csv ${*:4} 2>&1 | tee $3/training_log.log 
