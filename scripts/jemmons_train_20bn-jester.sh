#!/bin/bash

if [[ -z $1 || -z $2 || -z $3 ]]; then
    echo "usage: ./jemmons_train.sh <GPU_NUM> <MODEL_NAME> <CHKT_PREFIX>"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$1
mkdir -p $3

TWENTYBN_ROOT=/mnt/scratch/jemmons/20bn-jester

python3 -u vfeedbacknet_train $TWENTYBN_ROOT/jester-v1-labels.csv \
        $TWENTYBN_ROOT/jester-v1-validation.csv \
        $TWENTYBN_ROOT/jester-v1-train.csv \
        $TWENTYBN_ROOT/20bn-jester-v1 \
        $2 \
        $3 \
        $3/training_log.csv --twentybn ${*:4} 2>&1 | tee $3/training_log.log 
