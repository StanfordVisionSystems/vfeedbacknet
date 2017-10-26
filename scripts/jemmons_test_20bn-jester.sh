#!/bin/bash

if [[ -z $1 || -z $2 ]]; then
    echo "usage: ./jemmons_test.sh <GPU_NUM> <CHKT_PREFIX>"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$1

TWENTYBN_ROOT=/mnt/scratch/jemmons/20bn-jester

./vfeedbacknet_test $TWENTYBN_ROOT/jester-v1-labels.csv \
                    $TWENTYBN_ROOT/jester-v1-validation.csv \
                    $TWENTYBN_ROOT/20bn-jester-v1 \
                    $2 --twentybn ${*:3}

# ./vfeedbacknet_test $TWENTYBN_ROOT/classInd.txt \
#                     $TWENTYBN_ROOT/trainlist01.txt \
#                     $TWENTYBN_ROOT \
#                     $2 ${*:3}
