#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -z $1 || -z $2 ]]; then
    echo "usage: ./jemmons_test.sh <GPU_NUM> <CHKT_PREFIX>"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$1

TWENTYBN_ROOT=/mnt/scratch/jemmons/20bn-jester

python -u -B $DIR/vfeedbacknet_test $TWENTYBN_ROOT/jester-v1-labels.csv \
                                 $TWENTYBN_ROOT/jester-v1-validation.csv \
                                 $TWENTYBN_ROOT/20bn-jester-v1 \
                                 $2 --twentybn ${*:3}
