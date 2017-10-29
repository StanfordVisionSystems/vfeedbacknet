#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -z $1 || -z $2 ]]; then
    echo "usage: ./jemmons_test.sh <GPU_NUM> <CHKT_PREFIX>"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$1

UCF101_ROOT=/mnt/scratch/jemmons/UCF-101-dumpjpg

python -u -B $DIR/vfeedbacknet_test $UCF101_ROOT/classInd.txt \
                                 $UCF101_ROOT/testlist01.txt \
                                 $UCF101_ROOT \
                                 $2 --ucf101 ${*:3}

