#!/bin/bash

if [[ -z $1 || -z $2 ]]; then
    echo "usage: ./jemmons_test.sh <GPU_NUM> <CHKT_PREFIX>"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$1

./vfeedbacknet_test /dev/shm/UCF-101-dumpjpg/classInd.txt \
                    /dev/shm/UCF-101-dumpjpg/testlist01.txt \
                    /dev/shm/UCF-101-dumpjpg \
                    $2

# ./vfeedbacknet_test /dev/shm/UCF-101-dumpjpg/classInd.txt \
#                     /dev/shm/UCF-101-dumpjpg/trainlist01.txt \
#                     /dev/shm/UCF-101-dumpjpg \
#                     $2
