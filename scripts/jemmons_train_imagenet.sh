#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -z $1 || -z $2 || -z $3 ]]; then
    echo "usage: ./jemmons_train.sh <GPU_NUM> <MODEL_NAME> <CHKT_PREFIX>"
    exit 0
fi

EXTRA_ARGS=${*:4}
if [[ -d $3 ]]; then
    if [[ "$4" != "FORCE" ]]; then
        echo 'dir already exists! please specify FORCE as 4th arg if you want to append to existing log'
        exit 0
    fi
    echo 'forcing continued execution'
    EXTRA_ARGS=${*:5}
fi

export CUDA_VISIBLE_DEVICES=$1
mkdir -p $3

IMAGENET_ROOT=/mnt/scratch/jemmons/imagenet2012.xs

python3 -u -B $DIR/vfeedbacknet_train $IMAGENET_ROOT/imagenet2012_labels.txt \
        $IMAGENET_ROOT/imagenet2012_validation.txt \
        $IMAGENET_ROOT/imagenet2012_train.txt \
        $IMAGENET_ROOT \
        $2 \
        $3 \
        $3/training_log.csv --imagenet --imagenet_zipped --video_length=1 --video_downsample_ratio=1 $EXTRA_ARGS 2>&1 | tee -a $3/training_log.log 
