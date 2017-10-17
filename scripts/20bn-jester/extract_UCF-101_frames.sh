#!/usr/bin/env bash

#############################################
# modify the UCF-101 data directory:
UCF101DIR=/mnt/data/jemmons/ucf-101/UCF-101

# and, make sure ffmpeg is installed
FFMPEGBIN=ffmpeg
#############################################

for f in ${UCF101DIR}/*/*.avi; do
  dir=${f::-4}
  echo -----
  echo Extracting frames from ${f} into ${dir}...
  if [[ ! -d ${dir} ]]; then
    echo Creating directory=${dir}
    mkdir -p ${dir}
  fi

  ${FFMPEGBIN} \
    -i ${f} \
    ${dir}/image_%4d.jpg
done

wait
echo -------------------------------------------
echo Done!
