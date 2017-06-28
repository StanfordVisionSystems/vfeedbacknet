#!/usr/bin/env python3

import argparse
import csv
import os

import h5py
import multiprocessing as mp
import numpy as np
import json

import PIL
from PIL import Image

NUM_EPOCHS = 1

POOL_SIZE = 32
VIDEO_HEIGHT = 100
NUM_FRAMES_PER_VIDEO = 75
BATCH_SIZE = 2048 # num videos per batch

def prepare_video(args):
    video_num, video_width, video_root = args

    pathgen = lambda x : os.path.join(video_root, str(video_num), x)
    frames = sorted( os.listdir(pathgen('')) )
    num_frames = len(frames)

    if(num_frames > NUM_FRAMES_PER_VIDEO):
        return None
            
    video = np.zeros((NUM_FRAMES_PER_VIDEO, VIDEO_HEIGHT, video_width, 3), dtype=np.float32)

    for i in range(num_frames):
        image = Image.open(pathgen(frames[i])) # in RGB order by default
        image = np.asarray(image.resize((video_width, VIDEO_HEIGHT), PIL.Image.BICUBIC), dtype=np.float32)
        video[i,:,:,:] = (image / 128) - 1 # squash to interval (-1, 1)
        
    return { 'raw_video' : video, 'num_frames' : num_frames, 'video_num' : video_num }

def main(args):
    
    # read labels, training, and validation files
    with open(args.label_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        labels_num2str = [ item[0].lower() for item in reader ]
        labels_str2num =  { label : idx  for idx,label in zip(range(len(labels_num2str)), labels_num2str) }

    with open(args.data_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data_labels = { int(item[0]) : labels_str2num[item[1].lower()] for item in reader }
        data_video_nums = np.asarray( list(data_labels.keys()) )

    # start up a pool of workers
    with mp.Pool(POOL_SIZE) as pool:
        count = 0
        for i in range(BATCH_SIZE):
            begin = i
            end = min(i + BATCH_SIZE, len(data_video_nums))

            prepare_video_jobs = [ (video_nums, args.video_width, args.video_root) for video_nums in data_video_nums[begin:end] ]        
            prepared_videos = pool.map(prepare_video, prepare_video_jobs)
            
            prepared_videos = list(filter(None, prepared_videos))
            num_videos = len(prepared_videos)
    
            # store videos on disk
            with h5py.File(os.path.join(args.output_hdf5, 'chunk_'+str(i)+'.hdf5'), 'w') as f:
                metadata = { 'num_videos' : num_videos, 
                             'video_width' : args.video_width, 
                             'video_height' : VIDEO_HEIGHT, 
                             'video_length' : NUM_FRAMES_PER_VIDEO,
                             'video_channels' : 3, 
                             'video_channels_order' : 'RGB', 
                            }
                
                metadata = json.dumps(metadata, sort_keys=True, indent=2)
                f.create_dataset("metadata", data=np.string_(metadata))
                
                f.create_dataset("label_strings", data=np.string_( json.dumps(labels_num2str, sort_keys=True, indent=2) ))

                raw_video = f.create_dataset("raw_videos", (num_videos, NUM_FRAMES_PER_VIDEO, VIDEO_HEIGHT, args.video_width, 3), dtype='float32')
                num_frames = f.create_dataset("num_frames", (num_videos,), dtype='int32')
                video_labels = f.create_dataset('video_labels', (num_videos,), dtype='int32')
                
                for i in range(num_videos):
                    raw_video[i,:,:,:,:] = prepared_videos[i]['raw_video']
                    num_frames[i] = prepared_videos[i]['num_frames']
                    video_labels[i] = data_labels[ prepared_videos[i]['video_num'] ] 
                
            count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    train the feedbacknet for the 20BN data set
    ''')
    
    parser.add_argument('label_file', type=str, nargs=None,
                        help='the something-something labels file')

    parser.add_argument('data_file', type=str, nargs=None,
                        help='the something-something data file to preprocess')

    parser.add_argument('video_root', type=str, nargs=None,
                        help='the root directory of the something-something data set')

    parser.add_argument('video_width', type=int, nargs=None,
                        help='the width to rescale all videos (recommended 176)')

    parser.add_argument('output_hdf5', type=str, nargs=None,
                        help='the dir to output the proprocessed data file')

    args = parser.parse_args()
    main(args)
