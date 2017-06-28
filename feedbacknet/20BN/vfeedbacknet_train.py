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

NUM_EPOCHS = 1000

POOL_SIZE = 32
VIDEO_HEIGHT = 100
NUM_FRAMES_PER_VIDEO = 75
VIDEO_BATCH_SIZE = 2048 # num videos per batch

def vfeedback_model_basic(args):
    pass

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


def get_video_batch(pool, args):
    
    prepare_video_jobs = [ (video_nums, args.video_width, args.video_root) for video_nums in np.random.choice(args.data_video_nums, size=(VIDEO_BATCH_SIZE,), replace=False) ]
    prepared_videos = pool.map(prepare_video, prepare_video_jobs)
            
    prepared_videos = list(filter(None, prepared_videos))
    num_videos = len(prepared_videos)

    video_rawframes = np.zeros((num_videos, NUM_FRAMES_PER_VIDEO, VIDEO_HEIGHT, args.video_width, 3), dtype=np.float32)
    video_numframes = np.zeros((num_videos,), dtype=np.int32)
    video_labelnums = np.zeros((num_videos,), dtype=np.int32)    

    for i in range(num_videos):
        video_rawframes[i,:,:,:,:] = prepared_videos[i]['raw_video']
        video_numframes[i] = prepared_videos[i]['num_frames']
        video_labelnums[i] = args.data_labels[ prepared_videos[i]['video_num'] ]

    batch = {
        'num_videos' : num_videos,
        'video_rawframes' : video_rawframes,
        'video_numframes' : video_numframes,
        'video_labelnums' : video_labelnums,
    }

    return batch

def main(args):
    
    # read labels and training files
    with open(args.label_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        labels_num2str = [ item[0].lower() for item in reader ]
        labels_str2num =  { label : idx  for idx,label in zip(range(len(labels_num2str)), labels_num2str) }

    with open(args.data_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data_labels = { int(item[0]) : labels_str2num[item[1].lower()] for item in reader }; 
        data_video_nums = np.asarray( list(data_labels.keys()) )

        args.data_labels = data_labels
        args.data_video_nums = data_video_nums
    
    # build model
    model = vfeedback_model_basic(None)
    
    # get batch of data and train
    with mp.Pool(POOL_SIZE) as pool:
        for epoch in range(NUM_EPOCHS):
            batch = get_video_batch(pool, args)
            print(batch.keys())

            # train the model
            # ...

            break

    # save the trained model!
    # ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    train the feedbacknet for the 20BN data set
    ''')

    '''
    parser.add_argument('train_root', type=str, nargs=None,
                        help='directory with the preprocessed training files')
    '''

    parser.add_argument('label_file', type=str, nargs=None,
                        help='the something-something labels file')

    parser.add_argument('data_file', type=str, nargs=None,
                        help='the something-something data file to preprocess')

    parser.add_argument('video_root', type=str, nargs=None,
                        help='the root directory of the something-something data set')

    parser.add_argument('video_width', type=int, nargs=None,
                        help='the width to rescale all videos (recommended 176)')

    args = parser.parse_args()
    main(args)
