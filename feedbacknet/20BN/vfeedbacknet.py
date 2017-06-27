#!/usr/bin/env python2

import argparse
import csv
import os

import multiprocessing as mp
import numpy as np
import tensorflow as tf

import PIL
from PIL import Image

NUM_EPOCHS = 1
CONVERSION_BATCH_SIZE = 10 #32768

POOL_SIZE = 64
IMAGE_HEIGHT = 100
NUM_FRAMES_PER_VIDEO = 75

def vfeedback_model_basic():
    return None

def prepare_video(args):
    video_num, video_width, data_root = args
    pathgen = lambda x : os.path.join(data_root, str(video_num), x)
    frames = sorted( os.listdir(pathgen('')) )
    num_frames = len(frames)
    
    video = np.zeros((NUM_FRAMES_PER_VIDEO, IMAGE_HEIGHT, video_width, 3))
    
    for i in range(num_frames):
        image = Image.open(pathgen(frames[i]))
        image = image.resize((video_width, IMAGE_HEIGHT), PIL.Image.BICUBIC)
        video[i,:,:,:] = image

    return (video, num_frames)

def main(args):
    
    # read labels, training, and validation files
    with open(args.label_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        labels_num2str = [ item[0].lower() for item in reader ]
        labels_str2num =  { label : idx  for idx,label in zip(range(len(labels_num2str)), labels_num2str) }

    with open(args.train_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        train_labels = { int(item[0]) : labels_str2num[item[1].lower()] for item in reader }
        train_video_nums = np.asarray(train_labels.keys())

    with open(args.validation_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        validation_labels = { int(item[0]) : labels_str2num[item[1].lower()] for item in reader }

    # define model
    model = vfeedback_model_basic()

    # start up a pool of workers
    pool = mp.Pool(POOL_SIZE)

    # begin training
    for epoch in range(NUM_EPOCHS):
        video_nums = np.random.choice(train_video_nums, size=(CONVERSION_BATCH_SIZE,), replace=False)
        prepare_video_jobs = [ (video_num, int(args.video_width), args.data_root) for video_num in video_nums ]
        
        prepared_videos = pool.map(prepare_video, prepare_video_jobs)
        prepared_videos = filter(lambda x: x != None, prepared_videos)
        print(prepared_videos[0])
        print(prepared_videos[0][0].shape)

        # training the converted videos
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    train the feedbacknet for the 20BN data set
    ''')
    
    parser.add_argument('label_file', type=str, nargs=None,
                        help='the something-something labels file')

    parser.add_argument('train_file', type=str, nargs=None,
                        help='the something-something train file')

    parser.add_argument('validation_file', type=str, nargs=None,
                        help='the something-something validation file')

    parser.add_argument('data_root', type=str, nargs=None,
                        help='the root directory of the something-something data set')

    parser.add_argument('video_width', type=str, nargs=None,
                        help='the width to rescale all videos (recommended 176)')

    args = parser.parse_args()
    main(args)
