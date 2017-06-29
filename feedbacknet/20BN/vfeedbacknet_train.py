#!/usr/bin/env python3

import argparse
import csv
import os

import h5py
import multiprocessing as mp
import numpy as np

import keras
import tensorflow as tf
from convLSTM import ConvLSTMCell # https://github.com/StanfordVisionSystems/tensorflow-convlstm-cell

import PIL
from PIL import Image

NUM_EPOCHS = 1024
TRAINING_BATCH_SIZE = 32

POOL_SIZE = 32
VIDEO_HEIGHT = 100
NUM_FRAMES_PER_VIDEO = 75
VIDEO_BATCH_SIZE = 2048 # num videos per batch

def vfeedback_model_basic(args, input_placeholder, output_placeholder, video_length, zeros):
    '''
    conv_b = new_bais()
    conv_w = new_conv2dweight(10, 10, 3, 32)

    input_frames = tf.unstack(input_placeholder, axis=1)
    conv_outputs = [ conv2d(input_frame, conv_w, conv_b) for input_frame in input_frames ]
    '''

    return None, None, None

def vfeedback_model_nofeedback(args, input_placeholder, output_placeholder, video_length, zeros):
    '''
    This model is just an ConvLSTM based RNN. (Let's get something working first before we add feedback).
    '''
    
    conv_b = new_bias()
    conv_w = new_conv2dweight(10, 10, 3, 1)

    input_frames = tf.unstack(input_placeholder, axis=1)
    conv_outputs = tf.stack([ conv2d(input_frame, conv_w, conv_b) for input_frame in input_frames ], axis=1)

    num_filters = 1 # convLSTM internal fitlers
    output, state = tf.nn.dynamic_rnn(
        ConvLSTMCell([VIDEO_HEIGHT, args.video_width], num_filters, [5, 5]),
        conv_outputs,
        dtype=tf.float32,
        sequence_length=video_length,
    )

    intermediate_output = tf.unstack(output, axis=1)
    intermediate_output_flat = [ tf.reshape(output, [-1, VIDEO_HEIGHT*args.video_width*num_filters]) for output in intermediate_output ]

    b_fc = new_bias()
    w_fc = tf.Variable( tf.truncated_normal([VIDEO_HEIGHT*args.video_width*num_filters, len(args.labels)], stddev=0.1) )

    fc_outputs = [ tf.matmul(output_flat, w_fc) + b_fc for output_flat in intermediate_output_flat ]

    cross_entropies = [ tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=fc_output) for fc_output in fc_outputs ]
    cross_entropies_truncated = tf.stack([ tf.where(video_length > i, cross_entropies[i], zeros) for i in range(NUM_FRAMES_PER_VIDEO) ], axis=1)
    #print(cross_entropies_truncated.shape)

    loss = tf.reduce_sum(cross_entropies, axis=0) / tf.to_float(video_length)
    #print(loss.shape)

    return loss, None, None
    
def conv2d(x, w, b):
    '''
    Given an input, weight matrix, and bias this function will create a 2D convolution
    '''

    output = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME') + b)
    return output

def new_bias():
    return tf.Variable( tf.truncated_normal([1], stddev=0.1) )

def new_conv2dweight(xdim, ydim, input_depth, output_depth):
    return tf.Variable( tf.truncated_normal([xdim, ydim, input_depth, output_depth], stddev=0.1) )

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

        args.labels = labels_num2str

    with open(args.data_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data_labels = { int(item[0]) : labels_str2num[item[1].lower()] for item in reader }; 
        data_video_nums = np.asarray( list(data_labels.keys()) )

        args.data_labels = data_labels
        args.data_video_nums = data_video_nums
    
    # build model
    x_input = tf.placeholder(tf.float32, [None, NUM_FRAMES_PER_VIDEO, VIDEO_HEIGHT, args.video_width, 3], name='input')
    x_length = tf.placeholder(tf.int32, [None,], name='input_length')
    y_label = tf.placeholder(tf.float32, [None, len(labels_num2str)], name='label')
    y_zeros = tf.placeholder(tf.float32, [None,],  name='zeros')
    loss, correct_prediction, accruacy = vfeedback_model_nofeedback(args, x_input, y_label, x_length, y_zeros)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # get batch of data and train
    with mp.Pool(POOL_SIZE) as pool:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(NUM_EPOCHS):
                batch = get_video_batch(pool, args)
                print('completed batch creation')

                # train the model
                for i in range(0, batch['num_videos']-TRAINING_BATCH_SIZE+1, TRAINING_BATCH_SIZE):
                    begin = i
                    end = i + TRAINING_BATCH_SIZE
                    print('starting a training batch')
                    
                    train_step.run(feed_dict={ 
                        x_input : batch['video_rawframes'][begin:end,:,:,:,:],
                        x_length : batch['video_numframes'][begin:end],
                        y_label : keras.utils.to_categorical(batch['video_labelnums'][begin:end], len(args.labels)),
                        y_zeros : np.zeros((TRAINING_BATCH_SIZE,))})

                    print('completed an iteration!'); #break
                
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
