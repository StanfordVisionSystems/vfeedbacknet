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

def vfeedback_model_nofeedback(args, input_placeholder, output_placeholder, video_length, zeros, ):
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
