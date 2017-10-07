#!/usr/bin/env python3

import keras
import numpy as np
import tensorflow as tf
from vfeedbacknet.convLSTM import ConvLSTMCell # https://github.com/StanfordVisionSystems/tensorflow-convlstm-cell

POOL_SIZE = 32

# def basic_model(args, input_placeholder, output_placeholder, video_length, zeros):
#     '''
#     conv_b = new_bais()
#     conv_w = new_conv2dweight(10, 10, 3, 32)

#     input_frames = tf.unstack(input_placeholder, axis=1)
#     conv_outputs = [ conv2d(input_frame, conv_w, conv_b) for input_frame in input_frames ]
#     '''
#     return None, None, None
    
def nofeedback_model(video_length, video_width, video_height, num_labels, input_placeholder, input_length, output_placeholder, zeros):
    '''
    This model is just an ConvLSTM based RNN. (Let's get something working first before we add feedback...).
    '''
    
    conv_b = new_bias()
    conv_w = new_conv2dweight(10, 10, 3, 1)

    input_frames = tf.unstack(input_placeholder, axis=1)
    conv_outputs = tf.stack([ conv2d(input_frame, conv_w, conv_b) for input_frame in input_frames ], axis=1)

    num_filters = 1 # convLSTM internal fitlers
    output, state = tf.nn.dynamic_rnn(
        ConvLSTMCell([video_height, video_width], num_filters, [5, 5]),
        conv_outputs,
        dtype=tf.float32,
        sequence_length=input_length,
    )

    intermediate_output = tf.unstack(output, axis=1)
    intermediate_output_flat = [ tf.reshape(output, [-1, video_height*video_width*num_filters]) for output in intermediate_output ]

    b_fc = new_bias()
    w_fc = tf.Variable( tf.truncated_normal([video_height*video_width*num_filters, num_labels], stddev=0.1) )

    fc_outputs = [ tf.matmul(output_flat, w_fc) + b_fc for output_flat in intermediate_output_flat ]

    softmaxes = [ tf.nn.softmax(logits=fc_output) for fc_output in fc_outputs ]
    #print(softmaxes[0].shape)
    #print(len(softmaxes))
    predictions = tf.stack(fc_outputs) #tf.stack(softmaxes)
    
    cross_entropies = [ tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=fc_output) for fc_output in fc_outputs ]
    #cross_entropies = [-tf.reduce_sum(output_placeholder*tf.log(fc_output + 1e-10)) for fc_output in fc_outputs]
    #cross_entropies = tf.stack([ -tf.reduce_sum(output_placeholder * tf.log(tf.clip_by_value(softmax, 1e-10, 1.0))) for softmax in softmaxes ])
    #print(cross_entropies.shape)
    
    cross_entropies_truncated = tf.stack([ tf.where(input_length > i, cross_entropies[i], zeros) for i in range(video_length) ], axis=1)
    loss = tf.reduce_sum(tf.reduce_sum(cross_entropies, axis=0) / tf.to_float(input_length))

    return loss, predictions, None
    
def conv2d(x, w, b):
    output = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME') + b)
    return output

def new_bias():
    return tf.Variable( tf.truncated_normal([1], stddev=0.1) )

def new_conv2dweight(xdim, ydim, input_depth, output_depth):
    return tf.Variable( tf.truncated_normal([xdim, ydim, input_depth, output_depth], stddev=0.1) )
