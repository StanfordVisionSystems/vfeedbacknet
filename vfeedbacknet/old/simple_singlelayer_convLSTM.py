#!/usr/bin/env python3

import logging

import keras
import numpy as np
import tensorflow as tf
from vfeedbacknet.convLSTM import ConvLSTMCell # https://github.com/StanfordVisionSystems/tensorflow-convlstm-cell
    
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)

LEAKINESS = 0.0

class logger:
    count = {}

    @staticmethod
    def log(var_name, var):
        if var_name in logger.count.keys():
            logger.count[var_name] += 1
            logger._log(var_name, var)
        else:
            logger.count[var_name] = 0
            logger._log(var_name, var)

    @staticmethod        
    def _log(var_name, var):
        maxwidth = 15
        padding = 4
        
        n = var_name[0:maxwidth]
        c = str(logger.count[var_name])
        p = ' ' * (maxwidth + padding - len(n) - len(c))
        logging.debug('{}-{}:{}{}x{}'.format(n, c, p, len(var), var[0].shape))
            
def simple_singlelayer_convLSTM(video_length, video_width, video_height, num_labels, input_placeholder, input_length, output_placeholder, zeros):
    '''
    This model is just an ConvLSTM based RNN. (Let's get something working first before we add feedback...).
    '''

    input_placeholder = tf.expand_dims(input_placeholder, axis=4)
    input_frames = tf.unstack(input_placeholder, axis=1)
    logging.debug('input: {}x{}'.format(len(input_frames), input_frames[0].shape))

    logging.debug('input_length: {}'.format(input_length.shape))
    logging.debug('zeros_placeholder: {}'.format(input_length.shape))
    
    # BEGIN CNN ################################################################
    logging.debug('---------- BEGIN CNN DEFINITION ----------')
    outputs = input_frames
    
    # layer 0 ##############################################################
    with tf.device('/gpu:0'):
        conv_b = new_bias(64)
        conv_w = new_conv2dweight(7, 7, 1, 64)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        outputs = [ max_pool(output) for output in outputs ]
        logger.log('maxpool_output', outputs)
        
    # layer 1 ##############################################################
    with tf.device('/gpu:0'):
        conv_b = new_bias(64)
        conv_w = new_conv2dweight(3, 3, 64, 64)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        outputs = [ max_pool(output) for output in outputs ]
        logger.log('maxpool_output', outputs)

    # convLSTM 1 (parts need to run on CPU) ####################################
    with tf.variable_scope('convlstm1'):
        num_filters = 16 #convLSTM internal fitlers
        h, w = int(outputs[0].shape[1]), int(outputs[0].shape[2])
        cell = ConvLSTMCell([h, w], num_filters, [3, 3], reuse=False)
        outputs, state = tf.nn.dynamic_rnn(
            cell,
            tf.stack(outputs, axis=1),
            dtype=tf.float32,
            sequence_length=input_length,
        )
        outputs = tf.unstack(outputs, axis=1)
        logger.log('convLSTM_output', outputs)

    with tf.device('/cpu:0'):
        h, w, n = int(outputs[0].shape[1]), int(outputs[0].shape[2]), int(outputs[0].shape[3])
        # h, w, n = 1, 1, num_filters
        outputs = [ tf.reshape(output, [-1, h*w*n]) for output in outputs ]
        logger.log('flatten_output', outputs)
        
        b_fc = new_bias(num_labels)
        w_fc = tf.Variable( tf.truncated_normal([h*w*n, num_labels], stddev=0.1, ) )
        outputs = [ tf.matmul(output, w_fc) + b_fc for output in outputs ]
        logger.log('fc_output', outputs)
    
    final_outputs = outputs

    # END CNN ##################################################################
    logging.debug('---------- END CNN DEFINITION ----------')

    # ACCURACY AND LOSS ########################################################
    predictions = tf.stack([ tf.nn.softmax(logits=output) for output in final_outputs ], axis=1)
    logging.debug('predictions: {}'.format(predictions.shape))
    
    cross_entropies = [ tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=output) for output in final_outputs ]
    logging.debug('cross_entropies: {}x{}'.format(len(cross_entropies), cross_entropies[0].shape))
    
    # only use the final loss to update weights
    #cross_entropies_truncated = [ tf.where(i > 0, zeros, cross_entropies[i]) for i in range(video_length) ]
    cross_entropies_truncated = [ tf.where(i > input_length, zeros, cross_entropies[i]) for i in range(video_length) ]

    # boost the loss on the last output by 10x
    last_cross_entropy  = [ tf.where(i < input_length, zeros, cross_entropies_truncated[i]) for i in range(video_length) ]
    cross_entropies_truncated = [ cross_entropies_truncated[i] + 10*last_cross_entropy[i] for i in range(video_length) ]
    
    losses = tf.stack(cross_entropies_truncated, axis=1, name='loss')
    logging.debug('losses: {}'.format(losses.shape))
    
    # logging.debug('intermediate_loss: {}'.format(tf.stack(cross_entropies_truncated).shape))
    # logging.debug('divide length: {}'.format((tf.stack(cross_entropies_truncated)/tf.to_float(input_length)).shape))
    
    total_loss = tf.reduce_sum(tf.reduce_sum(tf.stack(cross_entropies_truncated)) / tf.to_float(input_length), name='total_loss')
    logging.debug('total_loss: {}'.format(total_loss.shape))
        
    return losses, total_loss, predictions
    
def conv2d(x, w, b):
    output = leaky_relu( tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME'), b), LEAKINESS)
    output = batch_norm(output)
    return output

def leaky_relu(x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.nn.relu(x) - leakiness * tf.nn.relu(-x)

def max_pool(x):
    """MaxPool
    tf.nn.max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
    )
    """
    return tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'SAME', data_format='NHWC')

def batch_norm(x):
    """BatchNorm
    tf.layers.batch_normalization(
    inputs,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer=tf.zeros_initializer(),
    gamma_initializer=tf.ones_initializer(),
    moving_mean_initializer=tf.zeros_initializer(),
    moving_variance_initializer=tf.ones_initializer(),
    beta_regularizer=None,
    gamma_regularizer=None,
    training=False,
    trainable=True,
    name=None,
    reuse=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=False
    )
    """
    return tf.layers.batch_normalization(x)

def new_bias(length):
    return tf.Variable( tf.truncated_normal([length], stddev=0.1) )

def new_conv2dweight(xdim, ydim, input_depth, output_depth):
    return tf.Variable( tf.truncated_normal([xdim, ydim, input_depth, output_depth], stddev=0.1) )
