#!/usr/bin/env python3

import logging

import keras
import numpy as np
import tensorflow as tf
from vfeedbacknet.convLSTM import ConvLSTMCell # https://github.com/StanfordVisionSystems/tensorflow-convlstm-cell
    
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)

LEAKINESS = 0.1

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
            
def nofeedbacknet_resnet_uniform(video_length, video_width, video_height, num_labels, input_placeholder, input_length, output_placeholder, zeros):
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
    with tf.device("/device:GPU:0"):
        outputs = input_frames
    
        # layer 0 ##############################################################
        conv_b = new_bias(32)
        conv_w = new_conv2dweight(10, 10, 1, 32)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        outputs = [ max_pool(output) for output in outputs ]
        logger.log('maxpool_output', outputs)

        
        # layer 1 ##############################################################
        prev_outputs = outputs
        
        conv_b = new_bias(32)
        conv_w = new_conv2dweight(3, 3, 32, 32)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        conv_b = new_bias(32)
        conv_w = new_conv2dweight(3, 3, 32, 32)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        outputs = [ outputs[i] + prev_outputs[i] for i in range(len(outputs))]
        logger.log('residual_output', outputs)

        outputs = [ max_pool(output) for output in outputs ]
        logger.log('maxpool_output', outputs)

        
        # layer 2 ##############################################################
        prev_outputs = outputs
        
        conv_b = new_bias(32)
        conv_w = new_conv2dweight(3, 3, 32, 32)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        conv_b = new_bias(32)
        conv_w = new_conv2dweight(3, 3, 32, 32)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        #outputs = [ outputs[i] + tf.contrib.keras.backend.repeat_elements(prev_outputs[i], 2, 3) for i in range(len(outputs))]
        outputs = [ outputs[i] + prev_outputs[i] for i in range(len(outputs))]
        logger.log('residual_output', outputs)

        outputs = [ max_pool(output) for output in outputs ]
        logger.log('maxpool_output', outputs)


        # layer 3 ##############################################################
        prev_outputs = outputs
        
        conv_b = new_bias(32)
        conv_w = new_conv2dweight(3, 3, 32, 32)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        conv_b = new_bias(32)
        conv_w = new_conv2dweight(3, 3, 32, 32)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        #outputs = [ outputs[i] + tf.contrib.keras.backend.repeat_elements(prev_outputs[i], 2, 3) for i in range(len(outputs))]
        outputs = [ outputs[i] + prev_outputs[i] for i in range(len(outputs))]
        logger.log('residual_output', outputs)

        outputs = [ max_pool(output) for output in outputs ]
        logger.log('maxpool_output', outputs)

        
        # layer 4 ##############################################################
        prev_outputs = outputs
        
        conv_b = new_bias(32)
        conv_w = new_conv2dweight(3, 3, 32, 32)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        conv_b = new_bias(32)
        conv_w = new_conv2dweight(3, 3, 32, 32)
        outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
        logger.log('conv_output', outputs)
        
        #outputs = [ outputs[i] + tf.contrib.keras.backend.repeat_elements(prev_outputs[i], 2, 3) for i in range(len(outputs))]
        outputs = [ outputs[i] + prev_outputs[i] for i in range(len(outputs))]
        logger.log('residual_output', outputs)

        outputs = [ max_pool(output) for output in outputs ]
        logger.log('maxpool_output', outputs)

        
    # convLSTM 1 (parts need to run on CPU) ####################################
    num_filters = 64 # convLSTM internal fitlers
    h, w = int(outputs[0].shape[1]), int(outputs[0].shape[2])
    output, state = tf.nn.dynamic_rnn(
        ConvLSTMCell([h, w], num_filters, [3, 3]),
        tf.stack(outputs, axis=1),
        dtype=tf.float32,
        sequence_length=input_length,
    )
    logger.log('convLSTM_output', outputs)

    outputs = [tf.reduce_mean(output, axis=[1,2]) for output in outputs]
    logger.log('avepool_output', outputs)

    with tf.device("/device:CPU:0"):
        outputs = tf.unstack(output, axis=1)
        logger.log('unstack_output', outputs)
    
        outputs = [ tf.reshape(output, [-1, h*w*num_filters]) for output in outputs ]
        logger.log('flatten_output', outputs)
        
        b_fc = new_bias(num_labels)
        w_fc = tf.Variable( tf.truncated_normal([h*w*num_filters, num_labels], stddev=0.1) )
        outputs = [ tf.matmul(output, w_fc) + b_fc for output in outputs ]

        logger.log('fc_output', outputs)
    
        final_outputs = outputs

    # END CNN ##################################################################
    logging.debug('---------- END CNN DEFINITION ----------')

    # ACCURACY AND LOSS ########################################################
    with tf.device("/device:CPU:0"):
        softmaxes = [ tf.nn.softmax(logits=output) for output in final_outputs ]
        predictions = tf.stack(final_outputs, name='predictions') #tf.stack(softmaxes)
        
        cross_entropies = [ tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=output) for output in final_outputs ]
        
        cross_entropies_truncated = tf.stack([ tf.where(input_length > i, cross_entropies[i], zeros) for i in range(video_length) ], axis=1)
        loss = tf.reduce_sum(tf.reduce_sum(cross_entropies, axis=0) / tf.to_float(input_length), name='loss')
        logging.debug('loss: {}'.format(loss.shape))
        
    return loss, predictions
    
def conv2d(x, w, b):
    output = leaky_relu( batch_norm(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME') + b) , LEAKINESS)
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
    return tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], 'SAME', data_format='NHWC')

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
