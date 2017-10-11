#!/usr/bin/env python3

import logging
import keras
import numpy as np
import tensorflow as tf
from vfeedbacknet.convLSTM import ConvLSTMCell # https://github.com/StanfordVisionSystems/tensorflow-convlstm-cell

#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)

LEAKINESS = 0.1

def vfeedbacknet_resnet(video_length, video_width, video_height, num_labels, input_placeholder, input_length, output_placeholder, zeros):
    '''
    A feedback model with resnet style model
    '''
    
    input_frames = tf.unstack(input_placeholder, axis=1)
    logging.debug('input: {}x{}'.format(len(input_frames), input_frames[0].shape))
    logging.debug('output: {}x{}'.format(0, output_placeholder.shape))

    logging.debug('---BEGIN MODEL DEFINTION---')

    # layer base (conv) ########################################################
    conv_b = new_bias(64)
    conv_w = new_conv2dweight(5, 5, 3, 64)
    outputs = [ conv2d(input_frame, conv_w, conv_b) for input_frame in input_frames ]
    logging.debug('conv1_output: {}x{}'.format(len(outputs), outputs[0].shape))
    
    outputs = [ batch_norm(output) for output in outputs ]
    logging.debug('batchNorm1_output: {}x{}'.format(len(outputs), outputs[0].shape))

    outputs = [ max_pool(output) for output in outputs ]
    logging.debug('maxPool1_output: {}x{}'.format(len(outputs), outputs[0].shape))

    # layer1 (conv) ############################################################
    outputs_res = outputs
    
    conv_b = new_bias(64)
    conv_w = new_conv2dweight(3, 3, 64, 64)
    outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
    logging.debug('conv2_output: {}x{}'.format(len(outputs), outputs[0].shape))
    
    conv_b = new_bias(64)
    conv_w = new_conv2dweight(3, 3, 64, 64)
    outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
    logging.debug('conv3_output: {}x{}'.format(len(outputs), outputs[0].shape))

    outputs = [outputs[i] + outputs_res[i] for i in range(len(outputs))]
    logging.debug('residual1_output: {}x{}'.format(len(outputs), outputs[0].shape))
    
    outputs = [ batch_norm(output) for output in outputs ]
    logging.debug('batchNorm2_output: {}x{}'.format(len(outputs), outputs[0].shape))

    # layer2 (conv) ############################################################
    outputs_res = outputs
    
    conv_b = new_bias(64)
    conv_w = new_conv2dweight(3, 3, 64, 64)
    outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
    logging.debug('conv4_output: {}x{}'.format(len(outputs), outputs[0].shape))
    
    conv_b = new_bias(64)
    conv_w = new_conv2dweight(3, 3, 64, 64)
    outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
    logging.debug('conv5_output: {}x{}'.format(len(outputs), outputs[0].shape))

    outputs = [outputs[i] + outputs_res[i] for i in range(len(outputs))]
    logging.debug('residual2_output: {}x{}'.format(len(outputs), outputs[0].shape))
    
    outputs = [ batch_norm(output) for output in outputs ]
    logging.debug('batchNorm3_output: {}x{}'.format(len(outputs), outputs[0].shape))

    # # layer3 (conv) ############################################################
    # outputs_res = outputs
    
    # conv_b = new_bias(64)
    # conv_w = new_conv2dweight(3, 3, 64, 64)
    # outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
    # logging.debug('conv6_output: {}x{}'.format(len(outputs), outputs[0].shape))
    
    # conv_b = new_bias(64)
    # conv_w = new_conv2dweight(3, 3, 64, 64)
    # outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
    # logging.debug('conv7_output: {}x{}'.format(len(outputs), outputs[0].shape))

    # outputs = [outputs[i] + outputs_res[i] for i in range(len(outputs))]
    # logging.debug('residual3_output: {}x{}'.format(len(outputs), outputs[0].shape))
    
    # outputs = [ batch_norm(output) for output in outputs ]
    # logging.debug('batchNorm4_output: {}x{}'.format(len(outputs), outputs[0].shape))

    # # layer4 (conv) ############################################################
    # outputs_res = outputs
    
    # conv_b = new_bias(64)
    # conv_w = new_conv2dweight(3, 3, 64, 64)
    # outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
    # logging.debug('conv8_output: {}x{}'.format(len(outputs), outputs[0].shape))
    
    # conv_b = new_bias(64)
    # conv_w = new_conv2dweight(3, 3, 64, 64)
    # outputs = [ conv2d(output, conv_w, conv_b) for output in outputs ]
    # logging.debug('conv9_output: {}x{}'.format(len(outputs), outputs[0].shape))

    # outputs = [outputs[i] + outputs_res[i] for i in range(len(outputs))]
    # logging.debug('residual4_output: {}x{}'.format(len(outputs), outputs[0].shape))
    
    # outputs = [ batch_norm(output) for output in outputs ]
    # logging.debug('batchNorm5_output: {}x{}'.format(len(outputs), outputs[0].shape))

    # layer (convLSTM) #########################################################
    #logging.debug('input_length: {}'.format(input_length.shape))
    num_filters = 128 # convLSTM internal fitlers
    outputs, state = tf.nn.dynamic_rnn(
        ConvLSTMCell([outputs[0].shape[1], outputs[0].shape[2]], num_filters, [3, 3]),
        tf.stack(outputs, axis=0),
        dtype=tf.float32,
        sequence_length=input_length,
        time_major=True,
    )
    outputs = tf.unstack(outputs, axis=0)
    logging.debug('convLSTM1_output: {}x{}'.format(len(outputs), outputs[0].shape))

    outputs = [tf.reduce_mean(output, axis=[1,2]) for output in outputs]
    logging.debug('avePool1_output: {}x{}'.format(len(outputs), outputs[0].shape))
    
    # fc layer #################################################################
    outputs = [ tf.reshape(output, [-1, num_filters]) for output in outputs ]
    logging.debug('flatten1_output: {}x{}'.format(len(outputs), outputs[0].shape))

    b_fc = new_bias(num_labels)
    w_fc = tf.truncated_normal([num_filters, num_labels], stddev=0.1)
    #logging.debug('fc1: {}'.format(w_fc.shape))
    
    outputs = [ tf.matmul(output, w_fc) + b_fc for output in outputs ]
    logging.debug('fc1_output: {}x{}'.format(len(outputs), outputs[0].shape))

    logging.debug('---END MODEL DEFINTION---')

    # loss and precitions (softmax) ############################################
    network_outputs = outputs
    
    cross_entropies = [ tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=network_output) for network_output in network_outputs ]
    logging.debug('cross_entropies: {}x{}'.format(len(cross_entropies), cross_entropies[0].shape))
    
    cross_entropies_truncated = [ tf.where(input_length > i, cross_entropies[i], zeros) for i in range(video_length) ]
    logging.debug('cross_entropies_truncated: {}x{}'.format(len(cross_entropies_truncated), cross_entropies_truncated[0].shape))

    per_video_losses = tf.reduce_sum(tf.stack(cross_entropies_truncated, axis=0), axis=0) / tf.to_float(input_length)
    logging.debug('per_video_losses: {}'.format(per_video_losses.shape))
    
    loss = tf.reduce_sum(per_video_losses, name='loss')
    logging.debug('loss: {}'.format(loss.shape))

    predictions = tf.stack([tf.nn.softmax(network_output) for network_output in network_outputs], name='predictions')
    logging.debug('predictions: {}'.format(predictions.shape))

    return loss, predictions
    
def conv2d(x, w, b):
    """Conv2d
    tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=None,
    data_format=None,
    name=None
    )
    """
    return leaky_relu(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME') + b, LEAKINESS)

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


def new_bias(output_depth):
    return tf.Variable( tf.truncated_normal([output_depth], stddev=0.1) )

def new_conv2dweight(xdim, ydim, input_depth, output_depth):
    return tf.Variable( tf.truncated_normal([xdim, ydim, input_depth, output_depth], stddev=0.1) )
