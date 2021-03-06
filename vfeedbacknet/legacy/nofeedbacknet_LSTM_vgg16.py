import vfeedbacknet.legacy.vgg16_model_short as vgg16_model
import vfeedbacknet.convLSTM as convLSTM

import tensorflow as tf
import logging

from vfeedbacknet.vfeedbacknet_utilities import ModelLogger
from vfeedbacknet.vfeedbacknet_lossfunctions import basic_loss_pred

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

class NoFeedbackNetLSTMVgg16:
    
    def __init__(self, sess, vgg16_weights, num_classes=101, fine_tune_vgg16=False, is_training=True):
        self.sess = sess
        self.vgg16_weights = vgg16_weights
        self.num_classes = num_classes
        self.fine_tune_vgg16 = fine_tune_vgg16
        self.is_training = is_training

        self.declare_variables()
        
    def declare_variables(self):
        '''
        Declare all the necessary variables so they can be referenced and reused
        during the model contruction
        '''
        with tf.variable_scope('NoFeedBackNetVgg16'):

            self.vgg_layers = vgg16_model.VGG16(sess=self.sess,
                                                weights=self.vgg16_weights,
                                                trainable=self.fine_tune_vgg16)
            with tf.variable_scope('fc'):
                regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                kernel = tf.get_variable('weights', shape=[512, self.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer, trainable=self.is_training)
                biases = tf.get_variable('biases', shape=[self.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer, trainable=self.is_training)
                
    def initialize_variables(self):
        '''
        Load the VGG16 pretrained parameters and initialize the other variables
        '''
        with tf.variable_scope('NoFeedBackNetVgg16', reuse=True):

            logging.debug('--- begin initialize variables ---')

            self.sess.run(tf.global_variables_initializer())
            self.vgg_layers.load_weights()

            logging.debug('--- end initialize variables ---')
        
    def __call__(self, inputs, inputs_sequence_length):
        '''
        inputs: A tensor fo size [batch, video_length, video_height, video_width, channels]
        '''

        with tf.variable_scope('NoFeedBackNetVgg16', reuse=True):
            
            ModelLogger.log('input', inputs)
            
            #assert(inputs.shape[1:] == (40, 96, 96)) # specific model shape for now        
            inputs = tf.expand_dims(inputs, axis=4)
            #assert(inputs.shape[1:] == (40, 96, 96, 1)) # specific model shape for now
            
            inputs = tf.unstack(inputs, axis=1)
            ModelLogger.log('input-unstack', inputs)
            
            logging.debug('--- begin model definition ---')
            
            # use VGG16 pretrained on imagenet as an initialization        
            inputs = [ self.vgg_layers(inp) for inp in inputs ]
            ModelLogger.log('vgg16_conv', inputs)
            
        # use feedback network architecture below
        with tf.variable_scope('NoFeedBackNetVgg16'):
            with tf.variable_scope('convlstm1'):
                
                num_filters = 512 # convLSTM internal fitlers
                h, w = int(inputs[0].shape[1]), int(inputs[0].shape[2])
                cell = convLSTM.ConvLSTMCell([h, w], num_filters, [3, 3])
                inputs, state = tf.nn.dynamic_rnn(
                    cell,
                    tf.stack(inputs, axis=1),
                    dtype=tf.float32,
                    sequence_length=inputs_sequence_length,
                )
                
                inputs = tf.unstack(inputs, axis=1)
                ModelLogger.log('convLSTM_output', inputs)
                
                inputs = [ tf.nn.max_pool(inp,
                                          ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME',
                                          name='pool1') for inp in inputs ]
                ModelLogger.log('pool_output', inputs)
                    
            with tf.variable_scope('convlstm2', reuse=False):
                    num_filters = 512 # convLSTM internal fitlers
                    h, w = int(inputs[0].shape[1]), int(inputs[0].shape[2])
                    cell = convLSTM.ConvLSTMCell([h, w], num_filters, [3, 3], reuse=False)
                    inputs, state = tf.nn.dynamic_rnn(
                        cell,
                        tf.stack(inputs, axis=1),
                        dtype=tf.float32,
                        sequence_length=inputs_sequence_length,
                    )

                    inputs = tf.unstack(inputs, axis=1)
                    ModelLogger.log('convLSTM_output', inputs)

                    inputs = [ tf.nn.max_pool(inp,
                                              ksize=[1, 2, 2, 1],
                                              strides=[1, 2, 2, 1],
                                              padding='SAME',
                                              name='pool1') for inp in inputs ]
                    ModelLogger.log('pool_output', inputs)
                    
        with tf.variable_scope('NoFeedBackNetVgg16', reuse=True):
            with tf.variable_scope('fc', reuse=True):

                inputs = [ tf.layers.average_pooling2d(
                    inputs=inp, pool_size=4, strides=1, padding='VALID',
                    data_format='channels_last', name='ave_pool') for inp in inputs ]
                ModelLogger.log('ave_pool_output', inputs)

                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')

                inputs = [ tf.reshape(inp, [-1, 512]) for inp in inputs ]
                ModelLogger.log('flatten_output', inputs)

                inputs = [ tf.matmul(inp, weights) + biases for inp in inputs ]
                ModelLogger.log('fc_output', inputs)

                logging.debug('--- end model definition ---')

                logits = inputs
                ModelLogger.log('logits', logits)

        return logits
        
if __name__ == '__main__':
    sess = tf.Session()
    
    model = NoFeedbackNetVgg16(sess, '/home/jemmons/vgg16_weights.npz')

    x = tf.placeholder(tf.float32, [None, 40, 96, 96], name='inputs')
    x_len = tf.placeholder(tf.float32, [None], name='inputs_len')
    zeros = tf.placeholder(tf.float32, [40], name='inputs_len')
    labels = tf.placeholder(tf.float32, [None], name='inputs_len')
    logits = model(x, x_len)

    losses, total_loss, predictions = basic_loss_pred(logits, x_len, len(logits), labels, zeros)
    
    model.initialize_variables()

    # use the model

    
    # graph = tf.get_default_graph()    
    # for op in graph.get_operations():
    #     print((op.name))
