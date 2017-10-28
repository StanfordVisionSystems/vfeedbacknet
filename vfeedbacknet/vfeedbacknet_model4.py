import numpy as np
import tensorflow as tf
import logging

from vfeedbacknet.vfeedbacknet_convLSTM import ConvLSTMCell
from vfeedbacknet.vfeedbacknet_utilities import ModelLogger
from vfeedbacknet.vfeedbacknet_base import VFeedbackNetBase

class Model:
    '''
    convLSTM based feedback mechanism
    '''

    
    model_name = 'model4'
    
    def __init__(self, sess, num_classes, batch_size,
                 train_featurizer='FINE_TUNE', train_main_model='FINE_TUNE', train_fc='FINE_TUNE',
                 weights_filename=None, is_training=True):

        self.sess = sess
        self.weights = np.load(weights_filename) if weights_filename is not None else None
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        assert train_featurizer in ['NO', 'FINE_TUNE', 'FROM_SCRATCH'], 'train_featurizer must be either: NO, FINE_TUNE, or FROM_SCRATCH'
        self.train_featurizer = train_featurizer if is_training else 'NO'

        assert train_main_model in ['NO', 'FINE_TUNE', 'FROM_SCRATCH'], 'train_main_model must be either: NO, FINE_TUNE, or FROM_SCRATCH'
        self.train_main_model = train_main_model if is_training else 'NO'

        assert train_fc in ['NO', 'FINE_TUNE', 'FROM_SCRATCH'], 'train_fc must be either: NO, FINE_TUNE, or FROM_SCRATCH'
        self.train_fc = train_fc if is_training else 'NO'

        self.is_training = is_training

        self.featurizer_variables = []
        self.main_model_variables = []
        self.fc_variables = []
        
        self.vfeedbacknet_base = VFeedbackNetBase(sess, num_classes, train_vgg16=train_featurizer, is_training=is_training)
        self._declare_variables()


    def _declare_variables(self):

        with tf.variable_scope('vfeedbacknet_{}'.format(Model.model_name)):
            with tf.variable_scope('convlstm1'):
                with tf.variable_scope('rnn'):
                    with tf.variable_scope('conv_lstm_cell'):

                        regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                        initializer = tf.contrib.layers.xavier_initializer()

                        n = 256
                        m = 4*n
                        input_size = [14, 14, n]
                        kernel2d_size = [3, 3]
                        kernel_size = kernel2d_size + [2*n] + [m] 

                        with tf.variable_scope('convlstm'):
                            kernel = tf.get_variable('kernel', kernel_size, initializer=initializer, regularizer=regularizer)
                            W_ci = tf.get_variable('W_ci', input_size, initializer=initializer, regularizer=regularizer)
                            W_cf = tf.get_variable('W_cf', input_size, initializer=initializer, regularizer=regularizer)
                            W_co = tf.get_variable('W_co', input_size, initializer=initializer, regularizer=regularizer)
                            bias = tf.get_variable('bias', [m], initializer=tf.zeros_initializer(), regularizer=regularizer)
                            
                self.convLSTMCell1 = ConvLSTMCell([14, 14], 256, [3, 3])
                        
            with tf.variable_scope('conv1'):

                regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                initializer = tf.contrib.layers.xavier_initializer()

                kernel = tf.get_variable('kernel', shape=[14, 14, 256, 512], dtype=tf.float32, regularizer=regularizer, initializer=initializer)
                biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, regularizer=regularizer, initializer=initializer)

            with tf.variable_scope('dconv1'):

                regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                initializer = tf.contrib.layers.xavier_initializer()

                kernel = tf.get_variable('kernel', [3, 3, 128, 256], initializer=initializer, regularizer=regularizer)
                biases = tf.get_variable('biases', [128], initializer=initializer, regularizer=regularizer)


    def get_variables(self):

        return self.featurizer_variables + self.main_model_variables + self.fc_variables

    
    def print_variables(self):

        for var in self.get_variables():
            print(var.name)


    def initialize_variables(self):

        logging.debug('--- begin variable initialization (vfeedbacknet) ---')

        if self.train_featurizer == 'FROM_SCRATCH':
            logging.debug('vgg16:FROM_SCRATCH; using random initialization')
            for var in self.featurizer_variables:
                self.sess.run(var.initializer)
        else:
            for var in self.featurizer_variables:
                logging.debug('LOADING FROM WEIGHTS_FILE {}: {}'.format(var.name, var.shape))
                assert self.weights is not None, 'Need to specify weights file to load from for featurizer_variables'
                self.sess.run(var.assign(self.weights[var.name]))

        if self.train_main_model  == 'FROM_SCRATCH':
            logging.debug('feedback: FROM_SCRATCH; using random initialization')
            for var in self.main_model_variables:
                self.sess.run(var.initializer)
        else:
            for var in self.main_model_variables:
                logging.debug(' LOADING FROM WEIGHTS_FILE {}: {}'.format(var.name, var.shape))
                assert self.weights is not None, 'Need to specify weights file to load from for main_model_variables'
                self.sess.run(var.assign(self.weights[var.name]))

        if self.train_fc == 'FROM_SCRATCH':
            logging.debug('fc: FROM_SCRATCH; using random initialization')
            for var in self.fc_variables:
                self.sess.run(var.initializer)
        else:
            for var in self.fc_variables:
                logging.debug('LOADING FROM WEIGHTS_FILE {}: {}'.format(var.name, var.shape))
                assert self.weights is not None, 'Need to specify weights file to load from for fc_variables'
                self.sess.run(var.assign(self.weights[var.name]))

        logging.debug('--- end variable initialization (vfeedbacknet) ---')


    def export_variables(self, export_filename):

        VFeedbackNetBase.export_variables(self.sess, self.get_variables(), export_filename)
            

    def __call__(self, inputs, inputs_sequence_length):

        #assert inputs.shape[1:] == (20, 112, 112), 'expected input shape of (20, 112, 112) but got {}'.format(inputs.shape)

        ModelLogger.log('raw_input', inputs)

        inputs = self.vfeedbacknet_base.split_video(inputs)
        ModelLogger.log('input', inputs)

        ## featurizer ##
        inputs = [ self.vfeedbacknet_base.vgg16_layer1(inp, var_list=self.featurizer_variables) for inp in inputs ]
        ModelLogger.log('vgg-layer1', inputs)

        inputs = [ self.vfeedbacknet_base.vgg16_layer2(inp, var_list=self.featurizer_variables) for inp in inputs ]
        ModelLogger.log('vgg-layer2', inputs)
        
        inputs = [ self.vfeedbacknet_base.vgg16_layer2(inp, var_list=self.featurizer_variables) for inp in inputs ]
        ModelLogger.log('vgg-layer2', inputs)

        inputs = [ self.vfeedbacknet_base.vgg16_layer2(inp, var_list=self.featurizer_variables) for inp in inputs ]
        ModelLogger.log('vgg-layer2', inputs)

        
        ## main model ##
        logits = []
        featurizer_outputs = inputs
        feedback_outputs = None

        # def feedback_block(

        # feedback 1
        inputs = featurizer_outputs
        inputs = [ self.vfeedbacknet_base.vgg16_layer3(inp, var_list=self.main_model_variables) for inp in inputs ]
        ModelLogger.log('vgg-layer3', inputs)

        inputs = self.convLSTM_layer1(inputs, inputs_sequence_length, var_list=self.main_model_variables)
        ModelLogger.log('convLSTM1', inputs)
        feedback_outputs = inputs
        
        inputs = [ self.conv_layer(inp, var_list=self.main_model_variables) for inp in inputs ]
        ModelLogger.log('conv', inputs)
                       
        inputs = [ self.vfeedbacknet_base.ave_pool(inp) for inp in inputs ]
        ModelLogger.log('ave_pool', inputs)

        inputs = [ self.vfeedbacknet_base.fc_layer(inp, var_list=self.fc_variables) for inp in inputs ]
        ModelLogger.log('fc', inputs)
        logits.append(tf.stack(inputs, axis=1))
        
        # feedback 2
        #inputs = [ featurizer_o + self.dconv_layer(feedback_o, var_list=self.main_model_variables) for featurizer_o,feedback_o in zip(featurizer_outputs,feedback_outputs) ]
        # inputs = [ self.dconv_layer(feedback_o, var_list=self.main_model_variables) for featurizer_o,feedback_o in zip(featurizer_outputs,feedback_outputs) ]
        # ModelLogger.log('dconv0', inputs)

        # inputs = [ self.vfeedbacknet_base.vgg16_layer3(inp, var_list=self.main_model_variables) for inp in inputs ]
        # ModelLogger.log('vgg-layer3', inputs)

        # inputs = self.convLSTM_layer1(inputs, inputs_sequence_length, var_list=self.main_model_variables)
        # ModelLogger.log('convLSTM1', inputs)
        # feedback_outputs = inputs

        # inputs = [ self.conv_layer(inp, var_list=self.main_model_variables) for inp in inputs ]
        # ModelLogger.log('conv0', inputs)
        
        # inputs = [ self.vfeedbacknet_base.ave_pool(inp) for inp in inputs ]
        # ModelLogger.log('ave_pool', inputs)

        # inputs = [ self.vfeedbacknet_base.fc_layer(inp, var_list=self.fc_variables) for inp in inputs ]
        # ModelLogger.log('fc', inputs)
        # logits.append(tf.stack(inputs, axis=1))

        # # feedback 3
        # inputs = [ featurizer_o + self.dconv_layer(feedback_o, var_list=self.main_model_variables) for featurizer_o,feedback_o in zip(featurizer_outputs,feedback_outputs) ]
        # ModelLogger.log('dconvo', inputs)

        # inputs = [ self.vfeedbacknet_base.vgg16_layer3(inp, var_list=self.featurizer_variables) for inp in inputs ]
        # ModelLogger.log('vgg-layer3', inputs)

        # inputs = self.convLSTM_layer1(inputs, inputs_sequence_length, var_list=self.main_model_variables)
        # ModelLogger.log('convLSTM1', inputs)
        # feedback_outputs = inputs

        # inputs = [ self.conv_layer(inp, var_list=self.main_model_variables) for inp in inputs ]
        # ModelLogger.log('conv0', inputs)

        # inputs = [ self.vfeedbacknet_base.ave_pool(inp) for inp in inputs ]
        # ModelLogger.log('ave_pool', inputs)

        # inputs = [ self.vfeedbacknet_base.fc_layer(inp, var_list=self.fc_variables) for inp in inputs ]
        # ModelLogger.log('fc', inputs)
        # logits.append(tf.stack(inputs, axis=1))

        logits = tf.stack(logits, axis=1)
        return logits

    
    # def conv_layer(self, inputs, var_list=None):

    #     with tf.variable_scope('vfeedbacknet_{}'.format(Model.model_name), reuse=True):
    #         with tf.variable_scope('conv1'):
    #             kernel = tf.get_variable('kernel')
    #             biases = tf.get_variable('biases')

    #             inputs = tf.nn.conv2d(inputs, kernel, [1, 2, 2, 1], padding='SAME')
    #             inputs = tf.nn.bias_add(inputs, biases)
    #             inputs = tf.nn.relu(inputs)

    #             # inputs = tf.nn.max_pool(inputs,
    #             #                         ksize=[1, 2, 2, 1],
    #             #                         strides=[1, 2, 2, 1],
    #             #                         padding='VALID')
                
    #             if var_list is not None and kernel not in var_list:
    #                 var_list.append(kernel)
    #             if var_list is not None and biases not in var_list:
    #                 var_list.append(biases)
                
    #             return inputs


    # def dconv_layer(self, inputs, var_list=None):

    #     with tf.variable_scope('vfeedbacknet_{}'.format(Model.model_name), reuse=True):
    #         with tf.variable_scope('dconv1'):
    #             kernel = tf.get_variable('kernel')
    #             biases = tf.get_variable('biases')

    #             inputs = tf.nn.conv2d_transpose(inputs, kernel, tf.stack([self.batch_size, 28, 28, 128]), [1, 2, 2, 1], padding='SAME')
    #             inputs = tf.nn.bias_add(inputs, biases)
    #             inputs = tf.nn.relu(inputs)

    #             if var_list is not None and kernel not in var_list:
    #                 var_list.append(kernel)
    #             if var_list is not None and biases not in var_list:
    #                 var_list.append(biases)
                
    #             return inputs
                

    def convLSTM_layer1(self, inputs, inputs_sequence_length, var_list=None):

        with tf.variable_scope('vfeedbacknet_{}'.format(Model.model_name)):
            with tf.variable_scope('convlstm1'):
                with tf.variable_scope('rnn'):
                    with tf.variable_scope('conv_lstm_cell'):
                        with tf.variable_scope('convlstm', reuse=True):
                            kernel = tf.get_variable('kernel')
                            W_ci = tf.get_variable('W_ci')
                            W_cf = tf.get_variable('W_cf')
                            W_co = tf.get_variable('W_co')
                            bias = tf.get_variable('bias')

                            if var_list is not None and kernel not in var_list:
                                var_list.append(kernel)
                            if var_list is not None and W_ci not in var_list:
                                var_list.append(W_ci)
                            if var_list is not None and W_cf not in var_list:
                                var_list.append(W_cf)
                            if var_list is not None and W_co not in var_list:
                                var_list.append(W_co)
                            if var_list is not None and bias not in var_list:
                                var_list.append(bias)

                print(inputs)
                inputs = tf.stack(inputs, axis=0),
                print(inputs)
                print(inputs.shape)
                
                inputs, state = tf.nn.dynamic_rnn(
                    self.convLSTMCell1,
                    tf.stack(inputs, axis=0),
                    dtype=tf.float32,
                    sequence_length=inputs_sequence_length,
                    swap_memory=True,
                    time_major=True
                )
                print(inputs.shape)

                inputs = tf.unstack(inputs, axis=0)
                print(len(inputs), inputs[0].shape)
                
                return inputs


if __name__ == '__main__':

    sess = tf.Session()

    video_length = 20

    x = tf.placeholder(tf.float32, [None, video_length, 112, 112], name='inputs')
    x_len = tf.placeholder(tf.float32, [None], name='inputs_len')
    zeros = tf.placeholder(tf.float32, [video_length], name='inputs_len')
    labels = tf.placeholder(tf.float32, [None], name='inputs_len')

    vfeedbacknet_model = Model(sess, 27, 8, train_featurizer='NO', train_main_model='FROM_SCRATCH', train_fc='FROM_SCRATCH', weights_filename='/home/jemmons/vfeedbacknet_base_weights.npz')

    logits = vfeedbacknet_model(x, x_len)
    ModelLogger.log('logits', logits)

    vfeedbacknet_model.initialize_variables()
    vfeedbacknet_model.export_variables('/tmp/weights.npz')
    vfeedbacknet_model.print_variables()
    
    # print out the model
    # graph = tf.get_default_graph()    
    # for op in graph.get_operations():
    #     print((op.name))

    
