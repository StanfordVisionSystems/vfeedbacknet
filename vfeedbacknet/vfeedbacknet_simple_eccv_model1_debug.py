import numpy as np
import tensorflow as tf
import logging

from vfeedbacknet.vfeedbacknet_convLSTM import ConvLSTMCell
from vfeedbacknet.vfeedbacknet_utilities import ModelLogger

from vfeedbacknet.vfeedbacknet_base import VFeedbackNetBase
from vfeedbacknet.vfeedbacknet_feedbackCell import FeedbackLSTMCell_stack1


class Model:
    '''
    TODO(jremmons) add description
    '''

    model_name = 'simple_eccv_model1_debug'
    NFEEDBACK_ITERATIONS = 3
    
    def __init__(self, sess, num_classes, batch_size,
                 train_featurizer='FINE_TUNE', train_main_model='FINE_TUNE', train_fc='FINE_TUNE',
                 weights_filename=None, is_training=True):

        self.sess = sess
        if weights_filename is not None:
            print('loading weights from: {}'.format(weights_filename))
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
        #self.vfeedbacknet_fb_base = VFeedbackNetFBBase(sess, num_classes, batch_size, train_featurizer='NO', train_main_model='NO', is_training=True, trainable=False)
        self._declare_variables()


    def _declare_variables(self):

        with tf.variable_scope('vfeedbacknet_{}'.format(Model.model_name)):

            with tf.variable_scope('feedbackcell1'):
                self.feedbackLSTMCell1 = FeedbackLSTMCell_stack1([56, 56, 64], Model.NFEEDBACK_ITERATIONS)

            with tf.variable_scope('convlstm1'):
                with tf.variable_scope('rnn'):
                    with tf.variable_scope('conv_lstm_cell'):

                        regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                        initializer = tf.contrib.layers.xavier_initializer()

                        n = 128
                        m = 4*n
                        input_size = [28, 28, n]
                        kernel2d_size = [3, 3]
                        kernel_size = kernel2d_size + [2*n] + [m] 

                        with tf.variable_scope('convlstm'):
                            kernel = tf.get_variable('kernel', kernel_size, initializer=initializer, regularizer=regularizer)
                            W_ci = tf.get_variable('W_ci', input_size, initializer=initializer, regularizer=regularizer)
                            W_cf = tf.get_variable('W_cf', input_size, initializer=initializer, regularizer=regularizer)
                            W_co = tf.get_variable('W_co', input_size, initializer=initializer, regularizer=regularizer)
                            bias = tf.get_variable('bias', [m], initializer=tf.zeros_initializer(), regularizer=regularizer)
                            
                self.convLSTMCell1 = ConvLSTMCell(input_size[:2], n, [3, 3])

            # with tf.variable_scope('convlstm2'):
            #     with tf.variable_scope('rnn'):
            #         with tf.variable_scope('conv_lstm_cell'):

            #             regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
            #             initializer = tf.contrib.layers.xavier_initializer()

            #             n = 512
            #             m = 4*n
            #             input_size = [7, 7, n]
            #             kernel2d_size = [3, 3]
            #             kernel_size = kernel2d_size + [2*n] + [m] 

            #             with tf.variable_scope('convlstm'):
            #                 kernel = tf.get_variable('kernel', kernel_size, initializer=initializer, regularizer=regularizer)
            #                 W_ci = tf.get_variable('W_ci', input_size, initializer=initializer, regularizer=regularizer)
            #                 W_cf = tf.get_variable('W_cf', input_size, initializer=initializer, regularizer=regularizer)
            #                 W_co = tf.get_variable('W_co', input_size, initializer=initializer, regularizer=regularizer)
            #                 bias = tf.get_variable('bias', [m], initializer=tf.zeros_initializer(), regularizer=regularizer)
                            
            #     self.convLSTMCell2 = ConvLSTMCell(input_size[:2], n, [3, 3])
                
            # with tf.variable_scope('convlstm3'):
            #     with tf.variable_scope('rnn'):
            #         with tf.variable_scope('conv_lstm_cell'):

            #             regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
            #             initializer = tf.contrib.layers.xavier_initializer()

            #             n = 512
            #             m = 4*n
            #             input_size = [7, 7, n]
            #             kernel2d_size = [3, 3]
            #             kernel_size = kernel2d_size + [2*n] + [m] 

            #             with tf.variable_scope('convlstm'):
            #                 kernel = tf.get_variable('kernel', kernel_size, initializer=initializer, regularizer=regularizer)
            #                 W_ci = tf.get_variable('W_ci', input_size, initializer=initializer, regularizer=regularizer)
            #                 W_cf = tf.get_variable('W_cf', input_size, initializer=initializer, regularizer=regularizer)
            #                 W_co = tf.get_variable('W_co', input_size, initializer=initializer, regularizer=regularizer)
            #                 bias = tf.get_variable('bias', [m], initializer=tf.zeros_initializer(), regularizer=regularizer)
                            
            #     self.convLSTMCell3 = ConvLSTMCell(input_size[:2], n, [3, 3])
                
            with tf.variable_scope('reshape_convs'):
                with tf.variable_scope('conv1'):

                    regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                    initializer = tf.contrib.layers.xavier_initializer()

                    kernel = tf.get_variable('kernel', shape=[7, 7, 64, 128], dtype=tf.float32, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, regularizer=regularizer, initializer=initializer)
            
                with tf.variable_scope('conv2'):

                    regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                    initializer = tf.contrib.layers.xavier_initializer()

                    kernel = tf.get_variable('kernel', shape=[3, 3, 128, 256], dtype=tf.float32, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, regularizer=regularizer, initializer=initializer)

                # with tf.variable_scope('conv3'):

                #     regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                #     initializer = tf.contrib.layers.xavier_initializer()

                #     kernel = tf.get_variable('kernel', shape=[3, 3, 128, 256], dtype=tf.float32, regularizer=regularizer, initializer=initializer)
                #     biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, regularizer=regularizer, initializer=initializer)
                    
                # with tf.variable_scope('conv4'):

                #     regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                #     initializer = tf.contrib.layers.xavier_initializer()

                #     kernel = tf.get_variable('kernel', shape=[3, 3, 64, 128], dtype=tf.float32, regularizer=regularizer, initializer=initializer)
                #     biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, regularizer=regularizer, initializer=initializer)

                # with tf.variable_scope('conv5'):

                #     regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                #     initializer = tf.contrib.layers.xavier_initializer()

                #     kernel = tf.get_variable('kernel', shape=[3, 3, 128, 256], dtype=tf.float32, regularizer=regularizer, initializer=initializer)
                #     biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, regularizer=regularizer, initializer=initializer

            with tf.variable_scope('fc1'):

                regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                initializer = tf.contrib.layers.xavier_initializer()

                trainable = False if self.train_fc == 'NO' else True

                weight = tf.get_variable('weights', shape=[14*14*256, 256], dtype=tf.float32, initializer=initializer, regularizer=regularizer, trainable=trainable)
                biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=initializer, regularizer=regularizer, trainable=trainable)

            with tf.variable_scope('fc2'):

                regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                initializer = tf.contrib.layers.xavier_initializer()

                trainable = False if self.train_fc == 'NO' else True

                weight = tf.get_variable('weights', shape=[256, self.num_classes], dtype=tf.float32, initializer=initializer, regularizer=regularizer, trainable=trainable)
                biases = tf.get_variable('biases', shape=[self.num_classes], dtype=tf.float32, initializer=initializer, regularizer=regularizer, trainable=trainable)


    def get_variables(self):

        return self.featurizer_variables + self.main_model_variables + self.fc_variables

    
    def print_variables(self):

        var_list = self.get_variables()
        var_list_len = len(var_list)
        for var,idx in zip(var_list, range(var_list_len)):
            print(str(idx).zfill(3), var.name)


    def initialize_variables(self):

        logging.debug('--- begin variable initialization (vfeedbacknet) ---')
        var_list = self.get_variables()

        self.print_variables()
        print('Number of variables in model: {}'.format( len(var_list) ))
        
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

        self.vfeedbacknet_base.export_variables(self.sess, self.get_variables(), export_filename)
            

    def __call__(self, inputs, inputs_sequence_length):

        ModelLogger.log('raw_input', inputs)

        inputs = VFeedbackNetBase.split_video(inputs)
        ModelLogger.log('input', inputs)

        ## featurizer ##
        inputs = [ self.vfeedbacknet_base.vgg16_layer1(inp, var_list=self.featurizer_variables) for inp in inputs ]
        ModelLogger.log('vgg-layer1', inputs)

        ## main model ##
        inputs = [self.feedbackLSTMCell1.apply_layer([inp for _ in range(Model.NFEEDBACK_ITERATIONS)], var_list=self.main_model_variables) for inp in inputs]
        fb_sequence = [ [] for _ in range(Model.NFEEDBACK_ITERATIONS) ]
        for inp in inputs:
            for i in range(Model.NFEEDBACK_ITERATIONS):
                fb_sequence[i].append(inp[i])
        ModelLogger.log('feedbackCell1', fb_sequence[0])        

        logits = []
        for seq in fb_sequence:
            inputs = [ self.reshape_conv_layer(inp, 1, var_list=self.main_model_variables) for inp in seq ]
            ModelLogger.log('reshape_conv_layer1', inputs)

            inputs = self.convLSTM_layer1(inputs, inputs_sequence_length, var_list=self.main_model_variables)
            ModelLogger.log('convLSTM1', inputs)

            inputs = [ self.reshape_conv_layer(inp, 2, var_list=self.main_model_variables) for inp in inputs ]
            ModelLogger.log('reshape_conv_layer2', inputs)

            inputs = [ self.fc_layer(inp, 1, var_list=self.fc_variables) for inp in inputs ]
            ModelLogger.log('fc1', inputs)

            inputs = [ self.fc_layer(inp, 2, var_list=self.fc_variables) for inp in inputs ]
            ModelLogger.log('fc2', inputs)

            seq1 = tf.stack(inputs, axis=1)
            logits.append(seq1)

        logits = tf.stack(logits, axis=1)
        ModelLogger.log('combined-feedback-logits', logits) 
       
        return logits

        
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

                            
                inputs, state = tf.nn.dynamic_rnn(
                    self.convLSTMCell1,
                    tf.stack(inputs, axis=1),
                    dtype=tf.float32,
                    sequence_length=None,
                )

                inputs = tf.unstack(inputs, axis=1)
                
                return inputs

    def convLSTM_layer2(self, inputs, inputs_sequence_length, var_list=None):

        with tf.variable_scope('vfeedbacknet_{}'.format(Model.model_name)):
            with tf.variable_scope('convlstm2'):
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

                            
                inputs, state = tf.nn.dynamic_rnn(
                    self.convLSTMCell2,
                    tf.stack(inputs, axis=1),
                    dtype=tf.float32,
                    sequence_length=None,
                )

                inputs = tf.unstack(inputs, axis=1)
                
                return inputs

    def convLSTM_layer3(self, inputs, inputs_sequence_length, var_list=None):

        with tf.variable_scope('vfeedbacknet_{}'.format(Model.model_name)):
            with tf.variable_scope('convlstm3'):
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

                            
                inputs, state = tf.nn.dynamic_rnn(
                    self.convLSTMCell3,
                    tf.stack(inputs, axis=1),
                    dtype=tf.float32,
                    sequence_length=None,
                )

                inputs = tf.unstack(inputs, axis=1)
                
                return inputs

            
    def reshape_conv_layer(self, inputs, conv_num, var_list=None):

        with tf.variable_scope('vfeedbacknet_{}'.format(Model.model_name)):
            with tf.variable_scope('reshape_convs'):
                with tf.variable_scope('conv{}'.format(conv_num), reuse=True):
                    kernel = tf.get_variable('kernel')
                    biases = tf.get_variable('biases')
                    
                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)

                    inputs = tf.nn.relu(inputs)

                    inputs = tf.nn.max_pool(inputs,
                                            ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1],
                                            padding='VALID')

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                    return inputs
            
    def conv_layer(self, inputs, var_list=None):

        with tf.variable_scope('vfeedbacknet_{}'.format(Model.model_name)):
            with tf.variable_scope('process_featurizer_output'):
                with tf.variable_scope('conv1', reuse=True):
                    kernel = tf.get_variable('kernel')
                    biases = tf.get_variable('biases')
                    
                inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                inputs = tf.nn.bias_add(inputs, biases)

                inputs = tf.nn.relu(inputs)

                inputs = tf.nn.max_pool(inputs,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='VALID')

                if var_list is not None and kernel not in var_list:
                    var_list.append(kernel)
                if var_list is not None and biases not in var_list:
                    var_list.append(biases)

                return inputs

                
    def fc_layer(self, inputs, fc_num, var_list=None):
        
        with tf.variable_scope('vfeedbacknet_{}'.format(Model.model_name), reuse=True):
            with tf.variable_scope('fc{}'.format(fc_num)):
                
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')

                #h, w, c, = inputs.shape[1:]
                #size = int(h) * int(w) * int(c)
                size = 1
                for d in inputs.shape[1:]:
                    size *= int(d)
                inputs = tf.reshape(inputs, [-1, size])
                inputs = tf.matmul(inputs, weights)
                inputs = tf.nn.bias_add(inputs, biases)
                
                if var_list is not None and weights not in var_list:
                    var_list.append(weights)
                if var_list is not None and biases not in var_list:
                    var_list.append(biases)
                    
                return inputs
                
            
if __name__ == '__main__':

    sess = tf.Session()

    video_length = 20
    x = tf.placeholder(tf.float32, [None, video_length, 112, 112, 3], name='inputs')
    x_len = tf.placeholder(tf.float32, [None], name='inputs_len')
    zeros = tf.placeholder(tf.float32, [video_length], name='inputs_len')
    labels = tf.placeholder(tf.float32, [None], name='inputs_len')

    model = Model(sess, 27, 16, train_featurizer='NO', train_main_model='FROM_SCRATCH', train_fc='FROM_SCRATCH', weights_filename='/home/jemmons/data/vfeedbacknet_base_weights.npz')

    logits = model(x, x_len)
    ModelLogger.log('logits', logits)

    model.initialize_variables()
    model.export_variables('/tmp/weights.npz')
    #model.print_variables()
    
    # print out the model
    # graph = tf.get_default_graph()    
    # for op in graph.get_operations():
    #     print((op.name))

    
