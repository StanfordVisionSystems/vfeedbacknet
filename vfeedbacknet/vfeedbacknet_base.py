import numpy as np
import tensorflow as tf
import logging

from vfeedbacknet.vfeedbacknet_utilities import ModelLogger

class VFeedbackNetBase:

    
    def __init__(self, sess, num_classes,
                 train_vgg16='FROM_SCRATCH', train_feedback='FROM_SCRATCH', train_fc='FROM_SCRATCH',
                 weights_filename=None, is_training=True):

        self.sess = sess
        self.weights = np.load(weights_filename) if weights_filename is not None else None
        self.num_classes = num_classes

        assert train_vgg16 in ['NO', 'FINE_TUNE', 'FROM_SCRATCH'], 'train_vgg16 must be either: NO, FINE_TUNE, or FROM_SCRATCH'
        self.train_vgg16 = train_vgg16 if is_training else 'NO'

        assert train_feedback in ['NO', 'FINE_TUNE', 'FROM_SCRATCH'], 'train_feedback must be either: NO, FINE_TUNE, or FROM_SCRATCH'
        self.train_feedback = train_feedback if is_training else 'NO'

        assert train_fc in ['NO', 'FINE_TUNE', 'FROM_SCRATCH'], 'train_fc must be either: NO, FINE_TUNE, or FROM_SCRATCH'
        self.train_fc = train_fc if is_training else 'NO'

        self.is_training = is_training

        self.vgg_variables = []
        self.vfeedbacknet_feedback_variables = []
        self.vfeedbacknet_fc_variables = []
        
        self._declare_variables()
        
    def _declare_variables(self):
        '''
        Declare all the necessary variables so they can be referenced and reused
        during the model contruction. (helps to ensure correct variable sharing)
        '''

        with tf.variable_scope('vfeedbacknet_base'):
            with tf.variable_scope('vgg16'):

                regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                initializer = tf.contrib.layers.xavier_initializer()

                trainable = False if self.train_vgg16 == 'NO' else True

                with tf.variable_scope('conv1_1'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 3, 64], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[64], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv1_2'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[64], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv2_1'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv2_2'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 128, 128], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv3_1'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 128, 256], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv3_2'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 256, 256], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv3_3'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 256, 256], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv4_1'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 256, 512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv4_2'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv4_3'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv5_1'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv5_2'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)

                with tf.variable_scope('conv5_3'):
                    kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
                    biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable, regularizer=regularizer, initializer=initializer)
            
            with tf.variable_scope('feedback'):

                regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                initializer = tf.contrib.layers.xavier_initializer()
                
                trainable = False if self.train_feedback == 'NO' else True
                self.vfeedbacknet_feedback_variables += []
                
            with tf.variable_scope('fc'):

                regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                initializer = tf.contrib.layers.xavier_initializer()

                trainable = False if self.train_fc == 'NO' else True

                weight = tf.get_variable('weights', shape=[512, self.num_classes], dtype=tf.float32, initializer=initializer, regularizer=regularizer, trainable=trainable)
                biases = tf.get_variable('biases', shape=[self.num_classes], dtype=tf.float32, initializer=initializer, regularizer=regularizer, trainable=trainable)
                

    def get_variables(self):

        return self.vgg_variables + self.vfeedbacknet_feedback_variables + self.vfeedbacknet_fc_variables

    
    def print_variables(self):

        for var in self.get_variables():
            print(var.name)

    
    def initialize_variables(self):

        logging.debug('--- begin variable initialization (vfeedbacknet_base) ---')

        if self.train_vgg16 == 'FROM_SCRATCH':
            logging.debug('vgg16:FROM_SCRATCH; using random initialization')
            for var in self.vgg_variables:
                self.sess.run(var.initializer)
        else:
            for var in self.vgg_variables:
                logging.debug('loading {}: {}'.format(var.name, var.shape))
                self.sess.run(var.assign(self.weights[var.name]))

        if self.train_feedback  == 'FROM_SCRATCH':
            logging.debug('feedback: FROM_SCRATCH; using random initialization')
            for var in self.vfeedbacknet_feedback_variables:
                self.sess.run(var.initializer)
        else:
            for var in self.vfeedbacknet_feedback_variables:
                logging.debug(' loading {}: {}'.format(var.name, var.shape))
                self.sess.run(var.assign(self.weights[var.name]))

        if self.train_fc == 'FROM_SCRATCH':
            logging.debug('fc: FROM_SCRATCH; using random initialization')
            for var in self.vfeedbacknet_fc_variables:
                self.sess.run(var.initializer)
        else:
            for var in self.vfeedbacknet_fc_variables:
                logging.debug('loading {}: {}'.format(var.name, var.shape))
                self.sess.run(var.assign(self.weights[var.name]))

        logging.debug('--- end variable initialization (vfeedbacknet_base) ---')


    @staticmethod
    def export_variables(sess, variables, export_filename):

        assert export_filename[-4:] == '.npz', 'export_filename must have the .npz filename extension'
        
        d = { var.name : sess.run(var) for var in variables }
        np.savez(export_filename, **d)

    def split_video(self, inputs):

        with tf.variable_scope('vfeedbacknet_base', reuse=True):

            inputs = tf.expand_dims(inputs, axis=4)
            ModelLogger.log('preprocess1', inputs)
            
            inputs = tf.unstack(inputs, axis=1)
            ModelLogger.log('preprocess2', inputs)

            inputs = [ tf.tile(inp, [1, 1, 1, 3]) for inp in inputs ]
            ModelLogger.log('preprocess3', inputs)
            return inputs

    @staticmethod
    def split_video(inputs):

        with tf.variable_scope('vfeedbacknet_base', reuse=True):

            inputs = tf.expand_dims(inputs, axis=4)
            ModelLogger.log('preprocess1', inputs)
            
            inputs = tf.unstack(inputs, axis=1)
            ModelLogger.log('preprocess2', inputs)

            inputs = [ tf.tile(inp, [1, 1, 1, 3]) for inp in inputs ]
            ModelLogger.log('preprocess3', inputs)
            return inputs

    def vgg16_layer1(self, inputs, var_list=None):
        
        with tf.variable_scope('vfeedbacknet_base', reuse=True):
            with tf.variable_scope('vgg16'):

                # conv1_1
                with tf.variable_scope('conv1_1'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # conv1_2
                with tf.variable_scope('conv1_2'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # pool1
                inputs = tf.nn.max_pool(inputs,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool1')

                return inputs

        
    def vgg16_layer2(self, inputs, var_list=None):

        with tf.variable_scope('vfeedbacknet_base', reuse=True):
            with tf.variable_scope('vgg16'):

                # conv2_1
                with tf.variable_scope('conv2_1'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # conv2_2
                with tf.variable_scope('conv2_2'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # pool2
                inputs = tf.nn.max_pool(inputs,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool2')

                return inputs

        
    def vgg16_layer3(self, inputs, var_list=None):

        with tf.variable_scope('vfeedbacknet_base', reuse=True):
            with tf.variable_scope('vgg16'):

                # conv3_1
                with tf.variable_scope('conv3_1'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # conv3_2
                with tf.variable_scope('conv3_2'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # conv3_3
                with tf.variable_scope('conv3_3'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # pool3
                inputs = tf.nn.max_pool(inputs,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool3')
                return inputs

        
    def vgg16_layer4(self, inputs, var_list=None): 

        with tf.variable_scope('vfeedbacknet_base', reuse=True):
            with tf.variable_scope('vgg16'):

                # conv4_1
                with tf.variable_scope('conv4_1'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # conv4_2
                with tf.variable_scope('conv4_2'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # conv4_3
                with tf.variable_scope('conv4_3'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)
                        
                # pool4
                inputs = tf.nn.max_pool(inputs,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool4')

                return inputs

        
    def vgg16_layer5(self, inputs, var_list=None):

        with tf.variable_scope('vfeedbacknet_base', reuse=True):
            with tf.variable_scope('vgg16'):

                # conv5_1
                with tf.variable_scope('conv5_1'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # conv5_2
                with tf.variable_scope('conv5_2'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)


                # conv5_3
                with tf.variable_scope('conv5_3'):
                    kernel = tf.get_variable('weights')
                    biases = tf.get_variable('biases')

                    inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                    inputs = tf.nn.bias_add(inputs, biases)
                    inputs = tf.nn.relu(inputs)

                    if kernel not in self.vgg_variables:
                        self.vgg_variables += [kernel]
                    if biases not in self.vgg_variables:
                        self.vgg_variables += [biases]

                    if var_list is not None and kernel not in var_list:
                        var_list.append(kernel)
                    if var_list is not None and biases not in var_list:
                        var_list.append(biases)

                # pool5
                inputs = tf.nn.max_pool(inputs,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool4')

                return inputs


    def ave_pool(self, inputs):

        with tf.variable_scope('vfeedbacknet_base', reuse=True):
            h, w = int(inputs.shape[1]), int(inputs.shape[2])
            assert h == w, 'Expected the height and width of the input to be equal. Got {}x{} instead. shape={}'.format(h,w,inputs.shape)
            
            inputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=h, strides=1, padding='VALID',
                data_format='channels_last', name='ave_pool')
            
            return inputs

            
    def fc_layer(self, inputs, var_list=None):
        
        with tf.variable_scope('vfeedbacknet_base', reuse=True):
            with tf.variable_scope('fc'):
                
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')

                h, w, c, = inputs.shape[1:]
                size = int(h) * int(w) * int(c)
                inputs = tf.reshape(inputs, [-1, size])
                inputs = tf.matmul(inputs, weights)
                inputs = tf.nn.bias_add(inputs, biases)
                
                if weights not in self.vfeedbacknet_fc_variables:
                    self.vfeedbacknet_fc_variables += [weights]
                if biases not in self.vfeedbacknet_fc_variables:
                    self.vfeedbacknet_fc_variables += [biases]

                if var_list is not None and weights not in var_list:
                    var_list.append(weights)
                if var_list is not None and biases not in var_list:
                    var_list.append(biases)
                    
                return inputs
                 
if __name__ == '__main__':
    sess = tf.Session()
    x = tf.placeholder(tf.float32, [None, 20, 112, 112], name='inputs')
    x_len = tf.placeholder(tf.float32, [None], name='inputs_len')
    zeros = tf.placeholder(tf.float32, [20], name='inputs_len')
    labels = tf.placeholder(tf.float32, [None], name='inputs_len')

    vfeedbacknet_base = VFeedbackNetBase(sess, 27, train_vgg16='FROM_SCRATCH')

    ModelLogger.log('input', x)
    
    inputs = vfeedbacknet_base.split_video(x)
    ModelLogger.log('split', inputs)

    variables = []
    
    inputs = [ vfeedbacknet_base.vgg16_layer1(inp, var_list=variables) for inp in inputs ]
    ModelLogger.log('vgg-layer', inputs)

    inputs = [ vfeedbacknet_base.vgg16_layer2(inp, var_list=variables) for inp in inputs ]
    ModelLogger.log('vgg-layer', inputs)

    inputs = [ vfeedbacknet_base.vgg16_layer3(inp, var_list=variables) for inp in inputs ]
    ModelLogger.log('vgg-layer', inputs)

    inputs = [ vfeedbacknet_base.vgg16_layer4(inp, var_list=variables) for inp in inputs ]
    ModelLogger.log('vgg-layer', inputs)

    inputs = [ vfeedbacknet_base.vgg16_layer5(inp, var_list=variables) for inp in inputs ]
    ModelLogger.log('vgg-layer', inputs)

    inputs = [ vfeedbacknet_base.ave_pool(inp) for inp in inputs ]
    ModelLogger.log('ave_pool', inputs)

    logits = [ vfeedbacknet_base.fc_layer(inp, var_list=variables) for inp in inputs ]
    ModelLogger.log('logits', logits)

    print('len(self.vgg_variables) =', len(vfeedbacknet_base.vgg_variables))
    print('len(self.vfeedbacknet_feedback_variables) =', len(vfeedbacknet_base.vfeedbacknet_feedback_variables))
    print('len(self.vfeedbacknet_fc_variables) =', len(vfeedbacknet_base.vfeedbacknet_fc_variables))
    print('len(var_list) =', len(variables))
    
    vfeedbacknet_base.initialize_variables()
    vfeedbacknet_base.print_variables()
    print('num variables:', len(vfeedbacknet_base.get_variables()))

    VFeedbackNetBase.export_variables(sess, vfeedbacknet_base.get_variables(), '/tmp/weights.npz')

    
    # sess = tf.Session()
    # x = tf.placeholder(tf.float32, [None, 20, 112, 112], name='inputs')
    # x_len = tf.placeholder(tf.float32, [None], name='inputs_len')
    # zeros = tf.placeholder(tf.float32, [20], name='inputs_len')
    # labels = tf.placeholder(tf.float32, [None], name='inputs_len')

    # new_vfeedbacknet_base = VFeedbackNetBase(sess, 27, train_vgg16='NO', train_fc='NO', weights='/tmp/weights.npz')
    
    # inputs = new_vfeedbacknet_base.split_video(x)
    # ModelLogger.log('split', inputs)
    
    # inputs = [ new_vfeedbacknet_base.vgg16_layer1(inp) for inp in inputs ]
    # ModelLogger.log('vgg-layer', inputs)

    # inputs = [ new_vfeedbacknet_base.vgg16_layer2(inp) for inp in inputs ]
    # ModelLogger.log('vgg-layer', inputs)

    # inputs = [ new_vfeedbacknet_base.vgg16_layer3(inp) for inp in inputs ]
    # ModelLogger.log('vgg-layer', inputs)

    # inputs = [ new_vfeedbacknet_base.vgg16_layer4(inp) for inp in inputs ]
    # ModelLogger.log('vgg-layer', inputs)

    # inputs = [ new_vfeedbacknet_base.vgg16_layer5(inp) for inp in inputs ]
    # ModelLogger.log('vgg-layer', inputs)

    # inputs = [ new_vfeedbacknet_base.ave_pool(inp) for inp in inputs ]
    # ModelLogger.log('ave_pool', inputs)

    # logits = [ new_vfeedbacknet_base.fc_layer(inp) for inp in inputs ]
    # ModelLogger.log('logits', logits)

    # new_vfeedbacknet_base.initialize_variables()
    # VFeedbackNetBase.export_variables(sess, new_vfeedbacknet_base.get_variables(), '/tmp/weights1.npz')
    
    # print out the model
    # graph = tf.get_default_graph()    
    # for op in graph.get_operations():
    #     print((op.name))
