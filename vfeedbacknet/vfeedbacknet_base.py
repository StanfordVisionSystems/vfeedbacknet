import vfeedbacknet.convLSTM as convLSTM

import tensorflow as tf
import logging

from vfeedbacknet.vfeedbacknet_utilities import ModelLogger

class VFeedbackNetBase:
    
    def __init__(self, sess, num_classes,
                 vgg16_weights=None, feedback_weights=None, fc_weights=None,
                 train_vgg16='NO', train_feedback='FROM_SCRATCH', train_fc='FROM_SCRATCH',
                 is_training=True):

        self.sess = sess
        self.vgg16_weights = vgg16_weights
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
            
        with tf.variable_scope('vfeedbacknet'):
            with tf.variable_scope('feedback'):

                regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                initializer = tf.contrib.layers.xavier_initializer()
                
                trainable = False if self.train_feedback == 'NO' else True


                #self.vfeedbacknet_variables += []
                
            with tf.variable_scope('fc'):

                regularizer = None # tf.contrib.layers.l2_regularizer(scale=0.25)
                initializer = tf.contrib.layers.xavier_initializer()

                trainable = False if self.train_fc == 'NO' else True

                weight = tf.get_variable('weights', shape=[512, self.num_classes], dtype=tf.float32, initializer=initializer, regularizer=regularizer, trainable=trainable)
                biases = tf.get_variable('biases', shape=[self.num_classes], dtype=tf.float32, initializer=initializer, regularizer=regularizer, trainable=trainable)
                

    def initialize_variables(self):

        with tf.variable_scope('NoFeedBackNetVgg16', reuse=True):

            logging.debug('--- begin variable initialization ---')

            if self.train_vgg16 == 'FROM_SCRATCH':
                logging.debug('vgg16:FROM_SCRATCH; using random initialization')
                for var in self.vgg_variables:
                    self.sess.run(var.initializer)
            else:
                pass
                    
            if self.train_feedback  == 'FROM_SCRATCH':
                logging.debug('feedback: FROM_SCRATCH; using random initialization')
                for var in self.vfeedbacknet_feedback_variables:
                    self.sess.run(var.initializer)
            else:
                pass

            if self.train_fc == 'FROM_SCRATCH':
                logging.debug('fc: FROM_SCRATCH; using random initialization')
                for var in self.vfeedbacknet_fc_variables:
                    self.sess.run(var.initializer)
            else:
                pass

            logging.debug('--- end variable initialization ---')

    def export_variables(self, export_file):

        with tf.variable_scope('NoFeedBackNetVgg16', reuse=True):

            logging.debug('--- begin variable initialization ---')

            if self.train_vgg16 == 'FROM_SCRATCH':
                logging.debug('vgg16:FROM_SCRATCH; using random initialization')
                for var in self.vgg_variables:
                    self.sess.run(var.initializer)
            else:
                pass
                    
            if self.train_feedback  == 'FROM_SCRATCH':
                logging.debug('feedback: FROM_SCRATCH; using random initialization')
                for var in self.vfeedbacknet_feedback_variables:
                    self.sess.run(var.initializer)
            else:
                pass

            if self.train_fc == 'FROM_SCRATCH':
                logging.debug('fc: FROM_SCRATCH; using random initialization')
                for var in self.vfeedbacknet_fc_variables:
                    self.sess.run(var.initializer)
            else:
                pass

            logging.debug('--- end variable initialization ---')
            

    def split_video(self, inputs):

        inputs = tf.expand_dims(inputs, axis=4)
        inputs = tf.unstack(inputs, axis=1)
        inputs = [ tf.tile(inp, [1, 1, 1, 3]) for inp in inputs ]
        return inputs
            
    def vgg16_layer1(self, inputs):
        
        with tf.variable_scope('vgg16', reuse=True):
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

            # pool1
            inputs = tf.nn.max_pool(inputs,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

            return inputs

        
    def vgg16_layer2(self, inputs):

        with tf.variable_scope('vgg16', reuse=True):
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

            # pool2
            inputs = tf.nn.max_pool(inputs,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

            return inputs

        
    def vgg16_layer3(self, inputs):

        with tf.variable_scope('vgg16', reuse=True):
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

            # pool3
            inputs = tf.nn.max_pool(inputs,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')
            return inputs

        
    def vgg16_layer4(self, inputs): 

        with tf.variable_scope('vgg16', reuse=True):
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

            # pool4
            inputs = tf.nn.max_pool(inputs,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

            return inputs

        
    def vgg16_layer5(self, inputs):

        with tf.variable_scope('vgg16', reuse=True):
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

            # conv5_2
            with tf.variable_scope('conv5_2'):
                kernel = tf.get_variable('weights')
                biases = tf.get_variable('biases')

                inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
                inputs = tf.nn.bias_add(inputs, biases)

                if kernel not in self.vgg_variables:
                    self.vgg_variables += [kernel]
                if biases not in self.vgg_variables:
                    self.vgg_variables += [biases]
                inputs = tf.nn.relu(inputs)

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

            # pool5
            inputs = tf.nn.max_pool(inputs,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

            return inputs


    def ave_pool(self, inputs):

        with tf.variable_scope('vfeedbacknet', reuse=True):
            h, w = int(inputs.shape[1]), int(inputs.shape[2])
            assert h == w, 'Expected the height and width of the input to be equal. Got {}x{} instead. shape={}'.format(h,w,inputs.shape)
            
            inputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=h, strides=1, padding='VALID',
                data_format='channels_last', name='ave_pool')
            
            return inputs

            
    def fc_layer(self, inputs):
        
        with tf.variable_scope('vfeedbacknet', reuse=True):
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

                return inputs
                 
if __name__ == '__main__':
    sess = tf.Session()
    
    x = tf.placeholder(tf.float32, [None, 20, 112, 112], name='inputs')
    x_len = tf.placeholder(tf.float32, [None], name='inputs_len')
    zeros = tf.placeholder(tf.float32, [20], name='inputs_len')
    labels = tf.placeholder(tf.float32, [None], name='inputs_len')

    vfeedbacknet_base = VFeedbackNetBase(sess, 27)

    ModelLogger.log('input', x)
    
    inputs = vfeedbacknet_base.split_video(x)
    ModelLogger.log('split', inputs)
    
    inputs = [ vfeedbacknet_base.vgg16_layer1(inp) for inp in inputs ]
    ModelLogger.log('vgg-layer', inputs)

    inputs = [ vfeedbacknet_base.vgg16_layer2(inp) for inp in inputs ]
    ModelLogger.log('vgg-layer', inputs)

    inputs = [ vfeedbacknet_base.vgg16_layer3(inp) for inp in inputs ]
    ModelLogger.log('vgg-layer', inputs)

    inputs = [ vfeedbacknet_base.vgg16_layer4(inp) for inp in inputs ]
    ModelLogger.log('vgg-layer', inputs)

    inputs = [ vfeedbacknet_base.vgg16_layer5(inp) for inp in inputs ]
    ModelLogger.log('vgg-layer', inputs)

    inputs = [ vfeedbacknet_base.ave_pool(inp) for inp in inputs ]
    ModelLogger.log('ave_pool', inputs)

    logits = [ vfeedbacknet_base.fc_layer(inp) for inp in inputs ]
    ModelLogger.log('logits', logits)

    print('len(self.vgg_variables) =', len(vfeedbacknet_base.vgg_variables))
    print('len(self.vfeedbacknet_feedback_variables) =', len(vfeedbacknet_base.vfeedbacknet_feedback_variables))
    print('len(self.vfeedbacknet_fc_variables) =', len(vfeedbacknet_base.vfeedbacknet_fc_variables))
    
    vfeedbacknet_base.initialize_variables()

    # print out the model
    # graph = tf.get_default_graph()    
    # for op in graph.get_operations():
    #     print((op.name))
