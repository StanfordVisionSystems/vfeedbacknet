########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave

class VGG16:
    def __init__(self, sess=None, weights=None, trainable=False):
        self.reuse = False
        self.weights = weights
        self.sess = sess
        self.trainable = trainable
        
        self.parameters = []
        with tf.variable_scope('vgg'):
            with tf.variable_scope('conv1_1'):
                kernel = tf.get_variable('weights', shape=[3, 3, 3, 64], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[64], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv1_2'):
                kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[64], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv2_1'):
                kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv2_2'):
                kernel = tf.get_variable('weights', shape=[3, 3, 128, 128], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv3_1'):
                kernel = tf.get_variable('weights', shape=[3, 3, 128, 256], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv3_2'):
                kernel = tf.get_variable('weights', shape=[3, 3, 256, 256], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv3_3'):
                kernel = tf.get_variable('weights', shape=[3, 3, 256, 256], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv4_1'):
                kernel = tf.get_variable('weights', shape=[3, 3, 256, 512], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv4_2'):
                kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv4_3'):
                kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv5_1'):
                kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv5_2'):
                kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

            with tf.variable_scope('conv5_3'):
                kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], dtype=tf.float32, trainable=trainable)
                biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, trainable=trainable)
                self.parameters += [kernel, biases]

        # if weights is not None or sess is not None:
        #     assert weights is not None and sess is not None, 'Both weights and sess must be set!'
        #     self.load_weights()
                
    def __call__(self, frame):

        # for black and white input (Y-component of YUV) 
        with tf.variable_scope('vgg', reuse=True):
            with tf.name_scope('preprocess'):
                frame = tf.tile(frame, [1, 1, 1, 3]) # expand the channels to 3
                img_mean = [123.68, 123.68, 123.68]
                
                #img_mean = [123.68, 116.779, 103.939]
                mean = tf.constant(img_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
                frame = frame - mean
                
            output = self.convlayers(frame, self.trainable)
            return output
                
    def convlayers(self, inputs, trainable):
        
        # conv1_1
        with tf.variable_scope('conv1_1'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # conv1_2
        with tf.variable_scope('conv1_2'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # pool1
        inputs = tf.nn.max_pool(inputs,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool1')

        # # conv2_1
        with tf.variable_scope('conv2_1'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # conv2_2
        with tf.variable_scope('conv2_2'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # pool2
        inputs = tf.nn.max_pool(inputs,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # conv3_2
        with tf.variable_scope('conv3_2'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # conv3_3
        with tf.variable_scope('conv3_3'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # pool3
        inputs = tf.nn.max_pool(inputs,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # conv4_2
        with tf.variable_scope('conv4_2'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # conv4_3
        with tf.variable_scope('conv4_3'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # pool4
        inputs = tf.nn.max_pool(inputs,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # conv5_2
        with tf.variable_scope('conv5_2'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # conv5_3
        with tf.variable_scope('conv5_3'):
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')

            inputs = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            inputs = tf.nn.bias_add(inputs, biases)
            inputs = tf.nn.relu(inputs)

        # pool5
        inputs = tf.nn.max_pool(inputs,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool4')

        return inputs
        
    def load_weights(self):
        '''
        load all the weight up to and including the stopping layer
        '''

        raw_weights = np.load(self.weights)
        keys = sorted(raw_weights.keys())
        for i, k in enumerate(keys):
            print('VGG16:', (i, k, np.shape(raw_weights[k])))
            self.sess.run(self.parameters[i].assign(raw_weights[k]))

            if k == 'conv5_3_b':
                # only load the conv layers
                break
