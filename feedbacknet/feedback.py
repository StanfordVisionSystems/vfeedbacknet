#!/usr/bin/env python

import keras

import numpy as np
import tensorflow as tf

def tensorflow_model():

    ###################################################################################################################
    # begin describing the model
    ###################################################################################################################

    # TODO: recreate the model to be feedback
    x_image = tf.placeholder(tf.float32, [None, 28, 28, 1])
    
    # 5x5x1 conv
    W_conv1 = tf.Variable( tf.truncated_normal([5, 5, 1, 32], stddev=0.1) )
    b_conv1 = tf.Variable( tf.truncated_normal([32], stddev=0.1) )
    h_conv1 = tf.nn.relu( tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)

    # 2x2 max pool
    h_pool1 = tf.nn.max_pool(h_conv1, 
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], 
                             padding='SAME')

    # 5x5x32 conv
    W_conv2 = tf.Variable( tf.truncated_normal([5, 5, 32, 64], stddev=0.1) )
    b_conv2 = tf.Variable( tf.truncated_normal([64], stddev=0.1) )
    h_conv2 = tf.nn.relu( tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)

    # 2x2 max pool
    h_pool2 = tf.nn.max_pool(h_conv2, 
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], 
                             padding='SAME')
    
    # fc layer
    W_fc1 = tf.Variable( tf.truncated_normal([7*7*64, 1024], stddev=0.1) )
    b_fc1 = tf.Variable( tf.truncated_normal([1024], stddev=0.1) )

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # fc
    W_fc2 = tf.Variable( tf.truncated_normal([1024, 10], stddev=0.1) )
    b_fc2 = tf.Variable( tf.truncated_normal([10], stddev=0.1) )

    # final output
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ###################################################################################################################
    # end describing the model
    ###################################################################################################################

    # prepare training
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # load the data
    import mnist_loaddata
    images, labels = mnist_loaddata.read_mnist('/root/mnist/train-images-idx3-ubyte', '/root/mnist/train-labels-idx1-ubyte')

    images = images.reshape(images.shape[-1], 28, 28, 1)
    images = images.astype(np.float32)
    images /= 255
    one_hot_labels = keras.utils.to_categorical(labels, 10)
    
    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())

         writer = tf.summary.FileWriter("/tmp/", sess.graph)
         writer.add_graph(sess.graph)

         # train the model
         count = 0
         for i in range(0,len(images), 32):
             x_inp = images[i:i+32]
             y_inp = one_hot_labels[i:i+32]

             if count % 100 == 0:
                 train_accuracy = accuracy.eval(feed_dict={
                     x_image: x_inp, y_: y_inp, keep_prob: 1.0})
                 print('step %d, training accuracy %g' % (count, train_accuracy))
             count += 1

             train_step.run(feed_dict={x_image: x_inp, y_: y_inp, keep_prob: 0.5})

         # get the accruacy on the test set
         images, labels = mnist_loaddata.read_mnist('/root/mnist/t10k-images-idx3-ubyte', '/root/mnist/t10k-labels-idx1-ubyte')
             
         images = images.reshape(images.shape[-1], 28, 28, 1)
         images = images.astype(np.float32)
         images /= 255
         one_hot_labels = keras.utils.to_categorical(labels, 10)
        
         print('test accuracy %g' % accuracy.eval(feed_dict={
             x_image: images, y_: one_hot_labels, keep_prob: 1.0}))

if(__name__ == '__main__'):
    tensorflow_model()
