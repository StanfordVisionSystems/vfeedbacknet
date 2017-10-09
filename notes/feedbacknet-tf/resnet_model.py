# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')
class ConvLSTM_12(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    # Iteration numbers
    iter1 = 4
    iter2 = 4
    second_order_iter = 1 
    inputs = self._images
    print('Second Order Model with: %d' % (second_order_iter))
    scope = ''
    # Core Model
    with tf.op_scope([inputs], scope, 'inception_v3'):
      # Initial Conv
      conv0 = self._conv2D(inputs, 3, 3, 16, 1, scope='conv0')
      norm1 = conv0

      for i in xrange(second_order_iter):

        h1_tm1 = []
        # ConvLSTM block1
        with tf.variable_scope('ConvLSTM_1') as scope:
          for i1 in xrange(iter1):
            # LSTM update
            if i1 > 0 or i > 0:
              conv_f11 = self._conv2D(norm1,  3,16,16,1, scope='conv_f11', reuse=scope)
              conv_i11 = self._conv2D(norm1,  3,16,16,1, scope='conv_i11', reuse=scope)
              conv_c11 = self._conv2D(norm1,  3,16,16,1, scope='conv_c11', reuse=scope)
              conv_o11 = self._conv2D(norm1,  3,16,16,1, scope='conv_o11', reuse=scope)

              conv_f12 = self._conv2D(conv_f11, 3,16,16,1, scope='conv_f12', reuse=scope)
              conv_i12 = self._conv2D(conv_i11, 3,16,16,1, scope='conv_i12', reuse=scope)
              conv_c12 = self._conv2D(conv_c11, 3,16,16,1, scope='conv_c12', reuse=scope)
              conv_o12 = self._conv2D(conv_o11, 3,16,16,1, scope='conv_o12', reuse=scope)

              conv_f13 = self._conv2D(conv_f12, 3,16,16,1, scope='conv_f13', reuse=scope, activation=None)
              conv_i13 = self._conv2D(conv_i12, 3,16,16,1, scope='conv_i13', reuse=scope, activation=None)
              conv_c13 = self._conv2D(conv_c12, 3,16,16,1, scope='conv_c13', reuse=scope, activation=None)
              conv_o13 = self._conv2D(conv_o12, 3,16,16,1, scope='conv_o13', reuse=scope, activation=None)
            
              if i1 > 1 or (i1 == 1 and i > 0):
                  conv_hf11 = self._conv2D(h1_tm1[i1-1], 3,16,16,1, scope='conv_hf11', reuse=scope)
                  conv_hi11 = self._conv2D(h1_tm1[i1-1], 3,16,16,1, scope='conv_hi11', reuse=scope)
                  conv_hc11 = self._conv2D(h1_tm1[i1-1], 3,16,16,1, scope='conv_hc11', reuse=scope)
                  conv_ho11 = self._conv2D(h1_tm1[i1-1], 3,16,16,1, scope='conv_ho11', reuse=scope)

                  conv_hf12 = self._conv2D(conv_hf11, 3,16,16,1, scope='conv_hf12', reuse=scope)
                  conv_hi12 = self._conv2D(conv_hi11, 3,16,16,1, scope='conv_hi12', reuse=scope)
                  conv_hc12 = self._conv2D(conv_hc11, 3,16,16,1, scope='conv_hc12', reuse=scope)
                  conv_ho12 = self._conv2D(conv_ho11, 3,16,16,1, scope='conv_ho12', reuse=scope)

                  conv_hf13 = self._conv2D(conv_hf12, 3,16,16,1, scope='conv_hf13', reuse=scope, activation=None)
                  conv_hi13 = self._conv2D(conv_hi12, 3,16,16,1, scope='conv_hi13', reuse=scope, activation=None)
                  conv_hc13 = self._conv2D(conv_hc12, 3,16,16,1, scope='conv_hc13', reuse=scope, activation=None)
                  conv_ho13 = self._conv2D(conv_ho12, 3,16,16,1, scope='conv_ho13', reuse=scope, activation=None)                
              elif i1 > 0:
                  conv_hf11 = self._conv2D(h1_tm1[i1-1],    3,16,16,1, scope='conv_hf11', reuse=None)
                  conv_hi11 = self._conv2D(h1_tm1[i1-1],    3,16,16,1, scope='conv_hi11', reuse=None)
                  conv_hc11 = self._conv2D(h1_tm1[i1-1],    3,16,16,1, scope='conv_hc11', reuse=None)
                  conv_ho11 = self._conv2D(h1_tm1[i1-1],    3,16,16,1, scope='conv_ho11', reuse=None)

                  conv_hf12 = self._conv2D(conv_hf11, 3,16,16,1, scope='conv_hf12', reuse=None)
                  conv_hi12 = self._conv2D(conv_hi11, 3,16,16,1, scope='conv_hi12', reuse=None)
                  conv_hc12 = self._conv2D(conv_hc11, 3,16,16,1, scope='conv_hc12', reuse=None)
                  conv_ho12 = self._conv2D(conv_ho11, 3,16,16,1, scope='conv_ho12', reuse=None)
                  
                  conv_hf13 = self._conv2D(conv_hf12, 3,16,16,1, scope='conv_hf13', reuse=None, activation=None)
                  conv_hi13 = self._conv2D(conv_hi12, 3,16,16,1, scope='conv_hi13', reuse=None, activation=None)
                  conv_hc13 = self._conv2D(conv_hc12, 3,16,16,1, scope='conv_hc13', reuse=None, activation=None)
                  conv_ho13 = self._conv2D(conv_ho12, 3,16,16,1, scope='conv_ho13', reuse=None, activation=None)

              if i1 == 0:
                conv_ft1 = tf.nn.sigmoid(conv_f13, name=scope.name)
                conv_it1 = tf.nn.sigmoid(conv_i13, name=scope.name)
                conv_ct1 = tf.nn.tanh(conv_c13,    name=scope.name)
                conv_ot1 = tf.nn.sigmoid(conv_o13, name=scope.name)
                state1 = conv_it1 * conv_ct1
              else:
                conv_ft1 = tf.nn.sigmoid(tf.add(conv_hf13, conv_f13), name=scope.name)
                conv_it1 = tf.nn.sigmoid(tf.add(conv_hi13, conv_i13), name=scope.name)
                conv_ct1 = tf.nn.tanh(tf.add(conv_hc13, conv_c13),    name=scope.name)
                conv_ot1 = tf.nn.sigmoid(tf.add(conv_ho13, conv_o13), name=scope.name)
                state1 = conv_ft1 * c1_tm1 + conv_it1 * conv_ct1
              recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 1 is the xt
              conv_f11 = self._conv2D(norm1,    3,16,16,1, scope='conv_f11')
              conv_i11 = self._conv2D(norm1,    3,16,16,1, scope='conv_i11')
              conv_c11 = self._conv2D(norm1,    3,16,16,1, scope='conv_c11')
              conv_o11 = self._conv2D(norm1,    3,16,16,1, scope='conv_o11')

              conv_f12 = self._conv2D(conv_f11, 3,16,16,1, scope='conv_f12')
              conv_i12 = self._conv2D(conv_i11, 3,16,16,1, scope='conv_i12')
              conv_c12 = self._conv2D(conv_c11, 3,16,16,1, scope='conv_c12')
              conv_o12 = self._conv2D(conv_o11, 3,16,16,1, scope='conv_o12')

              conv_f13 = self._conv2D(conv_f12, 3,16,16,1, scope='conv_f13', activation=None)
              conv_i13 = self._conv2D(conv_i12, 3,16,16,1, scope='conv_i13', activation=None)
              conv_c13 = self._conv2D(conv_c12, 3,16,16,1, scope='conv_c13', activation=None)
              conv_o13 = self._conv2D(conv_o12, 3,16,16,1, scope='conv_o13', activation=None)

              conv_ft1 = tf.nn.sigmoid(conv_f13, name=scope.name)
              conv_it1 = tf.nn.sigmoid(conv_i13, name=scope.name)
              conv_ct1 = tf.nn.tanh(conv_c13,    name=scope.name)
              conv_ot1 = tf.nn.sigmoid(conv_o13, name=scope.name)
              state1 = conv_it1 * conv_ct1
              recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)
                  
            h1_tm1.append(recurrent_hidden1)
            c1_tm1 = state1

        # Connect to the next ConvLSTM block
        norm2 = h1_tm1
        
        # ConvLSTM block2
        h2_tm1 = []
        with tf.variable_scope('ConvLSTM_2') as scope:
          for i2 in xrange(iter2):
            # LSTM update
            if i2 > 0 or i > 0:
              conv_f21 = self._conv2D(norm2[i2],    3,16,32,2, scope='conv_f21', reuse=scope)
              conv_i21 = self._conv2D(norm2[i2],    3,16,32,2, scope='conv_i21', reuse=scope)
              conv_c21 = self._conv2D(norm2[i2],    3,16,32,2, scope='conv_c21', reuse=scope)
              conv_o21 = self._conv2D(norm2[i2],    3,16,32,2, scope='conv_o21', reuse=scope)

              conv_f22 = self._conv2D(conv_f21, 3,32,32,1, scope='conv_f22', reuse=scope)
              conv_i22 = self._conv2D(conv_i21, 3,32,32,1, scope='conv_i22', reuse=scope)
              conv_c22 = self._conv2D(conv_c21, 3,32,32,1, scope='conv_c22', reuse=scope)
              conv_o22 = self._conv2D(conv_o21, 3,32,32,1, scope='conv_o22', reuse=scope)

              conv_f23 = self._conv2D(conv_f22, 3,32,32,1, scope='conv_f23', reuse=scope, activation=None)
              conv_i23 = self._conv2D(conv_i22, 3,32,32,1, scope='conv_i23', reuse=scope, activation=None)
              conv_c23 = self._conv2D(conv_c22, 3,32,32,1, scope='conv_c23', reuse=scope, activation=None)
              conv_o23 = self._conv2D(conv_o22, 3,32,32,1, scope='conv_o23', reuse=scope, activation=None) 

              if i2 > 1 or (i2 == 1 and i > 0):
                  conv_hf21 = self._conv2D(h2_tm1[i2-1],    3,32,32,1, scope='conv_hf21', reuse=scope)
                  conv_hi21 = self._conv2D(h2_tm1[i2-1],    3,32,32,1, scope='conv_hi21', reuse=scope)
                  conv_hc21 = self._conv2D(h2_tm1[i2-1],    3,32,32,1, scope='conv_hc21', reuse=scope)
                  conv_ho21 = self._conv2D(h2_tm1[i2-1],    3,32,32,1, scope='conv_ho21', reuse=scope)

                  conv_hf22 = self._conv2D(conv_hf21, 3,32,32,1, scope='conv_hf22', reuse=scope)
                  conv_hi22 = self._conv2D(conv_hi21, 3,32,32,1, scope='conv_hi22', reuse=scope)
                  conv_hc22 = self._conv2D(conv_hc21, 3,32,32,1, scope='conv_hc22', reuse=scope)
                  conv_ho22 = self._conv2D(conv_ho21, 3,32,32,1, scope='conv_ho22', reuse=scope)

                  conv_hf23 = self._conv2D(conv_hf22, 3,32,32,1, scope='conv_hf23', reuse=scope, activation=None)
                  conv_hi23 = self._conv2D(conv_hi22, 3,32,32,1, scope='conv_hi23', reuse=scope, activation=None)
                  conv_hc23 = self._conv2D(conv_hc22, 3,32,32,1, scope='conv_hc23', reuse=scope, activation=None)
                  conv_ho23 = self._conv2D(conv_ho22, 3,32,32,1, scope='conv_ho23', reuse=scope, activation=None)                
              elif i2 > 0:
                  conv_hf21 = self._conv2D(h2_tm1[i2-1],    3,32,32,1, scope='conv_hf21', reuse=None)
                  conv_hi21 = self._conv2D(h2_tm1[i2-1],    3,32,32,1, scope='conv_hi21', reuse=None)
                  conv_hc21 = self._conv2D(h2_tm1[i2-1],    3,32,32,1, scope='conv_hc21', reuse=None)
                  conv_ho21 = self._conv2D(h2_tm1[i2-1],    3,32,32,1, scope='conv_ho21', reuse=None)

                  conv_hf22 = self._conv2D(conv_hf21, 3,32,32,1, scope='conv_hf22', reuse=None)
                  conv_hi22 = self._conv2D(conv_hi21, 3,32,32,1, scope='conv_hi22', reuse=None)
                  conv_hc22 = self._conv2D(conv_hc21, 3,32,32,1, scope='conv_hc22', reuse=None)
                  conv_ho22 = self._conv2D(conv_ho21, 3,32,32,1, scope='conv_ho22', reuse=None)

                  conv_hf23 = self._conv2D(conv_hf22, 3,32,32,1, scope='conv_hf23', reuse=None, activation=None)
                  conv_hi23 = self._conv2D(conv_hi22, 3,32,32,1, scope='conv_hi23', reuse=None, activation=None)
                  conv_hc23 = self._conv2D(conv_hc22, 3,32,32,1, scope='conv_hc23', reuse=None, activation=None)
                  conv_ho23 = self._conv2D(conv_ho22, 3,32,32,1, scope='conv_ho23', reuse=None, activation=None)
              if i2 == 0:
                conv_ft2 = tf.nn.sigmoid(conv_f23, name=scope.name)
                conv_it2 = tf.nn.sigmoid(conv_i23, name=scope.name)
                conv_ct2 = tf.nn.tanh(conv_c23,    name=scope.name)
                conv_ot2 = tf.nn.sigmoid(conv_o23, name=scope.name)
                state2 = conv_it2 * conv_ct2
              else:
                conv_ft2 = tf.nn.sigmoid(tf.add(conv_hf23, conv_f23), name=scope.name)
                conv_it2 = tf.nn.sigmoid(tf.add(conv_hi23, conv_i23), name=scope.name)
                conv_ct2 = tf.nn.tanh(tf.add(conv_hc23, conv_c23),    name=scope.name)
                conv_ot2 = tf.nn.sigmoid(tf.add(conv_ho23, conv_o23), name=scope.name)
                state2 = conv_ft2 * c2_tm1 + conv_it2 * conv_ct2
              recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 2 is the xt
              conv_f21 = self._conv2D(norm2[i2],    3,16,32,2, scope='conv_f21')
              conv_i21 = self._conv2D(norm2[i2],    3,16,32,2, scope='conv_i21')
              conv_c21 = self._conv2D(norm2[i2],    3,16,32,2, scope='conv_c21')
              conv_o21 = self._conv2D(norm2[i2],    3,16,32,2, scope='conv_o21')

              conv_f22 = self._conv2D(conv_f21, 3,32,32,1, scope='conv_f22')
              conv_i22 = self._conv2D(conv_i21, 3,32,32,1, scope='conv_i22')
              conv_c22 = self._conv2D(conv_c21, 3,32,32,1, scope='conv_c22')
              conv_o22 = self._conv2D(conv_o21, 3,32,32,1, scope='conv_o22')

              conv_f23 = self._conv2D(conv_f22, 3,32,32,1, scope='conv_f23', activation=None)
              conv_i23 = self._conv2D(conv_i22, 3,32,32,1, scope='conv_i23', activation=None)
              conv_c23 = self._conv2D(conv_c22, 3,32,32,1, scope='conv_c23', activation=None)
              conv_o23 = self._conv2D(conv_o22, 3,32,32,1, scope='conv_o23', activation=None)

              conv_ft2 = tf.nn.sigmoid(conv_f23, name=scope.name)
              conv_it2 = tf.nn.sigmoid(conv_i23, name=scope.name)
              conv_ct2 = tf.nn.tanh(conv_c23,    name=scope.name)
              conv_ot2 = tf.nn.sigmoid(conv_o23, name=scope.name)
              state2 = conv_it2 * conv_ct2
              recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)
                  
            h2_tm1.append(recurrent_hidden2)
            c2_tm1 = state2

        # Connect to the next ConvLSTM block
        norm3 = h2_tm1
        
        # ConvLSTM block3
        h3_tm1 = []
        with tf.variable_scope('ConvLSTM_3') as scope:
          for i2 in xrange(iter2):
            # LSTM update
            if i2 > 0 or i > 0:
              conv_f31 = self._conv2D(norm3[i2],    3,32,64,2, scope='conv_f31', reuse=scope)
              conv_i31 = self._conv2D(norm3[i2],    3,32,64,2, scope='conv_i31', reuse=scope)
              conv_c31 = self._conv2D(norm3[i2],    3,32,64,2, scope='conv_c31', reuse=scope)
              conv_o31 = self._conv2D(norm3[i2],    3,32,64,2, scope='conv_o31', reuse=scope)

              conv_f32 = self._conv2D(conv_f31, 3,64,64,1, scope='conv_f32', reuse=scope)
              conv_i32 = self._conv2D(conv_i31, 3,64,64,1, scope='conv_i32', reuse=scope)
              conv_c32 = self._conv2D(conv_c31, 3,64,64,1, scope='conv_c32', reuse=scope)
              conv_o32 = self._conv2D(conv_o31, 3,64,64,1, scope='conv_o32', reuse=scope)

              conv_f33 = self._conv2D(conv_f32, 3,64,64,1, scope='conv_f33', reuse=scope, activation=None)
              conv_i33 = self._conv2D(conv_i32, 3,64,64,1, scope='conv_i33', reuse=scope, activation=None)
              conv_c33 = self._conv2D(conv_c32, 3,64,64,1, scope='conv_c33', reuse=scope, activation=None)
              conv_o33 = self._conv2D(conv_o32, 3,64,64,1, scope='conv_o33', reuse=scope, activation=None)
            
              if i2 > 1 or (i2 == 1 and i > 0):
                  conv_hf31 = self._conv2D(h3_tm1[i2-1],    3,64,64,1, scope='conv_hf31', reuse=scope)
                  conv_hi31 = self._conv2D(h3_tm1[i2-1],    3,64,64,1, scope='conv_hi31', reuse=scope)
                  conv_hc31 = self._conv2D(h3_tm1[i2-1],    3,64,64,1, scope='conv_hc31', reuse=scope)
                  conv_ho31 = self._conv2D(h3_tm1[i2-1],    3,64,64,1, scope='conv_ho31', reuse=scope)

                  conv_hf32 = self._conv2D(conv_hf31, 3,64,64,1, scope='conv_hf32', reuse=scope)
                  conv_hi32 = self._conv2D(conv_hi31, 3,64,64,1, scope='conv_hi32', reuse=scope)
                  conv_hc32 = self._conv2D(conv_hc31, 3,64,64,1, scope='conv_hc32', reuse=scope)
                  conv_ho32 = self._conv2D(conv_ho31, 3,64,64,1, scope='conv_ho32', reuse=scope)

                  conv_hf33 = self._conv2D(conv_hf32, 3,64,64,1, scope='conv_hf33', reuse=scope, activation=None)
                  conv_hi33 = self._conv2D(conv_hi32, 3,64,64,1, scope='conv_hi33', reuse=scope, activation=None)
                  conv_hc33 = self._conv2D(conv_hc32, 3,64,64,1, scope='conv_hc33', reuse=scope, activation=None)
                  conv_ho33 = self._conv2D(conv_ho32, 3,64,64,1, scope='conv_ho33', reuse=scope, activation=None)                
              elif i2 > 0:
                  conv_hf31 = self._conv2D(h3_tm1[i2-1],    3,64,64,1, scope='conv_hf31', reuse=None)
                  conv_hi31 = self._conv2D(h3_tm1[i2-1],    3,64,64,1, scope='conv_hi31', reuse=None)
                  conv_hc31 = self._conv2D(h3_tm1[i2-1],    3,64,64,1, scope='conv_hc31', reuse=None)
                  conv_ho31 = self._conv2D(h3_tm1[i2-1],    3,64,64,1, scope='conv_ho31', reuse=None)

                  conv_hf32 = self._conv2D(conv_hf31, 3,64,64,1, scope='conv_hf32', reuse=None)
                  conv_hi32 = self._conv2D(conv_hi31, 3,64,64,1, scope='conv_hi32', reuse=None)
                  conv_hc32 = self._conv2D(conv_hc31, 3,64,64,1, scope='conv_hc32', reuse=None)
                  conv_ho32 = self._conv2D(conv_ho31, 3,64,64,1, scope='conv_ho32', reuse=None)

                  conv_hf33 = self._conv2D(conv_hf32, 3,64,64,1, scope='conv_hf33', reuse=None, activation=None)
                  conv_hi33 = self._conv2D(conv_hi32, 3,64,64,1, scope='conv_hi33', reuse=None, activation=None)
                  conv_hc33 = self._conv2D(conv_hc32, 3,64,64,1, scope='conv_hc33', reuse=None, activation=None)
                  conv_ho33 = self._conv2D(conv_ho32, 3,64,64,1, scope='conv_ho33', reuse=None, activation=None)
              if i2 == 0:
                conv_ft3 = tf.nn.sigmoid(conv_f33, name=scope.name)
                conv_it3 = tf.nn.sigmoid(conv_i33, name=scope.name)
                conv_ct3 = tf.nn.tanh(conv_c33,    name=scope.name)
                conv_ot3 = tf.nn.sigmoid(conv_o33, name=scope.name)
                state3 = conv_it3 * conv_ct3                  
              else:
                conv_ft3 = tf.nn.sigmoid(tf.add(conv_hf33, conv_f33), name=scope.name)
                conv_it3 = tf.nn.sigmoid(tf.add(conv_hi33, conv_i33), name=scope.name)
                conv_ct3 = tf.nn.tanh(tf.add(conv_hc33, conv_c33),    name=scope.name)
                conv_ot3 = tf.nn.sigmoid(tf.add(conv_ho33, conv_o33), name=scope.name)
                state3 = conv_ft3 * c3_tm1 + conv_it3 * conv_ct3
              recurrent_hidden3 = conv_ot3 * tf.nn.tanh(state3, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 2 is the xt
              conv_f31 = self._conv2D(norm3[i2],    3,32,64,2, scope='conv_f31')
              conv_i31 = self._conv2D(norm3[i2],    3,32,64,2, scope='conv_i31')
              conv_c31 = self._conv2D(norm3[i2],    3,32,64,2, scope='conv_c31')
              conv_o31 = self._conv2D(norm3[i2],    3,32,64,2, scope='conv_o31')

              conv_f32 = self._conv2D(conv_f31, 3,64,64,1, scope='conv_f32')
              conv_i32 = self._conv2D(conv_i31, 3,64,64,1, scope='conv_i32')
              conv_c32 = self._conv2D(conv_c31, 3,64,64,1, scope='conv_c32')
              conv_o32 = self._conv2D(conv_o31, 3,64,64,1, scope='conv_o32')            

              conv_f33 = self._conv2D(conv_f32, 3,64,64,1, scope='conv_f33', activation=None)
              conv_i33 = self._conv2D(conv_i32, 3,64,64,1, scope='conv_i33', activation=None)
              conv_c33 = self._conv2D(conv_c32, 3,64,64,1, scope='conv_c33', activation=None)
              conv_o33 = self._conv2D(conv_o32, 3,64,64,1, scope='conv_o33', activation=None)

              conv_ft3 = tf.nn.sigmoid(conv_f33, name=scope.name)
              conv_it3 = tf.nn.sigmoid(conv_i33, name=scope.name)
              conv_ct3 = tf.nn.tanh(conv_c33,    name=scope.name)
              conv_ot3 = tf.nn.sigmoid(conv_o33, name=scope.name)
              state3 = conv_it3 * conv_ct3
              recurrent_hidden3 = conv_ot3 * tf.nn.tanh(state3, name=scope.name)
                  
            h3_tm1.append(recurrent_hidden3)
            c3_tm1 = state3

        # Connect to the next ConvLSTM block
        norm4 = h3_tm1
        
        # ConvLSTM block4
        h4_tm1 = []
        with tf.variable_scope('ConvLSTM_4') as scope:
          for i2 in xrange(iter2):
            # LSTM update
            if i2 > 0 or i > 0:
              conv_f41 = self._conv2D(norm4[i2],    3,64,64,1, scope='conv_f41', reuse=scope)
              conv_i41 = self._conv2D(norm4[i2],    3,64,64,1, scope='conv_i41', reuse=scope)
              conv_c41 = self._conv2D(norm4[i2],    3,64,64,1, scope='conv_c41', reuse=scope)
              conv_o41 = self._conv2D(norm4[i2],    3,64,64,1, scope='conv_o41', reuse=scope)

              conv_f42 = self._conv2D(conv_f41, 3,64,64,1, scope='conv_f42', reuse=scope)
              conv_i42 = self._conv2D(conv_i41, 3,64,64,1, scope='conv_i42', reuse=scope)
              conv_c42 = self._conv2D(conv_c41, 3,64,64,1, scope='conv_c42', reuse=scope)
              conv_o42 = self._conv2D(conv_o41, 3,64,64,1, scope='conv_o42', reuse=scope)

              conv_f43 = self._conv2D(conv_f42, 3,64,64,1, scope='conv_f43', reuse=scope, activation=None)
              conv_i43 = self._conv2D(conv_i42, 3,64,64,1, scope='conv_i43', reuse=scope, activation=None)
              conv_c43 = self._conv2D(conv_c42, 3,64,64,1, scope='conv_c43', reuse=scope, activation=None)
              conv_o43 = self._conv2D(conv_o42, 3,64,64,1, scope='conv_o43', reuse=scope, activation=None)
            
              if i2 > 1 or (i2 == 1 and i > 0):
                  conv_hf41 = self._conv2D(h4_tm1[i2-1],    3,64,64,1, scope='conv_hf41', reuse=scope)
                  conv_hi41 = self._conv2D(h4_tm1[i2-1],    3,64,64,1, scope='conv_hi41', reuse=scope)
                  conv_hc41 = self._conv2D(h4_tm1[i2-1],    3,64,64,1, scope='conv_hc41', reuse=scope)
                  conv_ho41 = self._conv2D(h4_tm1[i2-1],    3,64,64,1, scope='conv_ho41', reuse=scope)

                  conv_hf42 = self._conv2D(conv_hf41, 3,64,64,1, scope='conv_hf42', reuse=scope)
                  conv_hi42 = self._conv2D(conv_hi41, 3,64,64,1, scope='conv_hi42', reuse=scope)
                  conv_hc42 = self._conv2D(conv_hc41, 3,64,64,1, scope='conv_hc42', reuse=scope)
                  conv_ho42 = self._conv2D(conv_ho41, 3,64,64,1, scope='conv_ho42', reuse=scope)

                  conv_hf43 = self._conv2D(conv_hf42, 3,64,64,1, scope='conv_hf43', reuse=scope, activation=None)
                  conv_hi43 = self._conv2D(conv_hi42, 3,64,64,1, scope='conv_hi43', reuse=scope, activation=None)
                  conv_hc43 = self._conv2D(conv_hc42, 3,64,64,1, scope='conv_hc43', reuse=scope, activation=None)
                  conv_ho43 = self._conv2D(conv_ho42, 3,64,64,1, scope='conv_ho43', reuse=scope, activation=None)

              elif i2 > 0:
                  conv_hf41 = self._conv2D(h4_tm1[i2-1],    3,64,64,1, scope='conv_hf41', reuse=None)
                  conv_hi41 = self._conv2D(h4_tm1[i2-1],    3,64,64,1, scope='conv_hi41', reuse=None)
                  conv_hc41 = self._conv2D(h4_tm1[i2-1],    3,64,64,1, scope='conv_hc41', reuse=None)
                  conv_ho41 = self._conv2D(h4_tm1[i2-1],    3,64,64,1, scope='conv_ho41', reuse=None)

                  conv_hf42 = self._conv2D(conv_hf41, 3,64,64,1, scope='conv_hf42', reuse=None)
                  conv_hi42 = self._conv2D(conv_hi41, 3,64,64,1, scope='conv_hi42', reuse=None)
                  conv_hc42 = self._conv2D(conv_hc41, 3,64,64,1, scope='conv_hc42', reuse=None)
                  conv_ho42 = self._conv2D(conv_ho41, 3,64,64,1, scope='conv_ho42', reuse=None)

                  conv_hf43 = self._conv2D(conv_hf42, 3,64,64,1, scope='conv_hf43', reuse=None, activation=None)
                  conv_hi43 = self._conv2D(conv_hi42, 3,64,64,1, scope='conv_hi43', reuse=None, activation=None)
                  conv_hc43 = self._conv2D(conv_hc42, 3,64,64,1, scope='conv_hc43', reuse=None, activation=None)
                  conv_ho43 = self._conv2D(conv_ho42, 3,64,64,1, scope='conv_ho43', reuse=None, activation=None)

              if i2 == 0:
                conv_ft4 = tf.nn.sigmoid(conv_f43, name=scope.name)
                conv_it4 = tf.nn.sigmoid(conv_i43, name=scope.name)
                conv_ct4 = tf.nn.tanh(conv_c43,    name=scope.name)
                conv_ot4 = tf.nn.sigmoid(conv_o43, name=scope.name)
                state4 = conv_it4 * conv_ct4                  
              else:
                conv_ft4 = tf.nn.sigmoid(tf.add(conv_hf43, conv_f43), name=scope.name)
                conv_it4 = tf.nn.sigmoid(tf.add(conv_hi43, conv_i43), name=scope.name)
                conv_ct4 = tf.nn.tanh(tf.add(conv_hc43, conv_c43),    name=scope.name)
                conv_ot4 = tf.nn.sigmoid(tf.add(conv_ho43, conv_o43), name=scope.name)
                state4 = conv_ft4 * c4_tm1 + conv_it4 * conv_ct4
              recurrent_hidden4 = conv_ot4 * tf.nn.tanh(state4, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 2 is the xt
              conv_f41 = self._conv2D(norm4[i2],    3,64,64,1, scope='conv_f41', reuse=None)
              conv_i41 = self._conv2D(norm4[i2],    3,64,64,1, scope='conv_i41', reuse=None)
              conv_c41 = self._conv2D(norm4[i2],    3,64,64,1, scope='conv_c41', reuse=None)
              conv_o41 = self._conv2D(norm4[i2],    3,64,64,1, scope='conv_o41', reuse=None)

              conv_f42 = self._conv2D(conv_f41, 3,64,64,1, scope='conv_f42', reuse=None)
              conv_i42 = self._conv2D(conv_i41, 3,64,64,1, scope='conv_i42', reuse=None)
              conv_c42 = self._conv2D(conv_c41, 3,64,64,1, scope='conv_c42', reuse=None)
              conv_o42 = self._conv2D(conv_o41, 3,64,64,1, scope='conv_o42', reuse=None)

              conv_f43 = self._conv2D(conv_f42, 3,64,64,1, scope='conv_f43', reuse=None, activation=None)
              conv_i43 = self._conv2D(conv_i42, 3,64,64,1, scope='conv_i43', reuse=None, activation=None)
              conv_c43 = self._conv2D(conv_c42, 3,64,64,1, scope='conv_c43', reuse=None, activation=None)
              conv_o43 = self._conv2D(conv_o42, 3,64,64,1, scope='conv_o43', reuse=None, activation=None)

              conv_ft4 = tf.nn.sigmoid(conv_f43, name=scope.name)
              conv_it4 = tf.nn.sigmoid(conv_i43, name=scope.name)
              conv_ct4 = tf.nn.tanh(conv_c43,    name=scope.name)
              conv_ot4 = tf.nn.sigmoid(conv_o43, name=scope.name)
              state4 = conv_it4 * conv_ct4
              recurrent_hidden4 = conv_ot4 * tf.nn.tanh(state4, name=scope.name)
                  
            h4_tm1.append(recurrent_hidden4)
            c4_tm1 = state4

        upsample = tf.image.resize_images(h4_tm1[iter2 - 1], 32, 32, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)

        with tf.variable_scope('Outer') as scope:
          if i == 0:
            # Result is h2_tm1: 8 x 8 x 64
            #      upsample to: 32 x 32 x 16
            # Naive Upsampleing
            downsample = self._conv2D(upsample, 1,64,16,1, scope='upsample')
          else:
            downsample = self._conv2D(upsample, 1,64,16,1, scope='upsample', reuse=scope)
          
          norm1 = tf.add(downsample, norm1)

      # Avg Pooling
      net = h4_tm1
      print(net[0].get_shape())
    for i in xrange(iter2):
        net[i] = self._global_avg_pool(net[i])
    x = tf.reshape(net[0], [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], self.hps.num_classes],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [self.hps.num_classes],
                        initializer=tf.constant_initializer())

    predictions = []
    logits = []
    for i in xrange(iter2):
      x = tf.reshape(net[i], [self.hps.batch_size, -1])
      logits.append(tf.nn.xw_plus_b(x, w, b))
      predictions.append(tf.nn.softmax(logits[i]))

    self.predictions = predictions[iter2-1]
    self.cost = 0
    for i in xrange(iter2):
      with tf.variable_scope('costs'):
        xent = tf.nn.softmax_cross_entropy_with_logits(
            logits[i], self.labels)
        self.cost += tf.reduce_mean(xent, name='xent')
        self.cost += self._decay()

    tf.scalar_summary('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x, reuse=None):
    with tf.variable_op_scope([x], name, 'BatchNorm', reuse=reuse):
      """Batch normalization."""
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _conv2D(self, x, filter_size, in_filter, out_filter, stride, activation=tf.nn.relu, reuse=None, scope=None):
    with tf.variable_op_scope([x], scope, 'Conv', reuse=reuse):
      # kernel_h, kernel_w = _two_element_tuple(kernel_size)
      # stride_h, stride_w = _two_element_tuple(stride)
      # num_filters_in = inputs.get_shape()[-1]
      # weights_shape = [kernel_h, kernel_w,
      #                  num_filters_in, num_filters_out]
      # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      conv = self._conv('Conv', x, filter_size, in_filter, out_filter, [1,stride,stride,1])
      outputs = self._batch_norm('bn', conv)

      if activation:
        outputs = self._relu(outputs, self.hps.relu_leakiness)
      return outputs

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    n = filter_size * filter_size * out_filters
    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters],
        tf.float32, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])




class ResNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
      res_func = self._bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      res_func = self._residual
      filters = [16, 16, 32, 64]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      # filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 9

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in xrange(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in xrange(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in xrange(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits, self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()

      tf.scalar_summary('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.histogram_summary(mean.op.name, mean)
        tf.histogram_summary(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    """Bottleneck resisual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

class ConvLSTM_Baseline(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    # Iteration numbers
    iter1 = 4
    iter2 = 4
    inputs = self._images
    scope = ''
    # Core Model
    with tf.op_scope([inputs], scope, 'inception_v3'):
      # Initial Conv
      conv0 = self._conv2D(inputs, 3, 3, 16, 1, scope='conv0')
      norm1 = conv0

      h1_tm1 = []
      # ConvLSTM block1
      with tf.variable_scope('ConvLSTM_1') as scope:
        for i1 in xrange(iter1):
          # LSTM update
          if i1 > 0:
            conv_f11 = self._conv2D(norm1,  3,16,32,2, scope='conv_f11', reuse=scope)
            conv_i11 = self._conv2D(norm1,  3,16,32,2, scope='conv_i11', reuse=scope)
            conv_c11 = self._conv2D(norm1,  3,16,32,2, scope='conv_c11', reuse=scope)
            conv_o11 = self._conv2D(norm1,  3,16,32,2, scope='conv_o11', reuse=scope)

            conv_f12 = self._conv2D(conv_f11, 3,32,32,1, scope='conv_f12', reuse=scope, activation=None)
            conv_i12 = self._conv2D(conv_i11, 3,32,32,1, scope='conv_i12', reuse=scope, activation=None)
            conv_c12 = self._conv2D(conv_c11, 3,32,32,1, scope='conv_c12', reuse=scope, activation=None)
            conv_o12 = self._conv2D(conv_o11, 3,32,32,1, scope='conv_o12', reuse=scope, activation=None)
          
            if i1 > 1:
                conv_hf11 = self._conv2D(h1_tm1[i1-1], 3,32,32,1, scope='conv_hf11', reuse=scope)
                conv_hi11 = self._conv2D(h1_tm1[i1-1], 3,32,32,1, scope='conv_hi11', reuse=scope)
                conv_hc11 = self._conv2D(h1_tm1[i1-1], 3,32,32,1, scope='conv_hc11', reuse=scope)
                conv_ho11 = self._conv2D(h1_tm1[i1-1], 3,32,32,1, scope='conv_ho11', reuse=scope)

                conv_hf12 = self._conv2D(conv_hf11, 3,32,32,1, scope='conv_hf12', reuse=scope, activation=None)
                conv_hi12 = self._conv2D(conv_hi11, 3,32,32,1, scope='conv_hi12', reuse=scope, activation=None)
                conv_hc12 = self._conv2D(conv_hc11, 3,32,32,1, scope='conv_hc12', reuse=scope, activation=None)
                conv_ho12 = self._conv2D(conv_ho11, 3,32,32,1, scope='conv_ho12', reuse=scope, activation=None)
            else:
                conv_hf11 = self._conv2D(h1_tm1[i1-1],    3,32,32,1, scope='conv_hf11', reuse=None)
                conv_hi11 = self._conv2D(h1_tm1[i1-1],    3,32,32,1, scope='conv_hi11', reuse=None)
                conv_hc11 = self._conv2D(h1_tm1[i1-1],    3,32,32,1, scope='conv_hc11', reuse=None)
                conv_ho11 = self._conv2D(h1_tm1[i1-1],    3,32,32,1, scope='conv_ho11', reuse=None)

                conv_hf12 = self._conv2D(conv_hf11, 3,32,32,1, scope='conv_hf12', reuse=None, activation=None)
                conv_hi12 = self._conv2D(conv_hi11, 3,32,32,1, scope='conv_hi12', reuse=None, activation=None)
                conv_hc12 = self._conv2D(conv_hc11, 3,32,32,1, scope='conv_hc12', reuse=None, activation=None)
                conv_ho12 = self._conv2D(conv_ho11, 3,32,32,1, scope='conv_ho12', reuse=None, activation=None)

            conv_ft1 = tf.nn.sigmoid(tf.add(conv_hf12, conv_f12), name=scope.name)
            conv_it1 = tf.nn.sigmoid(tf.add(conv_hi12, conv_i12), name=scope.name)
            conv_ct1 = tf.nn.tanh(tf.add(conv_hc12, conv_c12),    name=scope.name)
            conv_ot1 = tf.nn.sigmoid(tf.add(conv_ho12, conv_o12), name=scope.name)
            state1 = conv_ft1 * c1_tm1 + conv_it1 * conv_ct1
            recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)

          else:
            # the first step, initial ctm1 htm1 is zeros
            # norm 1 is the xt
            conv_f11 = self._conv2D(norm1,    3,16,32,2, scope='conv_f11')
            conv_i11 = self._conv2D(norm1,    3,16,32,2, scope='conv_i11')
            conv_c11 = self._conv2D(norm1,    3,16,32,2, scope='conv_c11')
            conv_o11 = self._conv2D(norm1,    3,16,32,2, scope='conv_o11')

            conv_f12 = self._conv2D(conv_f11, 3,32,32,1, scope='conv_f12', activation=None)
            conv_i12 = self._conv2D(conv_i11, 3,32,32,1, scope='conv_i12', activation=None)
            conv_c12 = self._conv2D(conv_c11, 3,32,32,1, scope='conv_c12', activation=None)
            conv_o12 = self._conv2D(conv_o11, 3,32,32,1, scope='conv_o12', activation=None)

            conv_ft1 = tf.nn.sigmoid(conv_f12, name=scope.name)
            conv_it1 = tf.nn.sigmoid(conv_i12, name=scope.name)
            conv_ct1 = tf.nn.tanh(conv_c12,    name=scope.name)
            conv_ot1 = tf.nn.sigmoid(conv_o12, name=scope.name)
            state1 = conv_it1 * conv_ct1
            recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)
                
          h1_tm1.append(recurrent_hidden1)
          c1_tm1 = state1

      # Connect to the next ConvLSTM block
      norm2 = h1_tm1
      
      # ConvLSTM block2
      h2_tm1 = []
      with tf.variable_scope('ConvLSTM_2') as scope:
        for i2 in xrange(iter2):
          # LSTM update
          if i2 > 0:
            conv_f21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_f21', reuse=scope)
            conv_i21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_i21', reuse=scope)
            conv_c21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_c21', reuse=scope)
            conv_o21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_o21', reuse=scope)

            conv_f22 = self._conv2D(conv_f21, 3,64,64,1, scope='conv_f22', reuse=scope, activation=None)
            conv_i22 = self._conv2D(conv_i21, 3,64,64,1, scope='conv_i22', reuse=scope, activation=None)
            conv_c22 = self._conv2D(conv_c21, 3,64,64,1, scope='conv_c22', reuse=scope, activation=None)
            conv_o22 = self._conv2D(conv_o21, 3,64,64,1, scope='conv_o22', reuse=scope, activation=None)
          
            if i2 > 1:
                conv_hf21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hf21', reuse=scope)
                conv_hi21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hi21', reuse=scope)
                conv_hc21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hc21', reuse=scope)
                conv_ho21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_ho21', reuse=scope)

                conv_hf22 = self._conv2D(conv_hf21, 3,64,64,1, scope='conv_hf22', reuse=scope, activation=None)
                conv_hi22 = self._conv2D(conv_hi21, 3,64,64,1, scope='conv_hi22', reuse=scope, activation=None)
                conv_hc22 = self._conv2D(conv_hc21, 3,64,64,1, scope='conv_hc22', reuse=scope, activation=None)
                conv_ho22 = self._conv2D(conv_ho21, 3,64,64,1, scope='conv_ho22', reuse=scope, activation=None)
            else:
                conv_hf21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hf21', reuse=None)
                conv_hi21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hi21', reuse=None)
                conv_hc21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hc21', reuse=None)
                conv_ho21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_ho21', reuse=None)

                conv_hf22 = self._conv2D(conv_hf21, 3,64,64,1, scope='conv_hf22', reuse=None, activation=None)
                conv_hi22 = self._conv2D(conv_hi21, 3,64,64,1, scope='conv_hi22', reuse=None, activation=None)
                conv_hc22 = self._conv2D(conv_hc21, 3,64,64,1, scope='conv_hc22', reuse=None, activation=None)
                conv_ho22 = self._conv2D(conv_ho21, 3,64,64,1, scope='conv_ho22', reuse=None, activation=None)


            conv_ft2 = tf.nn.sigmoid(tf.add(conv_hf22, conv_f22), name=scope.name)
            conv_it2 = tf.nn.sigmoid(tf.add(conv_hi22, conv_i22), name=scope.name)
            conv_ct2 = tf.nn.tanh(tf.add(conv_hc22, conv_c22),    name=scope.name)
            conv_ot2 = tf.nn.sigmoid(tf.add(conv_ho22, conv_o22), name=scope.name)
            state2 = conv_ft2 * c2_tm1 + conv_it2 * conv_ct2
            recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)

          else:
            # the first step, initial ctm1 htm1 is zeros
            # norm 2 is the xt
            conv_f21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_f21')
            conv_i21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_i21')
            conv_c21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_c21')
            conv_o21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_o21')

            conv_f22 = self._conv2D(conv_f21, 3,64,64,1, scope='conv_f22', activation=None)
            conv_i22 = self._conv2D(conv_i21, 3,64,64,1, scope='conv_i22', activation=None)
            conv_c22 = self._conv2D(conv_c21, 3,64,64,1, scope='conv_c22', activation=None)
            conv_o22 = self._conv2D(conv_o21, 3,64,64,1, scope='conv_o22', activation=None)

            conv_ft2 = tf.nn.sigmoid(conv_f22, name=scope.name)
            conv_it2 = tf.nn.sigmoid(conv_i22, name=scope.name)
            conv_ct2 = tf.nn.tanh(conv_c22,    name=scope.name)
            conv_ot2 = tf.nn.sigmoid(conv_o22, name=scope.name)
            state2 = conv_it2 * conv_ct2
            recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)
                
          h2_tm1.append(recurrent_hidden2)
          c2_tm1 = state2

      # Avg Pooling
      net = h2_tm1
      print(net[0].get_shape())

    x = tf.reshape(net[0], [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], self.hps.num_classes],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [self.hps.num_classes],
                        initializer=tf.constant_initializer())

    predictions = []
    logits = []
    for i in xrange(iter2):
      x = tf.reshape(net[i], [self.hps.batch_size, -1])
      logits.append(tf.nn.xw_plus_b(x, w, b))
      predictions.append(tf.nn.softmax(logits[i]))

    self.predictions = predictions[iter2-1]
    self.cost = 0
    for i in xrange(iter2):
      with tf.variable_scope('costs'):
        xent = tf.nn.softmax_cross_entropy_with_logits(
            logits[i], self.labels)
        self.cost += tf.reduce_mean(xent, name='xent')
        self.cost += self._decay()

    tf.scalar_summary('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x, reuse=None):
    with tf.variable_op_scope([x], name, 'BatchNorm', reuse=reuse):
      """Batch normalization."""
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _conv2D(self, x, filter_size, in_filter, out_filter, stride, activation=tf.nn.relu, reuse=None, scope=None):
    with tf.variable_op_scope([x], scope, 'Conv', reuse=reuse):
      # kernel_h, kernel_w = _two_element_tuple(kernel_size)
      # stride_h, stride_w = _two_element_tuple(stride)
      # num_filters_in = inputs.get_shape()[-1]
      # weights_shape = [kernel_h, kernel_w,
      #                  num_filters_in, num_filters_out]
      # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      conv = self._conv('Conv', x, filter_size, in_filter, out_filter, [1,stride,stride,1])
      outputs = self._batch_norm('bn', conv)

      if activation:
        outputs = self._relu(outputs, self.hps.relu_leakiness)
      return outputs

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    n = filter_size * filter_size * out_filters
    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters],
        tf.float32, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

class ConvLSTM_SecondOrder(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    # Iteration numbers
    iter1 = 4
    iter2 = 4
    second_order_iter = 4
    print('Second Order Model with: %d %d %d' % (second_order_iter, iter1, iter2))
    inputs = self._images
    scope = ''
    # Core Model
    with tf.op_scope([inputs], scope, 'inception_v3'):
      # Initial Conv
      conv0 = self._conv2D(inputs, 3, 3, 16, 1, scope='conv0')
      norm1 = conv0
      for i in xrange(second_order_iter):
        # Connect to the next ConvLSTM block
        if i > 0:
          # ConvLSTM block1
          with tf.variable_scope('ConvLSTM_1') as scope:
            for i1 in xrange(iter1):
              # LSTM update
              if i1 > 0 or i > 1:
                conv_f1 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_f11', reuse=scope, activation=None)
                conv_i1 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_i11', reuse=scope, activation=None)
                conv_c1 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_c11', reuse=scope, activation=None)
                conv_o1 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_o11', reuse=scope, activation=None)

                if i1 > 1 or i > 1:
                    conv_hf1 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hf11', reuse=scope, activation=None)
                    conv_hi1 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hi11', reuse=scope, activation=None)
                    conv_hc1 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hc11', reuse=scope, activation=None)
                    conv_ho1 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_ho11', reuse=scope, activation=None)
                else:
                    conv_hf1 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hf11', reuse=None, activation=None)
                    conv_hi1 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hi11', reuse=None, activation=None)
                    conv_hc1 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hc11', reuse=None, activation=None)
                    conv_ho1 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_ho11', reuse=None, activation=None)
                if i1 == 0:
                  conv_ft1 = tf.nn.sigmoid(conv_f1, name=scope.name)
                  conv_it1 = tf.nn.sigmoid(conv_i1, name=scope.name)
                  conv_ct1 = tf.nn.tanh(conv_c1,    name=scope.name)
                  conv_ot1 = tf.nn.sigmoid(conv_o1, name=scope.name)
                  state1 = conv_it1 * conv_ct1
                else:
                  conv_ft1 = tf.nn.sigmoid(tf.add(conv_hf1, conv_f1), name=scope.name)
                  conv_it1 = tf.nn.sigmoid(tf.add(conv_hi1, conv_i1), name=scope.name)
                  conv_ct1 = tf.nn.tanh(tf.add(conv_hc1, conv_c1),    name=scope.name)
                  conv_ot1 = tf.nn.sigmoid(tf.add(conv_ho1, conv_o1), name=scope.name)
                  state1 = conv_ft1 * c1_tm1 + conv_it1 * conv_ct1
                recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)

              else:
                # the first step, initial ctm1 htm1 is zeros
                # norm 1 is the xt
                conv_f1 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_f11', activation=None)
                conv_i1 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_i11', activation=None)
                conv_c1 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_c11', activation=None)
                conv_o1 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_o11', activation=None)

                conv_ft1 = tf.nn.sigmoid(conv_f1, name=scope.name)
                conv_it1 = tf.nn.sigmoid(conv_i1, name=scope.name)
                conv_ct1 = tf.nn.tanh(conv_c1,    name=scope.name)
                conv_ot1 = tf.nn.sigmoid(conv_o1, name=scope.name)
                state1 = conv_it1 * conv_ct1
                recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)
                    
              h1_tm1 = recurrent_hidden1
              c1_tm1 = state1
   
          # Connect to the next ConvLSTM block
          norm2 = h1_tm1
          
          # ConvLSTM block2
          with tf.variable_scope('ConvLSTM_2') as scope:
            for i2 in xrange(iter2):
              # LSTM update
              if i2 > 0 or i > 1:
                conv_f2 = self._conv2D(norm2,    3,64,64,1, scope='conv_f21', reuse=scope, activation=None)
                conv_i2 = self._conv2D(norm2,    3,64,64,1, scope='conv_i21', reuse=scope, activation=None)
                conv_c2 = self._conv2D(norm2,    3,64,64,1, scope='conv_c21', reuse=scope, activation=None)
                conv_o2 = self._conv2D(norm2,    3,64,64,1, scope='conv_o21', reuse=scope, activation=None)
              
                if i2 > 1 or i > 1:
                    conv_hf2 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hf21', reuse=scope, activation=None)
                    conv_hi2 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hi21', reuse=scope, activation=None)
                    conv_hc2 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hc21', reuse=scope, activation=None)
                    conv_ho2 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_ho21', reuse=scope, activation=None)
                else:
                    conv_hf2 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hf21', reuse=None, activation=None)
                    conv_hi2 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hi21', reuse=None, activation=None)
                    conv_hc2 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hc21', reuse=None, activation=None)
                    conv_ho2 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_ho21', reuse=None, activation=None)
                if i2 == 0:
                  conv_ft2 = tf.nn.sigmoid(conv_f2, name=scope.name)
                  conv_it2 = tf.nn.sigmoid(conv_i2, name=scope.name)
                  conv_ct2 = tf.nn.tanh(conv_c2,    name=scope.name)
                  conv_ot2 = tf.nn.sigmoid(conv_o2, name=scope.name)
                  state2 = conv_it2 * conv_ct2
                else:
                  conv_ft2 = tf.nn.sigmoid(tf.add(conv_hf2, conv_f2), name=scope.name)
                  conv_it2 = tf.nn.sigmoid(tf.add(conv_hi2, conv_i2), name=scope.name)
                  conv_ct2 = tf.nn.tanh(tf.add(conv_hc2, conv_c2),    name=scope.name)
                  conv_ot2 = tf.nn.sigmoid(tf.add(conv_ho2, conv_o2), name=scope.name)
                  state2 = conv_ft2 * c2_tm1 + conv_it2 * conv_ct2
                recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)

              else:
                # the first step, initial ctm1 htm1 is zeros
                # norm 2 is the xt
                conv_f2 = self._conv2D(norm2,    3,64,64,1, scope='conv_f21', activation=None)
                conv_i2 = self._conv2D(norm2,    3,64,64,1, scope='conv_i21', activation=None)
                conv_c2 = self._conv2D(norm2,    3,64,64,1, scope='conv_c21', activation=None)
                conv_o2 = self._conv2D(norm2,    3,64,64,1, scope='conv_o21', activation=None)

                conv_ft2 = tf.nn.sigmoid(conv_f2, name=scope.name)
                conv_it2 = tf.nn.sigmoid(conv_i2, name=scope.name)
                conv_ct2 = tf.nn.tanh(conv_c2,    name=scope.name)
                conv_ot2 = tf.nn.sigmoid(conv_o2, name=scope.name)
                state2 = conv_it2 * conv_ct2
                recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)
                    
              h2_tm1 = recurrent_hidden2
              c2_tm1 = state2

        #for ii in xrange(4):
        with tf.variable_scope('Outer') as scope:
          if i == 0:
            conv_fo1 = self._conv2D(norm1,    3,16,32,2, scope='conv_fo1')
            conv_io1 = self._conv2D(norm1,    3,16,32,2, scope='conv_io1')
            conv_co1 = self._conv2D(norm1,    3,16,32,2, scope='conv_co1')
            conv_oo1 = self._conv2D(norm1,    3,16,32,2, scope='conv_oo1')

            conv_fo2 = self._conv2D(conv_fo1, 3,32,32,1, scope='conv_fo2')
            conv_io2 = self._conv2D(conv_io1, 3,32,32,1, scope='conv_io2')
            conv_co2 = self._conv2D(conv_co1, 3,32,32,1, scope='conv_co2')
            conv_oo2 = self._conv2D(conv_oo1, 3,32,32,1, scope='conv_oo2')

            conv_fo3 = self._conv2D(conv_fo2, 3,32,64,2, scope='conv_fo3')
            conv_io3 = self._conv2D(conv_io2, 3,32,64,2, scope='conv_io3')
            conv_co3 = self._conv2D(conv_co2, 3,32,64,2, scope='conv_co3')
            conv_oo3 = self._conv2D(conv_oo2, 3,32,64,2, scope='conv_oo3')

            conv_fo4 = self._conv2D(conv_fo3, 3,64,64,1, scope='conv_fo4', activation=None)
            conv_io4 = self._conv2D(conv_io3, 3,64,64,1, scope='conv_io4', activation=None)
            conv_co4 = self._conv2D(conv_co3, 3,64,64,1, scope='conv_co4', activation=None)
            conv_oo4 = self._conv2D(conv_oo3, 3,64,64,1, scope='conv_oo4', activation=None)

            conv_fo = tf.nn.sigmoid(conv_fo4, name=scope.name)
            conv_io = tf.nn.sigmoid(conv_io4, name=scope.name)
            conv_co = tf.nn.tanh(conv_co4,    name=scope.name)
            conv_oo = tf.nn.sigmoid(conv_oo4 , name=scope.name)
            stateo = conv_io * conv_co
            recurrent_hiddeno = conv_oo * tf.nn.tanh(stateo, name=scope.name) 
          else:
            conv_fo1 = self._conv2D(norm1,    3,16,32,2, scope='conv_fo1', reuse=scope)
            conv_io1 = self._conv2D(norm1,    3,16,32,2, scope='conv_io1', reuse=scope)
            conv_co1 = self._conv2D(norm1,    3,16,32,2, scope='conv_co1', reuse=scope)
            conv_oo1 = self._conv2D(norm1,    3,16,32,2, scope='conv_oo1', reuse=scope)

            conv_fo2 = self._conv2D(conv_fo1, 3,32,32,1, scope='conv_fo2', reuse=scope)
            conv_io2 = self._conv2D(conv_io1, 3,32,32,1, scope='conv_io2', reuse=scope)
            conv_co2 = self._conv2D(conv_co1, 3,32,32,1, scope='conv_co2', reuse=scope)
            conv_oo2 = self._conv2D(conv_oo1, 3,32,32,1, scope='conv_oo2', reuse=scope)

            conv_fo3 = self._conv2D(conv_fo2, 3,32,64,2, scope='conv_fo3', reuse=scope)
            conv_io3 = self._conv2D(conv_io2, 3,32,64,2, scope='conv_io3', reuse=scope)
            conv_co3 = self._conv2D(conv_co2, 3,32,64,2, scope='conv_co3', reuse=scope)
            conv_oo3 = self._conv2D(conv_oo2, 3,32,64,2, scope='conv_oo3', reuse=scope)

            conv_fo4 = self._conv2D(conv_fo3, 3,64,64,1, scope='conv_fo4', reuse=scope, activation=None)
            conv_io4 = self._conv2D(conv_io3, 3,64,64,1, scope='conv_io4', reuse=scope, activation=None)
            conv_co4 = self._conv2D(conv_co3, 3,64,64,1, scope='conv_co4', reuse=scope, activation=None)
            conv_oo4 = self._conv2D(conv_oo3, 3,64,64,1, scope='conv_oo4', reuse=scope, activation=None)


            conv_fo = tf.nn.sigmoid(tf.add(conv_ft2, conv_fo4), name=scope.name)
            conv_io = tf.nn.sigmoid(tf.add(conv_it2, conv_io4), name=scope.name)
            conv_co = tf.nn.tanh(tf.add(conv_ct2, conv_co4),    name=scope.name)
            conv_oo = tf.nn.sigmoid(tf.add(conv_ot2, conv_oo4), name=scope.name)
            stateo = conv_fo * co_tm1 + conv_io * conv_co
            recurrent_hiddeno = conv_oo * tf.nn.tanh(stateo, name=scope.name)

          ho_tm1 = recurrent_hiddeno
          co_tm1 = stateo

      net = ho_tm1

      print (net.get_shape())

    with tf.variable_scope('logit'):
      logits = self._fully_connected(net, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits, self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()

      tf.scalar_summary('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x, reuse=None):
    with tf.variable_op_scope([x], name, 'BatchNorm', reuse=reuse):
      """Batch normalization."""
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _conv2D(self, x, filter_size, in_filter, out_filter, stride, activation=tf.nn.relu, reuse=None, scope=None):
    with tf.variable_op_scope([x], scope, 'Conv', reuse=reuse):
      # kernel_h, kernel_w = _two_element_tuple(kernel_size)
      # stride_h, stride_w = _two_element_tuple(stride)
      # num_filters_in = inputs.get_shape()[-1]
      # weights_shape = [kernel_h, kernel_w,
      #                  num_filters_in, num_filters_out]
      # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      conv = self._conv('Conv', x, filter_size, in_filter, out_filter, [1,stride,stride,1])
      outputs = self._batch_norm('bn', conv)

      if activation:
        outputs = self._relu(outputs, self.hps.relu_leakiness)
      return outputs

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    n = filter_size * filter_size * out_filters
    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters],
        tf.float32, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

class ConvLSTM_SecondOrder_DeepInner(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    # Iteration numbers
    iter1 = 4
    iter2 = 4
    second_order_iter = 2
    print('Second Order Deep Inner Model with: %d %d %d' % (second_order_iter, iter1, iter2))
    inputs = self._images
    scope = ''
    # Core Model
    with tf.op_scope([inputs], scope, 'inception_v3'):
      # Initial Conv
      conv0 = self._conv2D(inputs, 3, 3, 16, 1, scope='conv0')
      norm1 = conv0
      for i in xrange(second_order_iter):
        # Connect to the next ConvLSTM block
        if i > 0:
          # ConvLSTM block1
          with tf.variable_scope('ConvLSTM_1') as scope:
            for i1 in xrange(iter1):
              # LSTM update
              if i1 > 0 or i > 1:
                conv_f11 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_f11', reuse=scope)
                conv_i11 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_i11', reuse=scope)
                conv_c11 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_c11', reuse=scope)
                conv_o11 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_o11', reuse=scope)
                conv_f12 = self._conv2D(conv_f11,   3,64,64,1, scope='conv_f12', reuse=scope, activation=None)
                conv_i12 = self._conv2D(conv_i11,   3,64,64,1, scope='conv_i12', reuse=scope, activation=None)
                conv_c12 = self._conv2D(conv_c11,   3,64,64,1, scope='conv_c12', reuse=scope, activation=None)
                conv_o12 = self._conv2D(conv_o11,   3,64,64,1, scope='conv_o12', reuse=scope, activation=None)

                if i1 > 1 or i > 1:
                    conv_hf11 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hf11', reuse=scope)
                    conv_hi11 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hi11', reuse=scope)
                    conv_hc11 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hc11', reuse=scope)
                    conv_ho11 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_ho11', reuse=scope)
                    conv_hf12 = self._conv2D(conv_hf11,   3,64,64,1, scope='conv_hf12', reuse=scope, activation=None)
                    conv_hi12 = self._conv2D(conv_hi11,   3,64,64,1, scope='conv_hi12', reuse=scope, activation=None)
                    conv_hc12 = self._conv2D(conv_hc11,   3,64,64,1, scope='conv_hc12', reuse=scope, activation=None)
                    conv_ho12 = self._conv2D(conv_ho11,   3,64,64,1, scope='conv_ho12', reuse=scope, activation=None)
                else:
                    conv_hf11 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hf11', reuse=None)
                    conv_hi11 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hi11', reuse=None)
                    conv_hc11 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_hc11', reuse=None)
                    conv_ho11 = self._conv2D(h1_tm1,   3,64,64,1, scope='conv_ho11', reuse=None)
                    conv_hf12 = self._conv2D(conv_hf11,   3,64,64,1, scope='conv_hf12', reuse=None, activation=None)
                    conv_hi12 = self._conv2D(conv_hi11,   3,64,64,1, scope='conv_hi12', reuse=None, activation=None)
                    conv_hc12 = self._conv2D(conv_hc11,   3,64,64,1, scope='conv_hc12', reuse=None, activation=None)
                    conv_ho12 = self._conv2D(conv_ho11,   3,64,64,1, scope='conv_ho12', reuse=None, activation=None)
                if i1 == 0:
                  conv_ft1 = tf.nn.sigmoid(conv_f12, name=scope.name)
                  conv_it1 = tf.nn.sigmoid(conv_i12, name=scope.name)
                  conv_ct1 = tf.nn.tanh(conv_c12,    name=scope.name)
                  conv_ot1 = tf.nn.sigmoid(conv_o12, name=scope.name)
                  state1 = conv_it1 * conv_ct1
                else:
                  conv_ft1 = tf.nn.sigmoid(tf.add(conv_hf12, conv_f12), name=scope.name)
                  conv_it1 = tf.nn.sigmoid(tf.add(conv_hi12, conv_i12), name=scope.name)
                  conv_ct1 = tf.nn.tanh(tf.add(conv_hc12, conv_c12),    name=scope.name)
                  conv_ot1 = tf.nn.sigmoid(tf.add(conv_ho12, conv_o12), name=scope.name)
                  state1 = conv_ft1 * c1_tm1 + conv_it1 * conv_ct1
                recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)

              else:
                # the first step, initial ctm1 htm1 is zeros
                # norm 1 is the xt
                conv_f11 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_f11', reuse=None)
                conv_i11 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_i11', reuse=None)
                conv_c11 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_c11', reuse=None)
                conv_o11 = self._conv2D(ho_tm1,   3,64,64,1, scope='conv_o11', reuse=None)
                conv_f12 = self._conv2D(conv_f11,   3,64,64,1, scope='conv_f12', reuse=None, activation=None)
                conv_i12 = self._conv2D(conv_i11,   3,64,64,1, scope='conv_i12', reuse=None, activation=None)
                conv_c12 = self._conv2D(conv_c11,   3,64,64,1, scope='conv_c12', reuse=None, activation=None)
                conv_o12 = self._conv2D(conv_o11,   3,64,64,1, scope='conv_o12', reuse=None, activation=None)

                conv_ft1 = tf.nn.sigmoid(conv_f12, name=scope.name)
                conv_it1 = tf.nn.sigmoid(conv_i12, name=scope.name)
                conv_ct1 = tf.nn.tanh(conv_c12,    name=scope.name)
                conv_ot1 = tf.nn.sigmoid(conv_o12, name=scope.name)
                state1 = conv_it1 * conv_ct1
                recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)
                    
              h1_tm1 = recurrent_hidden1
              c1_tm1 = state1
   
          # Connect to the next ConvLSTM block
          norm2 = h1_tm1
          
          # ConvLSTM block2
          with tf.variable_scope('ConvLSTM_2') as scope:
            for i2 in xrange(iter2):
              # LSTM update
              if i2 > 0 or i > 1:
                conv_f21 = self._conv2D(norm2,    3,64,64,1, scope='conv_f21', reuse=scope)
                conv_i21 = self._conv2D(norm2,    3,64,64,1, scope='conv_i21', reuse=scope)
                conv_c21 = self._conv2D(norm2,    3,64,64,1, scope='conv_c21', reuse=scope)
                conv_o21 = self._conv2D(norm2,    3,64,64,1, scope='conv_o21', reuse=scope)
                conv_f22 = self._conv2D(conv_f21,    3,64,64,1, scope='conv_f22', reuse=scope, activation=None)
                conv_i22 = self._conv2D(conv_i21,    3,64,64,1, scope='conv_i22', reuse=scope, activation=None)
                conv_c22 = self._conv2D(conv_c21,    3,64,64,1, scope='conv_c22', reuse=scope, activation=None)
                conv_o22 = self._conv2D(conv_o21,    3,64,64,1, scope='conv_o22', reuse=scope, activation=None)              
                if i2 > 1 or i > 1:
                    conv_hf21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hf21', reuse=scope)
                    conv_hi21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hi21', reuse=scope)
                    conv_hc21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hc21', reuse=scope)
                    conv_ho21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_ho21', reuse=scope)
                    conv_hf22 = self._conv2D(conv_hf21,    3,64,64,1, scope='conv_hf22', reuse=scope, activation=None)
                    conv_hi22 = self._conv2D(conv_hi21,    3,64,64,1, scope='conv_hi22', reuse=scope, activation=None)
                    conv_hc22 = self._conv2D(conv_hc21,    3,64,64,1, scope='conv_hc22', reuse=scope, activation=None)
                    conv_ho22 = self._conv2D(conv_ho21,    3,64,64,1, scope='conv_ho22', reuse=scope, activation=None)
                else:
                    conv_hf21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hf21', reuse=None)
                    conv_hi21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hi21', reuse=None)
                    conv_hc21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hc21', reuse=None)
                    conv_ho21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_ho21', reuse=None)
                    conv_hf22 = self._conv2D(conv_hf21,    3,64,64,1, scope='conv_hf22', reuse=None, activation=None)
                    conv_hi22 = self._conv2D(conv_hi21,    3,64,64,1, scope='conv_hi22', reuse=None, activation=None)
                    conv_hc22 = self._conv2D(conv_hc21,    3,64,64,1, scope='conv_hc22', reuse=None, activation=None)
                    conv_ho22 = self._conv2D(conv_ho21,    3,64,64,1, scope='conv_ho22', reuse=None, activation=None)
                if i2 == 0:
                  conv_ft2 = tf.nn.sigmoid(conv_f22, name=scope.name)
                  conv_it2 = tf.nn.sigmoid(conv_i22, name=scope.name)
                  conv_ct2 = tf.nn.tanh(conv_c22,    name=scope.name)
                  conv_ot2 = tf.nn.sigmoid(conv_o22, name=scope.name)
                  state2 = conv_it2 * conv_ct2
                else:
                  conv_ft2 = tf.nn.sigmoid(tf.add(conv_hf22, conv_f22), name=scope.name)
                  conv_it2 = tf.nn.sigmoid(tf.add(conv_hi22, conv_i22), name=scope.name)
                  conv_ct2 = tf.nn.tanh(tf.add(conv_hc22, conv_c22),    name=scope.name)
                  conv_ot2 = tf.nn.sigmoid(tf.add(conv_ho22, conv_o22), name=scope.name)
                  state2 = conv_ft2 * c2_tm1 + conv_it2 * conv_ct2
                recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)

              else:
                # the first step, initial ctm1 htm1 is zeros
                # norm 2 is the xt
                conv_f21 = self._conv2D(norm2,    3,64,64,1, scope='conv_f21', reuse=None)
                conv_i21 = self._conv2D(norm2,    3,64,64,1, scope='conv_i21', reuse=None)
                conv_c21 = self._conv2D(norm2,    3,64,64,1, scope='conv_c21', reuse=None)
                conv_o21 = self._conv2D(norm2,    3,64,64,1, scope='conv_o21', reuse=None)
                conv_f22 = self._conv2D(conv_f21,    3,64,64,1, scope='conv_f22', reuse=None, activation=None)
                conv_i22 = self._conv2D(conv_i21,    3,64,64,1, scope='conv_i22', reuse=None, activation=None)
                conv_c22 = self._conv2D(conv_c21,    3,64,64,1, scope='conv_c22', reuse=None, activation=None)
                conv_o22 = self._conv2D(conv_o21,    3,64,64,1, scope='conv_o22', reuse=None, activation=None)    

                conv_ft2 = tf.nn.sigmoid(conv_f22, name=scope.name)
                conv_it2 = tf.nn.sigmoid(conv_i22, name=scope.name)
                conv_ct2 = tf.nn.tanh(conv_c22,    name=scope.name)
                conv_ot2 = tf.nn.sigmoid(conv_o22, name=scope.name)
                state2 = conv_it2 * conv_ct2
                recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)
                    
              h2_tm1 = recurrent_hidden2
              c2_tm1 = state2

        #for ii in xrange(4):
        with tf.variable_scope('Outer') as scope:
          if i == 0:
            conv_fo1 = self._conv2D(norm1,    3,16,32,2, scope='conv_fo1')
            conv_io1 = self._conv2D(norm1,    3,16,32,2, scope='conv_io1')
            conv_co1 = self._conv2D(norm1,    3,16,32,2, scope='conv_co1')
            conv_oo1 = self._conv2D(norm1,    3,16,32,2, scope='conv_oo1')

            conv_fo2 = self._conv2D(conv_fo1, 3,32,64,2, scope='conv_fo2', activation=None)
            conv_io2 = self._conv2D(conv_io1, 3,32,64,2, scope='conv_io2', activation=None)
            conv_co2 = self._conv2D(conv_co1, 3,32,64,2, scope='conv_co2', activation=None)
            conv_oo2 = self._conv2D(conv_oo1, 3,32,64,2, scope='conv_oo2', activation=None)

            conv_fo = tf.nn.sigmoid(conv_fo2, name=scope.name)
            conv_io = tf.nn.sigmoid(conv_io2, name=scope.name)
            conv_co = tf.nn.tanh(conv_co2,    name=scope.name)
            conv_oo = tf.nn.sigmoid(conv_oo2 , name=scope.name)
            stateo = conv_io * conv_co
            recurrent_hiddeno = conv_oo * tf.nn.tanh(stateo, name=scope.name) 
          else:
            conv_fo1 = self._conv2D(norm1,    3,16,32,2, scope='conv_fo1', reuse=scope)
            conv_io1 = self._conv2D(norm1,    3,16,32,2, scope='conv_io1', reuse=scope)
            conv_co1 = self._conv2D(norm1,    3,16,32,2, scope='conv_co1', reuse=scope)
            conv_oo1 = self._conv2D(norm1,    3,16,32,2, scope='conv_oo1', reuse=scope)

            conv_fo2 = self._conv2D(conv_fo1, 3,32,64,2, scope='conv_fo2', reuse=scope, activation=None)
            conv_io2 = self._conv2D(conv_io1, 3,32,64,2, scope='conv_io2', reuse=scope, activation=None)
            conv_co2 = self._conv2D(conv_co1, 3,32,64,2, scope='conv_co2', reuse=scope, activation=None)
            conv_oo2 = self._conv2D(conv_oo1, 3,32,64,2, scope='conv_oo2', reuse=scope, activation=None)


            conv_fo = tf.nn.sigmoid(tf.add(conv_ft2, conv_fo2), name=scope.name)
            conv_io = tf.nn.sigmoid(tf.add(conv_it2, conv_io2), name=scope.name)
            conv_co = tf.nn.tanh(tf.add(conv_ct2, conv_co2),    name=scope.name)
            conv_oo = tf.nn.sigmoid(tf.add(conv_ot2, conv_oo2), name=scope.name)
            stateo = conv_fo * co_tm1 + conv_io * conv_co
            recurrent_hiddeno = conv_oo * tf.nn.tanh(stateo, name=scope.name)

          ho_tm1 = recurrent_hiddeno
          co_tm1 = stateo

      net = ho_tm1

      print (net.get_shape())

    with tf.variable_scope('logit'):
      logits = self._fully_connected(net, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits, self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()

      tf.scalar_summary('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x, reuse=None):
    with tf.variable_op_scope([x], name, 'BatchNorm', reuse=reuse):
      """Batch normalization."""
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _conv2D(self, x, filter_size, in_filter, out_filter, stride, activation=tf.nn.relu, reuse=None, scope=None):
    with tf.variable_op_scope([x], scope, 'Conv', reuse=reuse):
      # kernel_h, kernel_w = _two_element_tuple(kernel_size)
      # stride_h, stride_w = _two_element_tuple(stride)
      # num_filters_in = inputs.get_shape()[-1]
      # weights_shape = [kernel_h, kernel_w,
      #                  num_filters_in, num_filters_out]
      # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      conv = self._conv('Conv', x, filter_size, in_filter, out_filter, [1,stride,stride,1])
      outputs = self._batch_norm('bn', conv)

      if activation:
        outputs = self._relu(outputs, self.hps.relu_leakiness)
      return outputs

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    n = filter_size * filter_size * out_filters
    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters],
        tf.float32, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

class ConvLSTM_SecondOrder_UpAndDownSample(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    # Iteration numbers
    iter1 = 4
    iter2 = 4
    second_order_iter = 1
    print('Second Order Naive Model with: %d %d %d' % (second_order_iter, iter1, iter2))
    inputs = self._images
    scope = ''
    # Core Model
    with tf.op_scope([inputs], scope, 'inception_v3'):
      # Initial Conv
      conv0 = self._conv2D(inputs, 3, 3, 16, 1, scope='conv0')
      ho_tm1 = conv0
      for i in xrange(second_order_iter):
        # Connect to the next ConvLSTM block
        # ConvLSTM block1

        with tf.variable_scope('ConvLSTM_1') as scope:
          for i1 in xrange(iter1):
            # LSTM update
            if i1 > 0 or i > 0:
              conv_f11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_f11', reuse=scope)
              conv_i11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_i11', reuse=scope)
              conv_c11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_c11', reuse=scope)
              conv_o11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_o11', reuse=scope)
              conv_f12 = self._conv2D(conv_f11,   3,32,32,1, scope='conv_f12', reuse=scope, activation=None)
              conv_i12 = self._conv2D(conv_i11,   3,32,32,1, scope='conv_i12', reuse=scope, activation=None)
              conv_c12 = self._conv2D(conv_c11,   3,32,32,1, scope='conv_c12', reuse=scope, activation=None)
              conv_o12 = self._conv2D(conv_o11,   3,32,32,1, scope='conv_o12', reuse=scope, activation=None)

              if i1 > 1 or i > 0:
                  conv_hf11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hf11', reuse=scope)
                  conv_hi11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hi11', reuse=scope)
                  conv_hc11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hc11', reuse=scope)
                  conv_ho11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_ho11', reuse=scope)
                  conv_hf12 = self._conv2D(conv_hf11,   3,32,32,1, scope='conv_hf12', reuse=scope, activation=None)
                  conv_hi12 = self._conv2D(conv_hi11,   3,32,32,1, scope='conv_hi12', reuse=scope, activation=None)
                  conv_hc12 = self._conv2D(conv_hc11,   3,32,32,1, scope='conv_hc12', reuse=scope, activation=None)
                  conv_ho12 = self._conv2D(conv_ho11,   3,32,32,1, scope='conv_ho12', reuse=scope, activation=None)
              else:
                  conv_hf11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hf11', reuse=None)
                  conv_hi11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hi11', reuse=None)
                  conv_hc11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hc11', reuse=None)
                  conv_ho11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_ho11', reuse=None)
                  conv_hf12 = self._conv2D(conv_hf11,   3,32,32,1, scope='conv_hf12', reuse=None, activation=None)
                  conv_hi12 = self._conv2D(conv_hi11,   3,32,32,1, scope='conv_hi12', reuse=None, activation=None)
                  conv_hc12 = self._conv2D(conv_hc11,   3,32,32,1, scope='conv_hc12', reuse=None, activation=None)
                  conv_ho12 = self._conv2D(conv_ho11,   3,32,32,1, scope='conv_ho12', reuse=None, activation=None)
              if i1 == 0:
                conv_ft1 = tf.nn.sigmoid(conv_f12, name=scope.name)
                conv_it1 = tf.nn.sigmoid(conv_i12, name=scope.name)
                conv_ct1 = tf.nn.tanh(conv_c12,    name=scope.name)
                conv_ot1 = tf.nn.sigmoid(conv_o12, name=scope.name)
                state1 = conv_it1 * conv_ct1
              else:
                conv_ft1 = tf.nn.sigmoid(tf.add(conv_hf12, conv_f12), name=scope.name)
                conv_it1 = tf.nn.sigmoid(tf.add(conv_hi12, conv_i12), name=scope.name)
                conv_ct1 = tf.nn.tanh(tf.add(conv_hc12, conv_c12),    name=scope.name)
                conv_ot1 = tf.nn.sigmoid(tf.add(conv_ho12, conv_o12), name=scope.name)
                state1 = conv_ft1 * c1_tm1 + conv_it1 * conv_ct1
              recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 1 is the xt
              conv_f11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_f11', reuse=None)
              conv_i11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_i11', reuse=None)
              conv_c11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_c11', reuse=None)
              conv_o11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_o11', reuse=None)
              conv_f12 = self._conv2D(conv_f11,   3,32,32,1, scope='conv_f12', reuse=None, activation=None)
              conv_i12 = self._conv2D(conv_i11,   3,32,32,1, scope='conv_i12', reuse=None, activation=None)
              conv_c12 = self._conv2D(conv_c11,   3,32,32,1, scope='conv_c12', reuse=None, activation=None)
              conv_o12 = self._conv2D(conv_o11,   3,32,32,1, scope='conv_o12', reuse=None, activation=None)

              conv_ft1 = tf.nn.sigmoid(conv_f12, name=scope.name)
              conv_it1 = tf.nn.sigmoid(conv_i12, name=scope.name)
              conv_ct1 = tf.nn.tanh(conv_c12,    name=scope.name)
              conv_ot1 = tf.nn.sigmoid(conv_o12, name=scope.name)
              state1 = conv_it1 * conv_ct1
              recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)
                  
            h1_tm1 = recurrent_hidden1
            c1_tm1 = state1
 
        # Connect to the next ConvLSTM block
        norm2 = h1_tm1
        
        # ConvLSTM block2
        with tf.variable_scope('ConvLSTM_2') as scope:
          for i2 in xrange(iter2):
            # LSTM update
            if i2 > 0 or i > 0:
              conv_f21 = self._conv2D(norm2,    3,32,64,2, scope='conv_f21', reuse=scope)
              conv_i21 = self._conv2D(norm2,    3,32,64,2, scope='conv_i21', reuse=scope)
              conv_c21 = self._conv2D(norm2,    3,32,64,2, scope='conv_c21', reuse=scope)
              conv_o21 = self._conv2D(norm2,    3,32,64,2, scope='conv_o21', reuse=scope)
              conv_f22 = self._conv2D(conv_f21,    3,64,64,1, scope='conv_f22', reuse=scope, activation=None)
              conv_i22 = self._conv2D(conv_i21,    3,64,64,1, scope='conv_i22', reuse=scope, activation=None)
              conv_c22 = self._conv2D(conv_c21,    3,64,64,1, scope='conv_c22', reuse=scope, activation=None)
              conv_o22 = self._conv2D(conv_o21,    3,64,64,1, scope='conv_o22', reuse=scope, activation=None)              
              if i2 > 1 or i > 0:
                  conv_hf21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hf21', reuse=scope)
                  conv_hi21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hi21', reuse=scope)
                  conv_hc21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hc21', reuse=scope)
                  conv_ho21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_ho21', reuse=scope)
                  conv_hf22 = self._conv2D(conv_hf21,    3,64,64,1, scope='conv_hf22', reuse=scope, activation=None)
                  conv_hi22 = self._conv2D(conv_hi21,    3,64,64,1, scope='conv_hi22', reuse=scope, activation=None)
                  conv_hc22 = self._conv2D(conv_hc21,    3,64,64,1, scope='conv_hc22', reuse=scope, activation=None)
                  conv_ho22 = self._conv2D(conv_ho21,    3,64,64,1, scope='conv_ho22', reuse=scope, activation=None)
              else:
                  conv_hf21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hf21', reuse=None)
                  conv_hi21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hi21', reuse=None)
                  conv_hc21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_hc21', reuse=None)
                  conv_ho21 = self._conv2D(h2_tm1,    3,64,64,1, scope='conv_ho21', reuse=None)
                  conv_hf22 = self._conv2D(conv_hf21,    3,64,64,1, scope='conv_hf22', reuse=None, activation=None)
                  conv_hi22 = self._conv2D(conv_hi21,    3,64,64,1, scope='conv_hi22', reuse=None, activation=None)
                  conv_hc22 = self._conv2D(conv_hc21,    3,64,64,1, scope='conv_hc22', reuse=None, activation=None)
                  conv_ho22 = self._conv2D(conv_ho21,    3,64,64,1, scope='conv_ho22', reuse=None, activation=None)
              if i2 == 0:
                conv_ft2 = tf.nn.sigmoid(conv_f22, name=scope.name)
                conv_it2 = tf.nn.sigmoid(conv_i22, name=scope.name)
                conv_ct2 = tf.nn.tanh(conv_c22,    name=scope.name)
                conv_ot2 = tf.nn.sigmoid(conv_o22, name=scope.name)
                state2 = conv_it2 * conv_ct2
              else:
                conv_ft2 = tf.nn.sigmoid(tf.add(conv_hf22, conv_f22), name=scope.name)
                conv_it2 = tf.nn.sigmoid(tf.add(conv_hi22, conv_i22), name=scope.name)
                conv_ct2 = tf.nn.tanh(tf.add(conv_hc22, conv_c22),    name=scope.name)
                conv_ot2 = tf.nn.sigmoid(tf.add(conv_ho22, conv_o22), name=scope.name)
                state2 = conv_ft2 * c2_tm1 + conv_it2 * conv_ct2
              recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 2 is the xt
              conv_f21 = self._conv2D(norm2,    3,32,64,2, scope='conv_f21', reuse=None)
              conv_i21 = self._conv2D(norm2,    3,32,64,2, scope='conv_i21', reuse=None)
              conv_c21 = self._conv2D(norm2,    3,32,64,2, scope='conv_c21', reuse=None)
              conv_o21 = self._conv2D(norm2,    3,32,64,2, scope='conv_o21', reuse=None)
              conv_f22 = self._conv2D(conv_f21,    3,64,64,1, scope='conv_f22', reuse=None, activation=None)
              conv_i22 = self._conv2D(conv_i21,    3,64,64,1, scope='conv_i22', reuse=None, activation=None)
              conv_c22 = self._conv2D(conv_c21,    3,64,64,1, scope='conv_c22', reuse=None, activation=None)
              conv_o22 = self._conv2D(conv_o21,    3,64,64,1, scope='conv_o22', reuse=None, activation=None)    

              conv_ft2 = tf.nn.sigmoid(conv_f22, name=scope.name)
              conv_it2 = tf.nn.sigmoid(conv_i22, name=scope.name)
              conv_ct2 = tf.nn.tanh(conv_c22,    name=scope.name)
              conv_ot2 = tf.nn.sigmoid(conv_o22, name=scope.name)
              state2 = conv_it2 * conv_ct2
              recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)
                  
            h2_tm1 = recurrent_hidden2
            c2_tm1 = state2

        upsample = tf.image.resize_images(h2_tm1, 32, 32, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)

        with tf.variable_scope('Outer') as scope:
          if i == 0:
            # Result is h2_tm1: 8 x 8 x 64
            #      upsample to: 32 x 32 x 16
            #Naive Upsampleing
            downsample = self._conv2D(upsample, 1,64,16,1, scope='upsample')
          else:
            downsample = self._conv2D(upsample, 1,64,16,1, scope='upsample', reuse=scope)
          ho_tm1 = tf.add(downsample, ho_tm1)

      net = h2_tm1

      print (net.get_shape())

    with tf.variable_scope('logit'):
      logits = self._fully_connected(net, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits, self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()

      tf.scalar_summary('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x, reuse=None):
    with tf.variable_op_scope([x], name, 'BatchNorm', reuse=reuse):
      """Batch normalization."""
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _conv2D(self, x, filter_size, in_filter, out_filter, stride, activation=tf.nn.relu, reuse=None, scope=None):
    with tf.variable_op_scope([x], scope, 'Conv', reuse=reuse):
      # kernel_h, kernel_w = _two_element_tuple(kernel_size)
      # stride_h, stride_w = _two_element_tuple(stride)
      # num_filters_in = inputs.get_shape()[-1]
      # weights_shape = [kernel_h, kernel_w,
      #                  num_filters_in, num_filters_out]
      # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      conv = self._conv('Conv', x, filter_size, in_filter, out_filter, [1,stride,stride,1])
      outputs = self._batch_norm('bn', conv)

      if activation:
        outputs = self._relu(outputs, self.hps.relu_leakiness)
      return outputs

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    n = filter_size * filter_size * out_filters
    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters],
        tf.float32, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

class ConvLSTM_SecondOrder_DeepAvgPool(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    # Iteration numbers
    iter1 = 4
    iter2 = 4
    iter3 = 4
    iter4 = 4
    second_order_iter = 1
    print('Second Order Deep AvgPool Model with: %d %d %d %d %d' % (second_order_iter, iter1, iter2, iter3, iter4))
    inputs = self._images
    scope = ''
    # Core Model
    with tf.op_scope([inputs], scope, 'inception_v3'):
      # Initial Conv
      conv0 = self._conv2D(inputs, 3, 3, 16, 1, scope='conv0')
      ho_tm1 = conv0
      for i in xrange(second_order_iter):
        # Connect to the next ConvLSTM block
        # ConvLSTM block1
        with tf.variable_scope('ConvLSTM_1') as scope:
          for i1 in xrange(iter1):
            # LSTM update
            if i1 > 0 or i > 0:
              conv_f11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_f11', reuse=scope)
              conv_i11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_i11', reuse=scope)
              conv_c11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_c11', reuse=scope)
              conv_o11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_o11', reuse=scope)
              
              conv_f12 = self._conv2D(conv_f11,   3,32,32,1, scope='conv_f12', reuse=scope, activation=None)
              conv_i12 = self._conv2D(conv_i11,   3,32,32,1, scope='conv_i12', reuse=scope, activation=None)
              conv_c12 = self._conv2D(conv_c11,   3,32,32,1, scope='conv_c12', reuse=scope, activation=None)
              conv_o12 = self._conv2D(conv_o11,   3,32,32,1, scope='conv_o12', reuse=scope, activation=None)

              if i1 > 1 or i > 0:
                  conv_hf11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hf11', reuse=scope)
                  conv_hi11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hi11', reuse=scope)
                  conv_hc11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hc11', reuse=scope)
                  conv_ho11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_ho11', reuse=scope)
                  conv_hf12 = self._conv2D(conv_hf11,   3,32,32,1, scope='conv_hf12', reuse=scope, activation=None)
                  conv_hi12 = self._conv2D(conv_hi11,   3,32,32,1, scope='conv_hi12', reuse=scope, activation=None)
                  conv_hc12 = self._conv2D(conv_hc11,   3,32,32,1, scope='conv_hc12', reuse=scope, activation=None)
                  conv_ho12 = self._conv2D(conv_ho11,   3,32,32,1, scope='conv_ho12', reuse=scope, activation=None)
              else:
                  conv_hf11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hf11', reuse=None)
                  conv_hi11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hi11', reuse=None)
                  conv_hc11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hc11', reuse=None)
                  conv_ho11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_ho11', reuse=None)
                  conv_hf12 = self._conv2D(conv_hf11,   3,32,32,1, scope='conv_hf12', reuse=None, activation=None)
                  conv_hi12 = self._conv2D(conv_hi11,   3,32,32,1, scope='conv_hi12', reuse=None, activation=None)
                  conv_hc12 = self._conv2D(conv_hc11,   3,32,32,1, scope='conv_hc12', reuse=None, activation=None)
                  conv_ho12 = self._conv2D(conv_ho11,   3,32,32,1, scope='conv_ho12', reuse=None, activation=None)
              if i1 == 0:
                conv_ft1 = tf.nn.sigmoid(conv_f12, name=scope.name)
                conv_it1 = tf.nn.sigmoid(conv_i12, name=scope.name)
                conv_ct1 = tf.nn.tanh(conv_c12,    name=scope.name)
                conv_ot1 = tf.nn.sigmoid(conv_o12, name=scope.name)
                state1 = conv_it1 * conv_ct1
              else:
                conv_ft1 = tf.nn.sigmoid(tf.add(conv_hf12, conv_f12), name=scope.name)
                conv_it1 = tf.nn.sigmoid(tf.add(conv_hi12, conv_i12), name=scope.name)
                conv_ct1 = tf.nn.tanh(tf.add(conv_hc12, conv_c12),    name=scope.name)
                conv_ot1 = tf.nn.sigmoid(tf.add(conv_ho12, conv_o12), name=scope.name)
                state1 = conv_ft1 * c1_tm1 + conv_it1 * conv_ct1
              recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 1 is the xt
              conv_f11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_f11', reuse=None)
              conv_i11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_i11', reuse=None)
              conv_c11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_c11', reuse=None)
              conv_o11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_o11', reuse=None)
              conv_f12 = self._conv2D(conv_f11,   3,32,32,1, scope='conv_f12', reuse=None, activation=None)
              conv_i12 = self._conv2D(conv_i11,   3,32,32,1, scope='conv_i12', reuse=None, activation=None)
              conv_c12 = self._conv2D(conv_c11,   3,32,32,1, scope='conv_c12', reuse=None, activation=None)
              conv_o12 = self._conv2D(conv_o11,   3,32,32,1, scope='conv_o12', reuse=None, activation=None)

              conv_ft1 = tf.nn.sigmoid(conv_f12, name=scope.name)
              conv_it1 = tf.nn.sigmoid(conv_i12, name=scope.name)
              conv_ct1 = tf.nn.tanh(conv_c12,    name=scope.name)
              conv_ot1 = tf.nn.sigmoid(conv_o12, name=scope.name)
              state1 = conv_it1 * conv_ct1
              recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)
                  
            h1_tm1 = recurrent_hidden1
            c1_tm1 = state1
        
        # ConvLSTM block2
        with tf.variable_scope('ConvLSTM_2') as scope:
          for i2 in xrange(iter2):
            # LSTM update
            if i2 > 0 or i > 0:
              conv_f21 = self._conv2D(h1_tm1,    3,32,32,1, scope='conv_f21', reuse=scope)
              conv_i21 = self._conv2D(h1_tm1,    3,32,32,1, scope='conv_i21', reuse=scope)
              conv_c21 = self._conv2D(h1_tm1,    3,32,32,1, scope='conv_c21', reuse=scope)
              conv_o21 = self._conv2D(h1_tm1,    3,32,32,1, scope='conv_o21', reuse=scope)
              conv_f22 = self._conv2D(conv_f21,    3,32,32,1, scope='conv_f22', reuse=scope, activation=None)
              conv_i22 = self._conv2D(conv_i21,    3,32,32,1, scope='conv_i22', reuse=scope, activation=None)
              conv_c22 = self._conv2D(conv_c21,    3,32,32,1, scope='conv_c22', reuse=scope, activation=None)
              conv_o22 = self._conv2D(conv_o21,    3,32,32,1, scope='conv_o22', reuse=scope, activation=None)              
              if i2 > 1 or i > 0:
                  conv_hf21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hf21', reuse=scope)
                  conv_hi21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hi21', reuse=scope)
                  conv_hc21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hc21', reuse=scope)
                  conv_ho21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_ho21', reuse=scope)
                  conv_hf22 = self._conv2D(conv_hf21,    3,32,32,1, scope='conv_hf22', reuse=scope, activation=None)
                  conv_hi22 = self._conv2D(conv_hi21,    3,32,32,1, scope='conv_hi22', reuse=scope, activation=None)
                  conv_hc22 = self._conv2D(conv_hc21,    3,32,32,1, scope='conv_hc22', reuse=scope, activation=None)
                  conv_ho22 = self._conv2D(conv_ho21,    3,32,32,1, scope='conv_ho22', reuse=scope, activation=None)
              else:
                  conv_hf21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hf21', reuse=None)
                  conv_hi21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hi21', reuse=None)
                  conv_hc21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hc21', reuse=None)
                  conv_ho21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_ho21', reuse=None)
                  conv_hf22 = self._conv2D(conv_hf21,    3,32,32,1, scope='conv_hf22', reuse=None, activation=None)
                  conv_hi22 = self._conv2D(conv_hi21,    3,32,32,1, scope='conv_hi22', reuse=None, activation=None)
                  conv_hc22 = self._conv2D(conv_hc21,    3,32,32,1, scope='conv_hc22', reuse=None, activation=None)
                  conv_ho22 = self._conv2D(conv_ho21,    3,32,32,1, scope='conv_ho22', reuse=None, activation=None)
              if i2 == 0:
                conv_ft2 = tf.nn.sigmoid(conv_f22, name=scope.name)
                conv_it2 = tf.nn.sigmoid(conv_i22, name=scope.name)
                conv_ct2 = tf.nn.tanh(conv_c22,    name=scope.name)
                conv_ot2 = tf.nn.sigmoid(conv_o22, name=scope.name)
                state2 = conv_it2 * conv_ct2
              else:
                conv_ft2 = tf.nn.sigmoid(tf.add(conv_hf22, conv_f22), name=scope.name)
                conv_it2 = tf.nn.sigmoid(tf.add(conv_hi22, conv_i22), name=scope.name)
                conv_ct2 = tf.nn.tanh(tf.add(conv_hc22, conv_c22),    name=scope.name)
                conv_ot2 = tf.nn.sigmoid(tf.add(conv_ho22, conv_o22), name=scope.name)
                state2 = conv_ft2 * c2_tm1 + conv_it2 * conv_ct2
              recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 2 is the xt
              conv_f21 = self._conv2D(h1_tm1,    3,32,32,1, scope='conv_f21', reuse=None)
              conv_i21 = self._conv2D(h1_tm1,    3,32,32,1, scope='conv_i21', reuse=None)
              conv_c21 = self._conv2D(h1_tm1,    3,32,32,1, scope='conv_c21', reuse=None)
              conv_o21 = self._conv2D(h1_tm1,    3,32,32,1, scope='conv_o21', reuse=None)
              conv_f22 = self._conv2D(conv_f21,    3,32,32,1, scope='conv_f22', reuse=None, activation=None)
              conv_i22 = self._conv2D(conv_i21,    3,32,32,1, scope='conv_i22', reuse=None, activation=None)
              conv_c22 = self._conv2D(conv_c21,    3,32,32,1, scope='conv_c22', reuse=None, activation=None)
              conv_o22 = self._conv2D(conv_o21,    3,32,32,1, scope='conv_o22', reuse=None, activation=None)    

              conv_ft2 = tf.nn.sigmoid(conv_f22, name=scope.name)
              conv_it2 = tf.nn.sigmoid(conv_i22, name=scope.name)
              conv_ct2 = tf.nn.tanh(conv_c22,    name=scope.name)
              conv_ot2 = tf.nn.sigmoid(conv_o22, name=scope.name)
              state2 = conv_it2 * conv_ct2
              recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)
                  
            h2_tm1 = recurrent_hidden2
            c2_tm1 = state2
 
                # ConvLSTM block2
        
        # ConvLSTM block3
        with tf.variable_scope('ConvLSTM_3') as scope:
          for i2 in xrange(iter3):
            # LSTM update
            if i2 > 0 or i > 0:
              conv_f31 = self._conv2D(h2_tm1,    3,32,64,2, scope='conv_f31', reuse=scope)
              conv_i31 = self._conv2D(h2_tm1,    3,32,64,2, scope='conv_i31', reuse=scope)
              conv_c31 = self._conv2D(h2_tm1,    3,32,64,2, scope='conv_c31', reuse=scope)
              conv_o31 = self._conv2D(h2_tm1,    3,32,64,2, scope='conv_o31', reuse=scope)
              conv_f32 = self._conv2D(conv_f31,    3,64,64,1, scope='conv_f32', reuse=scope, activation=None)
              conv_i32 = self._conv2D(conv_i31,    3,64,64,1, scope='conv_i32', reuse=scope, activation=None)
              conv_c32 = self._conv2D(conv_c31,    3,64,64,1, scope='conv_c32', reuse=scope, activation=None)
              conv_o32 = self._conv2D(conv_o31,    3,64,64,1, scope='conv_o32', reuse=scope, activation=None)              
              if i2 > 1 or i > 0:
                  conv_hf31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hf31', reuse=scope)
                  conv_hi31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hi31', reuse=scope)
                  conv_hc31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hc31', reuse=scope)
                  conv_ho31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_ho31', reuse=scope)
                  conv_hf32 = self._conv2D(conv_hf31,    3,64,64,1, scope='conv_hf32', reuse=scope, activation=None)
                  conv_hi32 = self._conv2D(conv_hi31,    3,64,64,1, scope='conv_hi32', reuse=scope, activation=None)
                  conv_hc32 = self._conv2D(conv_hc31,    3,64,64,1, scope='conv_hc32', reuse=scope, activation=None)
                  conv_ho32 = self._conv2D(conv_ho31,    3,64,64,1, scope='conv_ho32', reuse=scope, activation=None)
              else:
                  conv_hf31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hf31', reuse=None)
                  conv_hi31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hi31', reuse=None)
                  conv_hc31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hc31', reuse=None)
                  conv_ho31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_ho31', reuse=None)
                  conv_hf32 = self._conv2D(conv_hf31,    3,64,64,1, scope='conv_hf32', reuse=None, activation=None)
                  conv_hi32 = self._conv2D(conv_hi31,    3,64,64,1, scope='conv_hi32', reuse=None, activation=None)
                  conv_hc32 = self._conv2D(conv_hc31,    3,64,64,1, scope='conv_hc32', reuse=None, activation=None)
                  conv_ho32 = self._conv2D(conv_ho31,    3,64,64,1, scope='conv_ho32', reuse=None, activation=None)
              if i2 == 0:
                conv_ft3 = tf.nn.sigmoid(conv_f32, name=scope.name)
                conv_it3 = tf.nn.sigmoid(conv_i32, name=scope.name)
                conv_ct3 = tf.nn.tanh(conv_c32,    name=scope.name)
                conv_ot3 = tf.nn.sigmoid(conv_o32, name=scope.name)
                state3 = conv_it3 * conv_ct3
              else:
                conv_ft3 = tf.nn.sigmoid(tf.add(conv_hf32, conv_f32), name=scope.name)
                conv_it3 = tf.nn.sigmoid(tf.add(conv_hi32, conv_i32), name=scope.name)
                conv_ct3 = tf.nn.tanh(tf.add(conv_hc32, conv_c32),    name=scope.name)
                conv_ot3 = tf.nn.sigmoid(tf.add(conv_ho32, conv_o32), name=scope.name)
                state3 = conv_ft3 * c3_tm1 + conv_it3 * conv_ct3
              recurrent_hidden3 = conv_ot3 * tf.nn.tanh(state3, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 2 is the xt
              conv_f31 = self._conv2D(h2_tm1,    3,32,64,2, scope='conv_f31', reuse=None)
              conv_i31 = self._conv2D(h2_tm1,    3,32,64,2, scope='conv_i31', reuse=None)
              conv_c31 = self._conv2D(h2_tm1,    3,32,64,2, scope='conv_c31', reuse=None)
              conv_o31 = self._conv2D(h2_tm1,    3,32,64,2, scope='conv_o31', reuse=None)
              conv_f32 = self._conv2D(conv_f31,    3,64,64,1, scope='conv_f32', reuse=None, activation=None)
              conv_i32 = self._conv2D(conv_i31,    3,64,64,1, scope='conv_i32', reuse=None, activation=None)
              conv_c32 = self._conv2D(conv_c31,    3,64,64,1, scope='conv_c32', reuse=None, activation=None)
              conv_o32 = self._conv2D(conv_o31,    3,64,64,1, scope='conv_o32', reuse=None, activation=None)    

              conv_ft3 = tf.nn.sigmoid(conv_f32, name=scope.name)
              conv_it3 = tf.nn.sigmoid(conv_i32, name=scope.name)
              conv_ct3 = tf.nn.tanh(conv_c32,    name=scope.name)
              conv_ot3 = tf.nn.sigmoid(conv_o32, name=scope.name)
              state3 = conv_it3 * conv_ct3
              recurrent_hidden3 = conv_ot3 * tf.nn.tanh(state3, name=scope.name)
                  
            h3_tm1 = recurrent_hidden3
            c3_tm1 = state3
 
         # ConvLSTM block2
        with tf.variable_scope('ConvLSTM_4') as scope:
          for i2 in xrange(iter4):
            # LSTM update
            if i2 > 0 or i > 0:
              conv_f41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_f41', reuse=scope)
              conv_i41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_i41', reuse=scope)
              conv_c41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_c41', reuse=scope)
              conv_o41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_o41', reuse=scope)
              conv_f42 = self._conv2D(conv_f41,    3,64,64,1, scope='conv_f42', reuse=scope, activation=None)
              conv_i42 = self._conv2D(conv_i41,    3,64,64,1, scope='conv_i42', reuse=scope, activation=None)
              conv_c42 = self._conv2D(conv_c41,    3,64,64,1, scope='conv_c42', reuse=scope, activation=None)
              conv_o42 = self._conv2D(conv_o41,    3,64,64,1, scope='conv_o42', reuse=scope, activation=None)              
              if i2 > 1 or i > 0:
                  conv_hf41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hf41', reuse=scope)
                  conv_hi41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hi41', reuse=scope)
                  conv_hc41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hc41', reuse=scope)
                  conv_ho41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_ho41', reuse=scope)
                  conv_hf42 = self._conv2D(conv_hf41,    3,64,64,1, scope='conv_hf42', reuse=scope, activation=None)
                  conv_hi42 = self._conv2D(conv_hi41,    3,64,64,1, scope='conv_hi42', reuse=scope, activation=None)
                  conv_hc42 = self._conv2D(conv_hc41,    3,64,64,1, scope='conv_hc42', reuse=scope, activation=None)
                  conv_ho42 = self._conv2D(conv_ho41,    3,64,64,1, scope='conv_ho42', reuse=scope, activation=None)
              else:
                  conv_hf41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hf41', reuse=None)
                  conv_hi41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hi41', reuse=None)
                  conv_hc41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hc41', reuse=None)
                  conv_ho41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_ho41', reuse=None)
                  conv_hf42 = self._conv2D(conv_hf41,    3,64,64,1, scope='conv_hf42', reuse=None, activation=None)
                  conv_hi42 = self._conv2D(conv_hi41,    3,64,64,1, scope='conv_hi42', reuse=None, activation=None)
                  conv_hc42 = self._conv2D(conv_hc41,    3,64,64,1, scope='conv_hc42', reuse=None, activation=None)
                  conv_ho42 = self._conv2D(conv_ho41,    3,64,64,1, scope='conv_ho42', reuse=None, activation=None)
              if i2 == 0:
                conv_ft4 = tf.nn.sigmoid(conv_f42, name=scope.name)
                conv_it4 = tf.nn.sigmoid(conv_i42, name=scope.name)
                conv_ct4 = tf.nn.tanh(conv_c42,    name=scope.name)
                conv_ot4 = tf.nn.sigmoid(conv_o42, name=scope.name)
                state4 = conv_it4 * conv_ct4
              else:
                conv_ft4 = tf.nn.sigmoid(tf.add(conv_hf42, conv_f42), name=scope.name)
                conv_it4 = tf.nn.sigmoid(tf.add(conv_hi42, conv_i42), name=scope.name)
                conv_ct4 = tf.nn.tanh(tf.add(conv_hc42, conv_c42),    name=scope.name)
                conv_ot4 = tf.nn.sigmoid(tf.add(conv_ho42, conv_o42), name=scope.name)
                state4 = conv_ft4 * c4_tm1 + conv_it4 * conv_ct4
              recurrent_hidden4 = conv_ot4 * tf.nn.tanh(state4, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 2 is the xt
              conv_f41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_f41', reuse=None)
              conv_i41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_i41', reuse=None)
              conv_c41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_c41', reuse=None)
              conv_o41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_o41', reuse=None)
              conv_f42 = self._conv2D(conv_f41,    3,64,64,1, scope='conv_f42', reuse=None, activation=None)
              conv_i42 = self._conv2D(conv_i41,    3,64,64,1, scope='conv_i42', reuse=None, activation=None)
              conv_c42 = self._conv2D(conv_c41,    3,64,64,1, scope='conv_c42', reuse=None, activation=None)
              conv_o42 = self._conv2D(conv_o41,    3,64,64,1, scope='conv_o42', reuse=None, activation=None)    

              conv_ft4 = tf.nn.sigmoid(conv_f42, name=scope.name)
              conv_it4 = tf.nn.sigmoid(conv_i42, name=scope.name)
              conv_ct4 = tf.nn.tanh(conv_c42,    name=scope.name)
              conv_ot4 = tf.nn.sigmoid(conv_o42, name=scope.name)
              state4 = conv_it4 * conv_ct4
              recurrent_hidden4 = conv_ot4 * tf.nn.tanh(state4, name=scope.name)
                  
            h4_tm1 = recurrent_hidden4
            c4_tm1 = state4
 
        upsample = tf.image.resize_images(h4_tm1, 32, 32, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)

        with tf.variable_scope('Outer') as scope:
          if i == 0:
            # Result is h2_tm1: 8 x 8 x 64
            #      upsample to: 32 x 32 x 16
            #Naive Upsampleing
            downsample = self._conv2D(upsample, 1,64,16,1, scope='upsample')
          else:
            downsample = self._conv2D(upsample, 1,64,16,1, scope='upsample', reuse=scope)
          ho_tm1 = tf.add(downsample, ho_tm1)
      print (h4_tm1.get_shape())

      net = self._global_avg_pool(h4_tm1)

      print (net.get_shape())

    with tf.variable_scope('logit'):
      logits = self._fully_connected(net, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits, self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()

      tf.scalar_summary('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x, reuse=None):
    with tf.variable_op_scope([x], name, 'BatchNorm', reuse=reuse):
      """Batch normalization."""
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _conv2D(self, x, filter_size, in_filter, out_filter, stride, activation=tf.nn.relu, reuse=None, scope=None):
    with tf.variable_op_scope([x], scope, 'Conv', reuse=reuse):
      # kernel_h, kernel_w = _two_element_tuple(kernel_size)
      # stride_h, stride_w = _two_element_tuple(stride)
      # num_filters_in = inputs.get_shape()[-1]
      # weights_shape = [kernel_h, kernel_w,
      #                  num_filters_in, num_filters_out]
      # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      conv = self._conv('Conv', x, filter_size, in_filter, out_filter, [1,stride,stride,1])
      outputs = self._batch_norm('bn', conv)

      if activation:
        outputs = self._relu(outputs, self.hps.relu_leakiness)
      return outputs

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    n = filter_size * filter_size * out_filters
    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters],
        tf.float32, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

class ConvLSTM_SecondOrder_DeepAvgPool_shorter(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    # Iteration numbers
    iter1 = 4
    iter2 = 4    
    iter3 = 4
    iter4 = 4
    second_order_iter1 = 2
    second_order_iter2 = 2
    print('Shorter Second Order Deep AvgPool Model with: (%d %d %d) (%d %d %d)' % (second_order_iter1, iter1, iter2, second_order_iter2, iter3, iter4))
    inputs = self._images
    scope = ''
    # Core Model
    with tf.op_scope([inputs], scope, 'inception_v3'):
      # Initial Conv
      conv0 = self._conv2D(inputs, 3, 3, 16, 1, scope='conv0')
      ho_tm1 = conv0
      for i in xrange(second_order_iter1):
        # Connect to the next ConvLSTM block
        # ConvLSTM block1
        with tf.variable_scope('ConvLSTM_1') as scope:
          for i1 in xrange(iter1):
            # LSTM update
            if i1 > 0 or i > 0:
              conv_f11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_f11', reuse=scope)
              conv_i11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_i11', reuse=scope)
              conv_c11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_c11', reuse=scope)
              conv_o11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_o11', reuse=scope)
              conv_f12 = self._conv2D(conv_f11,   3,32,32,1, scope='conv_f12', reuse=scope, activation=None)
              conv_i12 = self._conv2D(conv_i11,   3,32,32,1, scope='conv_i12', reuse=scope, activation=None)
              conv_c12 = self._conv2D(conv_c11,   3,32,32,1, scope='conv_c12', reuse=scope, activation=None)
              conv_o12 = self._conv2D(conv_o11,   3,32,32,1, scope='conv_o12', reuse=scope, activation=None)

              if i1 > 1 or i > 0:
                  conv_hf11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hf11', reuse=scope)
                  conv_hi11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hi11', reuse=scope)
                  conv_hc11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hc11', reuse=scope)
                  conv_ho11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_ho11', reuse=scope)
                  conv_hf12 = self._conv2D(conv_hf11,   3,32,32,1, scope='conv_hf12', reuse=scope, activation=None)
                  conv_hi12 = self._conv2D(conv_hi11,   3,32,32,1, scope='conv_hi12', reuse=scope, activation=None)
                  conv_hc12 = self._conv2D(conv_hc11,   3,32,32,1, scope='conv_hc12', reuse=scope, activation=None)
                  conv_ho12 = self._conv2D(conv_ho11,   3,32,32,1, scope='conv_ho12', reuse=scope, activation=None)
              else:
                  conv_hf11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hf11', reuse=None)
                  conv_hi11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hi11', reuse=None)
                  conv_hc11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_hc11', reuse=None)
                  conv_ho11 = self._conv2D(h1_tm1,   3,32,32,1, scope='conv_ho11', reuse=None)
                  conv_hf12 = self._conv2D(conv_hf11,   3,32,32,1, scope='conv_hf12', reuse=None, activation=None)
                  conv_hi12 = self._conv2D(conv_hi11,   3,32,32,1, scope='conv_hi12', reuse=None, activation=None)
                  conv_hc12 = self._conv2D(conv_hc11,   3,32,32,1, scope='conv_hc12', reuse=None, activation=None)
                  conv_ho12 = self._conv2D(conv_ho11,   3,32,32,1, scope='conv_ho12', reuse=None, activation=None)
              if i1 == 0:
                conv_ft1 = tf.nn.sigmoid(conv_f12, name=scope.name)
                conv_it1 = tf.nn.sigmoid(conv_i12, name=scope.name)
                conv_ct1 = tf.nn.tanh(conv_c12,    name=scope.name)
                conv_ot1 = tf.nn.sigmoid(conv_o12, name=scope.name)
                state1 = conv_it1 * conv_ct1
              else:
                conv_ft1 = tf.nn.sigmoid(tf.add(conv_hf12, conv_f12), name=scope.name)
                conv_it1 = tf.nn.sigmoid(tf.add(conv_hi12, conv_i12), name=scope.name)
                conv_ct1 = tf.nn.tanh(tf.add(conv_hc12, conv_c12),    name=scope.name)
                conv_ot1 = tf.nn.sigmoid(tf.add(conv_ho12, conv_o12), name=scope.name)
                state1 = conv_ft1 * c1_tm1 + conv_it1 * conv_ct1
              recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 1 is the xt
              conv_f11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_f11', reuse=None)
              conv_i11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_i11', reuse=None)
              conv_c11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_c11', reuse=None)
              conv_o11 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv_o11', reuse=None)
              conv_f12 = self._conv2D(conv_f11,   3,32,32,1, scope='conv_f12', reuse=None, activation=None)
              conv_i12 = self._conv2D(conv_i11,   3,32,32,1, scope='conv_i12', reuse=None, activation=None)
              conv_c12 = self._conv2D(conv_c11,   3,32,32,1, scope='conv_c12', reuse=None, activation=None)
              conv_o12 = self._conv2D(conv_o11,   3,32,32,1, scope='conv_o12', reuse=None, activation=None)

              conv_ft1 = tf.nn.sigmoid(conv_f12, name=scope.name)
              conv_it1 = tf.nn.sigmoid(conv_i12, name=scope.name)
              conv_ct1 = tf.nn.tanh(conv_c12,    name=scope.name)
              conv_ot1 = tf.nn.sigmoid(conv_o12, name=scope.name)
              state1 = conv_it1 * conv_ct1
              recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)
                  
            h1_tm1 = recurrent_hidden1
            c1_tm1 = state1
 
        # Connect to the next ConvLSTM block
        norm2 = h1_tm1
        
        # ConvLSTM block2
        with tf.variable_scope('ConvLSTM_2') as scope:
          for i2 in xrange(iter2):
            # LSTM update
            if i2 > 0 or i > 0:
              conv_f21 = self._conv2D(norm2,    3,32,32,1, scope='conv_f21', reuse=scope)
              conv_i21 = self._conv2D(norm2,    3,32,32,1, scope='conv_i21', reuse=scope)
              conv_c21 = self._conv2D(norm2,    3,32,32,1, scope='conv_c21', reuse=scope)
              conv_o21 = self._conv2D(norm2,    3,32,32,1, scope='conv_o21', reuse=scope)
              conv_f22 = self._conv2D(conv_f21,    3,32,32,1, scope='conv_f22', reuse=scope, activation=None)
              conv_i22 = self._conv2D(conv_i21,    3,32,32,1, scope='conv_i22', reuse=scope, activation=None)
              conv_c22 = self._conv2D(conv_c21,    3,32,32,1, scope='conv_c22', reuse=scope, activation=None)
              conv_o22 = self._conv2D(conv_o21,    3,32,32,1, scope='conv_o22', reuse=scope, activation=None)              
              if i2 > 1 or i > 0:
                  conv_hf21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hf21', reuse=scope)
                  conv_hi21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hi21', reuse=scope)
                  conv_hc21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hc21', reuse=scope)
                  conv_ho21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_ho21', reuse=scope)
                  conv_hf22 = self._conv2D(conv_hf21,    3,32,32,1, scope='conv_hf22', reuse=scope, activation=None)
                  conv_hi22 = self._conv2D(conv_hi21,    3,32,32,1, scope='conv_hi22', reuse=scope, activation=None)
                  conv_hc22 = self._conv2D(conv_hc21,    3,32,32,1, scope='conv_hc22', reuse=scope, activation=None)
                  conv_ho22 = self._conv2D(conv_ho21,    3,32,32,1, scope='conv_ho22', reuse=scope, activation=None)
              else:
                  conv_hf21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hf21', reuse=None)
                  conv_hi21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hi21', reuse=None)
                  conv_hc21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_hc21', reuse=None)
                  conv_ho21 = self._conv2D(h2_tm1,    3,32,32,1, scope='conv_ho21', reuse=None)
                  conv_hf22 = self._conv2D(conv_hf21,    3,32,32,1, scope='conv_hf22', reuse=None, activation=None)
                  conv_hi22 = self._conv2D(conv_hi21,    3,32,32,1, scope='conv_hi22', reuse=None, activation=None)
                  conv_hc22 = self._conv2D(conv_hc21,    3,32,32,1, scope='conv_hc22', reuse=None, activation=None)
                  conv_ho22 = self._conv2D(conv_ho21,    3,32,32,1, scope='conv_ho22', reuse=None, activation=None)
              if i2 == 0:
                conv_ft2 = tf.nn.sigmoid(conv_f22, name=scope.name)
                conv_it2 = tf.nn.sigmoid(conv_i22, name=scope.name)
                conv_ct2 = tf.nn.tanh(conv_c22,    name=scope.name)
                conv_ot2 = tf.nn.sigmoid(conv_o22, name=scope.name)
                state2 = conv_it2 * conv_ct2
              else:
                conv_ft2 = tf.nn.sigmoid(tf.add(conv_hf22, conv_f22), name=scope.name)
                conv_it2 = tf.nn.sigmoid(tf.add(conv_hi22, conv_i22), name=scope.name)
                conv_ct2 = tf.nn.tanh(tf.add(conv_hc22, conv_c22),    name=scope.name)
                conv_ot2 = tf.nn.sigmoid(tf.add(conv_ho22, conv_o22), name=scope.name)
                state2 = conv_ft2 * c2_tm1 + conv_it2 * conv_ct2
              recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 2 is the xt
              conv_f21 = self._conv2D(norm2,    3,32,32,1, scope='conv_f21', reuse=None)
              conv_i21 = self._conv2D(norm2,    3,32,32,1, scope='conv_i21', reuse=None)
              conv_c21 = self._conv2D(norm2,    3,32,32,1, scope='conv_c21', reuse=None)
              conv_o21 = self._conv2D(norm2,    3,32,32,1, scope='conv_o21', reuse=None)
              conv_f22 = self._conv2D(conv_f21,    3,32,32,1, scope='conv_f22', reuse=None, activation=None)
              conv_i22 = self._conv2D(conv_i21,    3,32,32,1, scope='conv_i22', reuse=None, activation=None)
              conv_c22 = self._conv2D(conv_c21,    3,32,32,1, scope='conv_c22', reuse=None, activation=None)
              conv_o22 = self._conv2D(conv_o21,    3,32,32,1, scope='conv_o22', reuse=None, activation=None)    

              conv_ft2 = tf.nn.sigmoid(conv_f22, name=scope.name)
              conv_it2 = tf.nn.sigmoid(conv_i22, name=scope.name)
              conv_ct2 = tf.nn.tanh(conv_c22,    name=scope.name)
              conv_ot2 = tf.nn.sigmoid(conv_o22, name=scope.name)
              state2 = conv_it2 * conv_ct2
              recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)
                  
            h2_tm1 = recurrent_hidden2
            c2_tm1 = state2

        upsample1 = tf.image.resize_images(h2_tm1, 32, 32, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)

        with tf.variable_scope('Outer1') as scope:
          if i == 0:
            # Result is h2_tm1: 8 x 8 x 64
            #      upsample to: 32 x 32 x 16
            #Naive Upsampleing
            downsample1 = self._conv2D(upsample1, 1,32,16,1, scope='upsample1')
          else:
            downsample1 = self._conv2D(upsample1, 1,32,16,1, scope='upsample1', reuse=scope)
          ho_tm1 = tf.add(downsample1, ho_tm1)

      # 16x16x32
      ho_tm2 = h2_tm1

      for i in xrange(second_order_iter2):
        # Connect to the next ConvLSTM block
        # ConvLSTM block3
        with tf.variable_scope('ConvLSTM_3') as scope:
          for i2 in xrange(iter3):
            # LSTM update
            if i2 > 0 or i > 0:
              conv_f31 = self._conv2D(ho_tm2,    3,32,64,2, scope='conv_f31', reuse=scope)
              conv_i31 = self._conv2D(ho_tm2,    3,32,64,2, scope='conv_i31', reuse=scope)
              conv_c31 = self._conv2D(ho_tm2,    3,32,64,2, scope='conv_c31', reuse=scope)
              conv_o31 = self._conv2D(ho_tm2,    3,32,64,2, scope='conv_o31', reuse=scope)
              conv_f32 = self._conv2D(conv_f31,    3,64,64,1, scope='conv_f32', reuse=scope, activation=None)
              conv_i32 = self._conv2D(conv_i31,    3,64,64,1, scope='conv_i32', reuse=scope, activation=None)
              conv_c32 = self._conv2D(conv_c31,    3,64,64,1, scope='conv_c32', reuse=scope, activation=None)
              conv_o32 = self._conv2D(conv_o31,    3,64,64,1, scope='conv_o32', reuse=scope, activation=None)              
              if i2 > 1 or i > 0:
                  conv_hf31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hf31', reuse=scope)
                  conv_hi31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hi31', reuse=scope)
                  conv_hc31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hc31', reuse=scope)
                  conv_ho31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_ho31', reuse=scope)
                  conv_hf32 = self._conv2D(conv_hf31,    3,64,64,1, scope='conv_hf32', reuse=scope, activation=None)
                  conv_hi32 = self._conv2D(conv_hi31,    3,64,64,1, scope='conv_hi32', reuse=scope, activation=None)
                  conv_hc32 = self._conv2D(conv_hc31,    3,64,64,1, scope='conv_hc32', reuse=scope, activation=None)
                  conv_ho32 = self._conv2D(conv_ho31,    3,64,64,1, scope='conv_ho32', reuse=scope, activation=None)
              else:
                  conv_hf31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hf31', reuse=None)
                  conv_hi31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hi31', reuse=None)
                  conv_hc31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_hc31', reuse=None)
                  conv_ho31 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_ho31', reuse=None)
                  conv_hf32 = self._conv2D(conv_hf31,    3,64,64,1, scope='conv_hf32', reuse=None, activation=None)
                  conv_hi32 = self._conv2D(conv_hi31,    3,64,64,1, scope='conv_hi32', reuse=None, activation=None)
                  conv_hc32 = self._conv2D(conv_hc31,    3,64,64,1, scope='conv_hc32', reuse=None, activation=None)
                  conv_ho32 = self._conv2D(conv_ho31,    3,64,64,1, scope='conv_ho32', reuse=None, activation=None)
              if i2 == 0:
                conv_ft3 = tf.nn.sigmoid(conv_f32, name=scope.name)
                conv_it3 = tf.nn.sigmoid(conv_i32, name=scope.name)
                conv_ct3 = tf.nn.tanh(conv_c32,    name=scope.name)
                conv_ot3 = tf.nn.sigmoid(conv_o32, name=scope.name)
                state3 = conv_it3 * conv_ct3
              else:
                conv_ft3 = tf.nn.sigmoid(tf.add(conv_hf32, conv_f32), name=scope.name)
                conv_it3 = tf.nn.sigmoid(tf.add(conv_hi32, conv_i32), name=scope.name)
                conv_ct3 = tf.nn.tanh(tf.add(conv_hc32, conv_c32),    name=scope.name)
                conv_ot3 = tf.nn.sigmoid(tf.add(conv_ho32, conv_o32), name=scope.name)
                state3 = conv_ft3 * c3_tm1 + conv_it3 * conv_ct3
              recurrent_hidden3 = conv_ot3 * tf.nn.tanh(state3, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 2 is the xt
              conv_f31 = self._conv2D(ho_tm2,    3,32,64,2, scope='conv_f31', reuse=None)
              conv_i31 = self._conv2D(ho_tm2,    3,32,64,2, scope='conv_i31', reuse=None)
              conv_c31 = self._conv2D(ho_tm2,    3,32,64,2, scope='conv_c31', reuse=None)
              conv_o31 = self._conv2D(ho_tm2,    3,32,64,2, scope='conv_o31', reuse=None)
              conv_f32 = self._conv2D(conv_f31,    3,64,64,1, scope='conv_f32', reuse=None, activation=None)
              conv_i32 = self._conv2D(conv_i31,    3,64,64,1, scope='conv_i32', reuse=None, activation=None)
              conv_c32 = self._conv2D(conv_c31,    3,64,64,1, scope='conv_c32', reuse=None, activation=None)
              conv_o32 = self._conv2D(conv_o31,    3,64,64,1, scope='conv_o32', reuse=None, activation=None)    

              conv_ft3 = tf.nn.sigmoid(conv_f32, name=scope.name)
              conv_it3 = tf.nn.sigmoid(conv_i32, name=scope.name)
              conv_ct3 = tf.nn.tanh(conv_c32,    name=scope.name)
              conv_ot3 = tf.nn.sigmoid(conv_o32, name=scope.name)
              state3 = conv_it3 * conv_ct3
              recurrent_hidden3 = conv_ot3 * tf.nn.tanh(state3, name=scope.name)
                  
            h3_tm1 = recurrent_hidden3
            c3_tm1 = state3
 
         # ConvLSTM block2
        with tf.variable_scope('ConvLSTM_4') as scope:
          for i2 in xrange(iter4):
            # LSTM update
            if i2 > 0 or i > 0:
              conv_f41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_f41', reuse=scope)
              conv_i41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_i41', reuse=scope)
              conv_c41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_c41', reuse=scope)
              conv_o41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_o41', reuse=scope)
              conv_f42 = self._conv2D(conv_f41,    3,64,64,1, scope='conv_f42', reuse=scope, activation=None)
              conv_i42 = self._conv2D(conv_i41,    3,64,64,1, scope='conv_i42', reuse=scope, activation=None)
              conv_c42 = self._conv2D(conv_c41,    3,64,64,1, scope='conv_c42', reuse=scope, activation=None)
              conv_o42 = self._conv2D(conv_o41,    3,64,64,1, scope='conv_o42', reuse=scope, activation=None)              
              if i2 > 1 or i > 0:
                  conv_hf41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hf41', reuse=scope)
                  conv_hi41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hi41', reuse=scope)
                  conv_hc41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hc41', reuse=scope)
                  conv_ho41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_ho41', reuse=scope)
                  conv_hf42 = self._conv2D(conv_hf41,    3,64,64,1, scope='conv_hf42', reuse=scope, activation=None)
                  conv_hi42 = self._conv2D(conv_hi41,    3,64,64,1, scope='conv_hi42', reuse=scope, activation=None)
                  conv_hc42 = self._conv2D(conv_hc41,    3,64,64,1, scope='conv_hc42', reuse=scope, activation=None)
                  conv_ho42 = self._conv2D(conv_ho41,    3,64,64,1, scope='conv_ho42', reuse=scope, activation=None)
              else:
                  conv_hf41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hf41', reuse=None)
                  conv_hi41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hi41', reuse=None)
                  conv_hc41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_hc41', reuse=None)
                  conv_ho41 = self._conv2D(h4_tm1,    3,64,64,1, scope='conv_ho41', reuse=None)
                  conv_hf42 = self._conv2D(conv_hf41,    3,64,64,1, scope='conv_hf42', reuse=None, activation=None)
                  conv_hi42 = self._conv2D(conv_hi41,    3,64,64,1, scope='conv_hi42', reuse=None, activation=None)
                  conv_hc42 = self._conv2D(conv_hc41,    3,64,64,1, scope='conv_hc42', reuse=None, activation=None)
                  conv_ho42 = self._conv2D(conv_ho41,    3,64,64,1, scope='conv_ho42', reuse=None, activation=None)
              if i2 == 0:
                conv_ft4 = tf.nn.sigmoid(conv_f42, name=scope.name)
                conv_it4 = tf.nn.sigmoid(conv_i42, name=scope.name)
                conv_ct4 = tf.nn.tanh(conv_c42,    name=scope.name)
                conv_ot4 = tf.nn.sigmoid(conv_o42, name=scope.name)
                state4 = conv_it4 * conv_ct4
              else:
                conv_ft4 = tf.nn.sigmoid(tf.add(conv_hf42, conv_f42), name=scope.name)
                conv_it4 = tf.nn.sigmoid(tf.add(conv_hi42, conv_i42), name=scope.name)
                conv_ct4 = tf.nn.tanh(tf.add(conv_hc42, conv_c42),    name=scope.name)
                conv_ot4 = tf.nn.sigmoid(tf.add(conv_ho42, conv_o42), name=scope.name)
                state4 = conv_ft4 * c4_tm1 + conv_it4 * conv_ct4
              recurrent_hidden4 = conv_ot4 * tf.nn.tanh(state4, name=scope.name)

            else:
              # the first step, initial ctm1 htm1 is zeros
              # norm 2 is the xt
              conv_f41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_f41', reuse=None)
              conv_i41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_i41', reuse=None)
              conv_c41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_c41', reuse=None)
              conv_o41 = self._conv2D(h3_tm1,    3,64,64,1, scope='conv_o41', reuse=None)
              conv_f42 = self._conv2D(conv_f41,    3,64,64,1, scope='conv_f42', reuse=None, activation=None)
              conv_i42 = self._conv2D(conv_i41,    3,64,64,1, scope='conv_i42', reuse=None, activation=None)
              conv_c42 = self._conv2D(conv_c41,    3,64,64,1, scope='conv_c42', reuse=None, activation=None)
              conv_o42 = self._conv2D(conv_o41,    3,64,64,1, scope='conv_o42', reuse=None, activation=None)    

              conv_ft4 = tf.nn.sigmoid(conv_f42, name=scope.name)
              conv_it4 = tf.nn.sigmoid(conv_i42, name=scope.name)
              conv_ct4 = tf.nn.tanh(conv_c42,    name=scope.name)
              conv_ot4 = tf.nn.sigmoid(conv_o42, name=scope.name)
              state4 = conv_it4 * conv_ct4
              recurrent_hidden4 = conv_ot4 * tf.nn.tanh(state4, name=scope.name)
                  
            h4_tm1 = recurrent_hidden4
            c4_tm1 = state4
 

        upsample2 = tf.image.resize_images(h4_tm1, 16, 16, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)

        with tf.variable_scope('Outer2') as scope:
          if i == 0:
            # Result is h2_tm1: 8 x 8 x 64
            #      upsample to: 32 x 32 x 16
            #Naive Upsampleing
            downsample2 = self._conv2D(upsample2, 1,64,32,1, scope='upsample2')
          else:
            downsample2 = self._conv2D(upsample2, 1,64,32,1, scope='upsample2', reuse=scope)
          ho_tm2 = tf.add(downsample2, ho_tm2)

      print (h4_tm1.get_shape())

      net = self._global_avg_pool(h4_tm1)

      print (net.get_shape())

    with tf.variable_scope('logit'):
      logits = self._fully_connected(net, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits, self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()

      tf.scalar_summary('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x, reuse=None):
    with tf.variable_op_scope([x], name, 'BatchNorm', reuse=reuse):
      """Batch normalization."""
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _conv2D(self, x, filter_size, in_filter, out_filter, stride, activation=tf.nn.relu, reuse=None, scope=None):
    with tf.variable_op_scope([x], scope, 'Conv', reuse=reuse):
      # kernel_h, kernel_w = _two_element_tuple(kernel_size)
      # stride_h, stride_w = _two_element_tuple(stride)
      # num_filters_in = inputs.get_shape()[-1]
      # weights_shape = [kernel_h, kernel_w,
      #                  num_filters_in, num_filters_out]
      # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      conv = self._conv('Conv', x, filter_size, in_filter, out_filter, [1,stride,stride,1])
      outputs = self._batch_norm('bn', conv)

      if activation:
        outputs = self._relu(outputs, self.hps.relu_leakiness)
      return outputs

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)
    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    n = filter_size * filter_size * out_filters
    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters],
        tf.float32, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu"""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

class SecondOrder_ConvNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    # Iteration numbers

    second_order_iter = 2
    print('Second Order Naive Model with: %d' % (second_order_iter))
    inputs = self._images
    scope = ''
    # Core Model
    with tf.op_scope([inputs], scope, 'inception_v3'):
      # Initial Conv
      conv0 = self._conv2D(inputs, 3, 3, 16, 1, scope='conv0')
      ho_tm1 = conv0
      for i in xrange(second_order_iter):
        # Connect to the next ConvLSTM block
        with tf.variable_scope('ConvNet_block') as scope:
          if i > 0:
            conv1 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv1', reuse=scope)
            conv2 = self._conv2D(conv1,   3,32,32,1, scope='conv2', reuse=scope)
            conv3 = self._conv2D(conv2,   3,32,64,2, scope='conv3', reuse=scope)
            conv4 = self._conv2D(conv3,   3,64,64,1, scope='conv4', reuse=scope)

          else:
            # the first step, initial ctm1 htm1 is zeros
            # norm 1 is the xt
            conv1 = self._conv2D(ho_tm1,   3,16,32,2, scope='conv1', reuse=None)
            conv2 = self._conv2D(conv1,   3,32,32,1, scope='conv2', reuse=None)
            conv3 = self._conv2D(conv2,   3,32,64,2, scope='conv3', reuse=None)
            conv4 = self._conv2D(conv3,   3,64,64,1, scope='conv4', reuse=None)
                  
        upsample = tf.image.resize_images(conv4, 32, 32, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)

        with tf.variable_scope('Outer') as scope:
          if i == 0:
            # Result is h2_tm1: 8 x 8 x 64
            #      upsample to: 32 x 32 x 16
            #Naive Upsampleing
            downsample = self._conv2D(upsample, 1,64,16,1, scope='upsample')
          else:
            downsample = self._conv2D(upsample, 1,64,16,1, scope='upsample', reuse=scope)
          ho_tm1 = tf.add(downsample, ho_tm1)

      net = self._global_avg_pool(conv4)

      print (net.get_shape())

    with tf.variable_scope('logit'):
      logits = self._fully_connected(net, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits, self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()

      tf.scalar_summary('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x, reuse=None):
    with tf.variable_op_scope([x], name, 'BatchNorm', reuse=reuse):
      """Batch normalization."""
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _conv2D(self, x, filter_size, in_filter, out_filter, stride, activation=tf.nn.relu, reuse=None, scope=None):
    with tf.variable_op_scope([x], scope, 'Conv', reuse=reuse):
      # kernel_h, kernel_w = _two_element_tuple(kernel_size)
      # stride_h, stride_w = _two_element_tuple(stride)
      # num_filters_in = inputs.get_shape()[-1]
      # weights_shape = [kernel_h, kernel_w,
      #                  num_filters_in, num_filters_out]
      # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      conv = self._conv('Conv', x, filter_size, in_filter, out_filter, [1,stride,stride,1])
      outputs = self._batch_norm('bn', conv)

      if activation:
        outputs = self._relu(outputs, self.hps.relu_leakiness)
      return outputs

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    n = filter_size * filter_size * out_filters
    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters],
        tf.float32, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

class ConvLSTM_Baseline_AvgPool(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    # Iteration numbers
    iter1 = 4
    iter2 = 4
    inputs = self._images
    scope = ''
    # Core Model
    with tf.op_scope([inputs], scope, 'inception_v3'):
      # Initial Conv
      conv0 = self._conv2D(inputs, 3, 3, 16, 1, scope='conv0')
      norm1 = conv0

      h1_tm1 = []
      # ConvLSTM block1
      with tf.variable_scope('ConvLSTM_1') as scope:
        for i1 in xrange(iter1):
          # LSTM update
          if i1 > 0:
            conv_f11 = self._conv2D(norm1,  3,16,32,2, scope='conv_f11', reuse=scope)
            conv_i11 = self._conv2D(norm1,  3,16,32,2, scope='conv_i11', reuse=scope)
            conv_c11 = self._conv2D(norm1,  3,16,32,2, scope='conv_c11', reuse=scope)
            conv_o11 = self._conv2D(norm1,  3,16,32,2, scope='conv_o11', reuse=scope)

            conv_f12 = self._conv2D(conv_f11, 3,32,32,1, scope='conv_f12', reuse=scope, activation=None)
            conv_i12 = self._conv2D(conv_i11, 3,32,32,1, scope='conv_i12', reuse=scope, activation=None)
            conv_c12 = self._conv2D(conv_c11, 3,32,32,1, scope='conv_c12', reuse=scope, activation=None)
            conv_o12 = self._conv2D(conv_o11, 3,32,32,1, scope='conv_o12', reuse=scope, activation=None)
          
            if i1 > 1:
                conv_hf11 = self._conv2D(h1_tm1[i1-1], 3,32,32,1, scope='conv_hf11', reuse=scope)
                conv_hi11 = self._conv2D(h1_tm1[i1-1], 3,32,32,1, scope='conv_hi11', reuse=scope)
                conv_hc11 = self._conv2D(h1_tm1[i1-1], 3,32,32,1, scope='conv_hc11', reuse=scope)
                conv_ho11 = self._conv2D(h1_tm1[i1-1], 3,32,32,1, scope='conv_ho11', reuse=scope)

                conv_hf12 = self._conv2D(conv_hf11, 3,32,32,1, scope='conv_hf12', reuse=scope, activation=None)
                conv_hi12 = self._conv2D(conv_hi11, 3,32,32,1, scope='conv_hi12', reuse=scope, activation=None)
                conv_hc12 = self._conv2D(conv_hc11, 3,32,32,1, scope='conv_hc12', reuse=scope, activation=None)
                conv_ho12 = self._conv2D(conv_ho11, 3,32,32,1, scope='conv_ho12', reuse=scope, activation=None)
            else:
                conv_hf11 = self._conv2D(h1_tm1[i1-1],    3,32,32,1, scope='conv_hf11', reuse=None)
                conv_hi11 = self._conv2D(h1_tm1[i1-1],    3,32,32,1, scope='conv_hi11', reuse=None)
                conv_hc11 = self._conv2D(h1_tm1[i1-1],    3,32,32,1, scope='conv_hc11', reuse=None)
                conv_ho11 = self._conv2D(h1_tm1[i1-1],    3,32,32,1, scope='conv_ho11', reuse=None)

                conv_hf12 = self._conv2D(conv_hf11, 3,32,32,1, scope='conv_hf12', reuse=None, activation=None)
                conv_hi12 = self._conv2D(conv_hi11, 3,32,32,1, scope='conv_hi12', reuse=None, activation=None)
                conv_hc12 = self._conv2D(conv_hc11, 3,32,32,1, scope='conv_hc12', reuse=None, activation=None)
                conv_ho12 = self._conv2D(conv_ho11, 3,32,32,1, scope='conv_ho12', reuse=None, activation=None)

            conv_ft1 = tf.nn.sigmoid(tf.add(conv_hf12, conv_f12), name=scope.name)
            conv_it1 = tf.nn.sigmoid(tf.add(conv_hi12, conv_i12), name=scope.name)
            conv_ct1 = tf.nn.tanh(tf.add(conv_hc12, conv_c12),    name=scope.name)
            conv_ot1 = tf.nn.sigmoid(tf.add(conv_ho12, conv_o12), name=scope.name)
            state1 = conv_ft1 * c1_tm1 + conv_it1 * conv_ct1
            recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)

          else:
            # the first step, initial ctm1 htm1 is zeros
            # norm 1 is the xt
            conv_f11 = self._conv2D(norm1,    3,16,32,2, scope='conv_f11')
            conv_i11 = self._conv2D(norm1,    3,16,32,2, scope='conv_i11')
            conv_c11 = self._conv2D(norm1,    3,16,32,2, scope='conv_c11')
            conv_o11 = self._conv2D(norm1,    3,16,32,2, scope='conv_o11')

            conv_f12 = self._conv2D(conv_f11, 3,32,32,1, scope='conv_f12', activation=None)
            conv_i12 = self._conv2D(conv_i11, 3,32,32,1, scope='conv_i12', activation=None)
            conv_c12 = self._conv2D(conv_c11, 3,32,32,1, scope='conv_c12', activation=None)
            conv_o12 = self._conv2D(conv_o11, 3,32,32,1, scope='conv_o12', activation=None)

            conv_ft1 = tf.nn.sigmoid(conv_f12, name=scope.name)
            conv_it1 = tf.nn.sigmoid(conv_i12, name=scope.name)
            conv_ct1 = tf.nn.tanh(conv_c12,    name=scope.name)
            conv_ot1 = tf.nn.sigmoid(conv_o12, name=scope.name)
            state1 = conv_it1 * conv_ct1
            recurrent_hidden1 = conv_ot1 * tf.nn.tanh(state1, name=scope.name)
                
          h1_tm1.append(recurrent_hidden1)
          c1_tm1 = state1

      # Connect to the next ConvLSTM block
      norm2 = h1_tm1
      
      # ConvLSTM block2
      h2_tm1 = []
      with tf.variable_scope('ConvLSTM_2') as scope:
        for i2 in xrange(iter2):
          # LSTM update
          if i2 > 0:
            conv_f21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_f21', reuse=scope)
            conv_i21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_i21', reuse=scope)
            conv_c21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_c21', reuse=scope)
            conv_o21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_o21', reuse=scope)

            conv_f22 = self._conv2D(conv_f21, 3,64,64,1, scope='conv_f22', reuse=scope, activation=None)
            conv_i22 = self._conv2D(conv_i21, 3,64,64,1, scope='conv_i22', reuse=scope, activation=None)
            conv_c22 = self._conv2D(conv_c21, 3,64,64,1, scope='conv_c22', reuse=scope, activation=None)
            conv_o22 = self._conv2D(conv_o21, 3,64,64,1, scope='conv_o22', reuse=scope, activation=None)
          
            if i2 > 1:
                conv_hf21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hf21', reuse=scope)
                conv_hi21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hi21', reuse=scope)
                conv_hc21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hc21', reuse=scope)
                conv_ho21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_ho21', reuse=scope)

                conv_hf22 = self._conv2D(conv_hf21, 3,64,64,1, scope='conv_hf22', reuse=scope, activation=None)
                conv_hi22 = self._conv2D(conv_hi21, 3,64,64,1, scope='conv_hi22', reuse=scope, activation=None)
                conv_hc22 = self._conv2D(conv_hc21, 3,64,64,1, scope='conv_hc22', reuse=scope, activation=None)
                conv_ho22 = self._conv2D(conv_ho21, 3,64,64,1, scope='conv_ho22', reuse=scope, activation=None)
            else:
                conv_hf21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hf21', reuse=None)
                conv_hi21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hi21', reuse=None)
                conv_hc21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_hc21', reuse=None)
                conv_ho21 = self._conv2D(h2_tm1[i2-1],    3,64,64,1, scope='conv_ho21', reuse=None)

                conv_hf22 = self._conv2D(conv_hf21, 3,64,64,1, scope='conv_hf22', reuse=None, activation=None)
                conv_hi22 = self._conv2D(conv_hi21, 3,64,64,1, scope='conv_hi22', reuse=None, activation=None)
                conv_hc22 = self._conv2D(conv_hc21, 3,64,64,1, scope='conv_hc22', reuse=None, activation=None)
                conv_ho22 = self._conv2D(conv_ho21, 3,64,64,1, scope='conv_ho22', reuse=None, activation=None)


            conv_ft2 = tf.nn.sigmoid(tf.add(conv_hf22, conv_f22), name=scope.name)
            conv_it2 = tf.nn.sigmoid(tf.add(conv_hi22, conv_i22), name=scope.name)
            conv_ct2 = tf.nn.tanh(tf.add(conv_hc22, conv_c22),    name=scope.name)
            conv_ot2 = tf.nn.sigmoid(tf.add(conv_ho22, conv_o22), name=scope.name)
            state2 = conv_ft2 * c2_tm1 + conv_it2 * conv_ct2
            recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)

          else:
            # the first step, initial ctm1 htm1 is zeros
            # norm 2 is the xt
            conv_f21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_f21')
            conv_i21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_i21')
            conv_c21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_c21')
            conv_o21 = self._conv2D(norm2[i2],    3,32,64,2, scope='conv_o21')

            conv_f22 = self._conv2D(conv_f21, 3,64,64,1, scope='conv_f22', activation=None)
            conv_i22 = self._conv2D(conv_i21, 3,64,64,1, scope='conv_i22', activation=None)
            conv_c22 = self._conv2D(conv_c21, 3,64,64,1, scope='conv_c22', activation=None)
            conv_o22 = self._conv2D(conv_o21, 3,64,64,1, scope='conv_o22', activation=None)

            conv_ft2 = tf.nn.sigmoid(conv_f22, name=scope.name)
            conv_it2 = tf.nn.sigmoid(conv_i22, name=scope.name)
            conv_ct2 = tf.nn.tanh(conv_c22,    name=scope.name)
            conv_ot2 = tf.nn.sigmoid(conv_o22, name=scope.name)
            state2 = conv_it2 * conv_ct2
            recurrent_hidden2 = conv_ot2 * tf.nn.tanh(state2, name=scope.name)
                
          h2_tm1.append(recurrent_hidden2)
          c2_tm1 = state2

      net = []
      # Avg Pooling
      for i in xrange(iter2):
        print(h2_tm1[i].get_shape())
        net.append(self._global_avg_pool(h2_tm1[i]))
      print(net[0].get_shape())

    x = tf.reshape(net[0], [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], self.hps.num_classes],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [self.hps.num_classes],
                        initializer=tf.constant_initializer())

    predictions = []
    logits = []
    for i in xrange(iter2):
      x = tf.reshape(net[i], [self.hps.batch_size, -1])
      logits.append(tf.nn.xw_plus_b(x, w, b))
      predictions.append(tf.nn.softmax(logits[i]))

    self.predictions = predictions[iter2-1]
    self.cost = 0
    for i in xrange(iter2):
      with tf.variable_scope('costs'):
        xent = tf.nn.softmax_cross_entropy_with_logits(
            logits[i], self.labels)
        self.cost += tf.reduce_mean(xent, name='xent')
        self.cost += self._decay()

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x, reuse=None):
    with tf.variable_op_scope([x], name, 'BatchNorm', reuse=reuse):
      """Batch normalization."""
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _conv2D(self, x, filter_size, in_filter, out_filter, stride, activation=tf.nn.relu, reuse=None, scope=None):
    with tf.variable_op_scope([x], scope, 'Conv', reuse=reuse):
      # kernel_h, kernel_w = _two_element_tuple(kernel_size)
      # stride_h, stride_w = _two_element_tuple(stride)
      # num_filters_in = inputs.get_shape()[-1]
      # weights_shape = [kernel_h, kernel_w,
      #                  num_filters_in, num_filters_out]
      # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
      conv = self._conv('Conv', x, filter_size, in_filter, out_filter, [1,stride,stride,1])
      outputs = self._batch_norm('bn', conv)

      if activation:
        outputs = self._relu(outputs, self.hps.relu_leakiness)
      return outputs

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    n = filter_size * filter_size * out_filters
    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters],
        tf.float32, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
