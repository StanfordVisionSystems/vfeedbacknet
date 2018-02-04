import tensorflow as tf
import sys

def feedback_dynamic_rnn(cell, inputs, feedback_states, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):    

    # assert isinstance(inputs, list), 'inputs must be a list of tensors'
    # assert isinstance(feedback_states, list), 'feedback_states must be a list of tensors'
    
    x = tf.concat([inputs, feedback_states], axis=-1)
    
    outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=sequence_length, initial_state=initial_state,
                                    dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                    time_major=time_major, scope=scope)
    
    #outputs = tf.split(outputs, 2, axis=-1)
    #lstm_outputs = outputs[0]
    #feedback_states = outputs[1]
    lstm_outputs = outputs
    feedback_states = None
    
    lstm_states = states
    #print(outputs)
        
    return lstm_outputs, feedback_states, lstm_states


class FeedbackConvLSTMCell_v1(tf.nn.rnn_cell.RNNCell):

    
    def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True, data_format='channels_last', is_training=True):

        super(FeedbackConvLSTMCell_v1, self).__init__()

        self._kernel = kernel
        self._filters = filters
        self._forget_bias = forget_bias
        self._activation = activation
        self._normalize = normalize
        self._peephole = peephole

        if data_format == 'channels_last':
            self._size = tf.TensorShape(shape + [self._filters])
            self._feature_axis = self._size.ndims
            self._data_format = None
        elif data_format == 'channels_first':
            raise ValueError('Not yet implemeneted')
            # self._size = tf.TensorShape([self._filters] + shape)
            # self._feature_axis = 0
            # self._data_format = 'NC'
        else:
            raise ValueError('Unknown data_format')

        # TODO(jremmons) fix this.
        self.is_training = True #is_training
      
    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)
 
    @property
    def output_size(self):
        return self._size
 
    def call(self, x, state):
        c, h = state

        x, fb_state = tf.split(x, 2, axis=-1)

        x = tf.concat([x, fb_state, h], axis=self._feature_axis)
        n = x.shape[-1].value
        m = 4 * self._filters if self._filters > 1 else 4

        with tf.variable_scope('convlstm', reuse=True):
            W = tf.get_variable('kernel', self._kernel + [n, m])
            y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
        if not self._normalize:
            with tf.variable_scope('convlstm', reuse=True):
                y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
        j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

        if self._peephole:
            with tf.variable_scope('convlstm', reuse=True):
                i += tf.get_variable('W_ci', c.shape[1:]) * c
                f += tf.get_variable('W_cf', c.shape[1:]) * c

        if self._normalize:
            j = tf.contrib.layers.layer_norm(j, trainable=self.is_training)
            i = tf.contrib.layers.layer_norm(i, trainable=self.is_training)
            f = tf.contrib.layers.layer_norm(f, trainable=self.is_training)

        f = tf.sigmoid(f + self._forget_bias)
        i = tf.sigmoid(i)
        c = c * f + i * self._activation(j)

        if self._peephole:
            with tf.variable_scope('convlstm', reuse=True):
                o += tf.get_variable('W_co', c.shape[1:]) * c

        if self._normalize:
            o = tf.contrib.layers.layer_norm(o, trainable=self.is_training)
            c = tf.contrib.layers.layer_norm(c, trainable=self.is_training)

        o = tf.sigmoid(o)
        h = o * self._activation(c)

        # TODO 
        #tf.summary.histogram('forget_gate', f)
        #tf.summary.histogram('input_gate', i)
        #tf.summary.histogram('output_gate', o)
        #tf.summary.histogram('cell_state', c)
        
        #state = tf.nn.rnn_cell.LSTMStateTuple(fb_state, c, h)
        #state = (fb_state, c, h)
        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
        #output = tf.concat([h, fb_state], axis=-1)
        output = h
        
        return output, state


# class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
#   """A GRU cell with convolutions instead of multiplications."""

#   def __init__(self, shape, filters, kernel, initializer=tf.contrib.layers.xavier_initializer(), activation=tf.tanh, normalize=True, data_format='channels_last', reuse=None):
#     super(ConvGRUCell, self).__init__(_reuse=reuse)
#     self._filters = filters
#     self._kernel = kernel
#     self._initializer = initializer
#     self._activation = activation
#     self._normalize = normalize
#     if data_format == 'channels_last':
#         self._size = tf.TensorShape(shape + [self._filters])
#         self._feature_axis = self._size.ndims
#         self._data_format = None
#     elif data_format == 'channels_first':
#         self._size = tf.TensorShape([self._filters] + shape)
#         self._feature_axis = 0
#         self._data_format = 'NC'
#     else:
#         raise ValueError('Unknown data_format')

#   @property
#   def state_size(self):
#     return self._size

#   @property
#   def output_size(self):
#     return self._size

#   def call(self, x, h):
#     channels = x.shape[self._feature_axis].value

#     with tf.variable_scope('gates'):
#       inputs = tf.concat([x, h], axis=self._feature_axis)
#       n = channels + self._filters
#       m = 2 * self._filters if self._filters > 1 else 2
#       W = tf.get_variable('kernel', self._kernel + [n, m])
#       y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
#       if self._normalize:
#         r, u = tf.split(y, 2, axis=self._feature_axis)
#         r = tf.contrib.layers.layer_norm(r)
#         u = tf.contrib.layers.layer_norm(u)
#       else:
#         y += tf.get_variable('bias', [m], initializer=tf.ones_initializer())
#         r, u = tf.split(y, 2, axis=self._feature_axis)
#       r, u = tf.sigmoid(r), tf.sigmoid(u)

#       # TODO
#       #tf.summary.histogram('reset_gate', r)
#       #tf.summary.histogram('update_gate', u)

#     with tf.variable_scope('candidate'):
#       inputs = tf.concat([x, r * h], axis=self._feature_axis)
#       n = channels + self._filters
#       m = self._filters
#       W = tf.get_variable('kernel', self._kernel + [n, m])
#       y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
#       if self._normalize:
#         y = tf.contrib.layers.layer_norm(y)
#       else:
#         y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
#       h = u * h + (1 - u) * self._activation(y)

#     return h, h
