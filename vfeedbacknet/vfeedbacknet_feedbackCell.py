import tensorflow as tf

class FeedbackLSTMCell_stack1(tf.nn.rnn_cell.RNNCell):
    '''
    A feedback cell based on a convLSTM structure with a resnet-like set of 
    convolutions. 
    '''
    
    
    def __init__(self, input_shape, feedback_iterations,
                 forget_bias=1.0,
                 activation=tf.tanh,
                 is_training=True,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=None):

        super().__init__(_reuse=None)

        self._kernel = [3, 3]
        self._input_shape = input_shape
        self._feedback_iterations = feedback_iterations

        self._forget_bias = forget_bias
        self._activation = activation

        self._is_training = is_training

        with tf.variable_scope('feedback_cell'):
            with tf.variable_scope('rnn'):
                with tf.variable_scope('convlstm'):

                    input_nchannels = input_shape[-1]

                    kernel_size = self._kernel + [input_nchannels, input_nchannels]

                    self.W_xf = tf.get_variable('W_xf', kernel_size, initializer=initializer, regularizer=regularizer)
                    self.W_xi = tf.get_variable('W_xi', kernel_size, initializer=initializer, regularizer=regularizer)
                    self.W_xc = tf.get_variable('W_xc', kernel_size, initializer=initializer, regularizer=regularizer)
                    self.W_xo = tf.get_variable('W_xo', kernel_size, initializer=initializer, regularizer=regularizer)

                    self.W_hf = tf.get_variable('W_hf', kernel_size, initializer=initializer, regularizer=regularizer)
                    self.W_hi = tf.get_variable('W_hi', kernel_size, initializer=initializer, regularizer=regularizer)
                    self.W_hc = tf.get_variable('W_hc', kernel_size, initializer=initializer, regularizer=regularizer)
                    self.W_ho = tf.get_variable('W_ho', kernel_size, initializer=initializer, regularizer=regularizer)

                    self.W_cf = tf.get_variable('W_cf', input_shape, initializer=initializer, regularizer=regularizer)
                    self.W_ci = tf.get_variable('W_ci', input_shape, initializer=initializer, regularizer=regularizer)
                    self.W_co = tf.get_variable('W_co', input_shape, initializer=initializer, regularizer=regularizer)

                    self.b_f = tf.get_variable('b_f', [input_nchannels], initializer=initializer, regularizer=regularizer)
                    self.b_i = tf.get_variable('b_i', [input_nchannels], initializer=initializer, regularizer=regularizer)
                    self.b_c = tf.get_variable('b_c', [input_nchannels], initializer=initializer, regularizer=regularizer)
                    self.b_o = tf.get_variable('b_o', [input_nchannels], initializer=initializer, regularizer=regularizer)

                    # self._x_kernel1 = tf.get_variable('x_kernel1', conv_kernel1_size, initializer=initializer, regularizer=regularizer)
                    # self._x_bias1 = tf.get_variable('x_bias1', [conv_nchannels], initializer=initializer, regularizer=regularizer)

                    # self._h_kernel1 = tf.get_variable('h_kernel1', conv_kernel1_size, initializer=initializer, regularizer=regularizer)

                    # self._W_ci = tf.get_variable('W_ci', input_shape, initializer=initializer, regularizer=regularizer)
                    # self._W_cf = tf.get_variable('W_cf', input_shape, initializer=initializer, regularizer=regularizer)
                    # self._W_co = tf.get_variable('W_co', input_shape, initializer=initializer, regularizer=regularizer)

                    
    @property
    def defined_variables(self):

        return [self.W_xf, self.W_xi, self.W_xc, self.W_xo,
                self.W_hf, self.W_hi, self.W_hc, self.W_ho,
                self.W_cf, self.W_ci, self.W_co,
                self.b_f, self.b_i, self.b_c, self.b_o]

    
    @property
    def state_size(self):

        return tf.nn.rnn_cell.LSTMStateTuple(tf.TensorShape(self._input_shape), tf.TensorShape(self._input_shape))

    
    @property
    def output_size(self):
        
        return tf.TensorShape(self._input_shape)

    
    def apply_layer(self, x, sequence_length=None, initial_state=None, var_list=None):
        '''
        Input should be a python list of length `feedback_iterations`. It should
        contain tensors with `input_shape`.
        '''

        assert len(x) == self._feedback_iterations, 'input should be {} elements long, but was {}'.format(self._feedback_iterations, len(x))
        
        # add variables to var_list for model exporting
        if var_list is not None:
            for var in self.defined_variables:
                if var not in var_list:
                    var_list.append(var)

        with tf.variable_scope('feedback_cell', reuse=True):

            outputs, state = tf.nn.static_rnn(
                self,
                x,
                dtype=tf.float32,
                sequence_length=None,
                initial_state=initial_state,
            )
        
        return outputs
    
    
    def call(self, inputs, state):
        '''
        See equation 3 from https://arxiv.org/pdf/1506.04214.pdf
        '''

        cell_state, hidden_state = state

        i_t = tf.sigmoid(
            tf.nn.bias_add(
                tf.nn.conv2d(inputs, self.W_xi, [1, 1, 1, 1], padding='SAME')  +
                tf.nn.conv2d(hidden_state, self.W_hi, [1, 1, 1, 1], padding='SAME') +
                tf.multiply(cell_state, self.W_ci, name='element_wise_multipy'),
                self.b_i)
        )
        #i_t = tf.contrib.layers.layer_norm(i_t)

        f_t = tf.sigmoid(
            tf.nn.bias_add(
                tf.nn.conv2d(inputs, self.W_xf, [1, 1, 1, 1], padding='SAME')  +
                tf.nn.conv2d(hidden_state, self.W_hf, [1, 1, 1, 1], padding='SAME') +
                tf.multiply(cell_state, self.W_cf, name='element_wise_multipy_ft'),
                self.b_f)
        )
        #f_t = tf.contrib.layers.layer_norm(f_t)

        j = tf.nn.bias_add(tf.nn.conv2d(inputs, self.W_xc, [1, 1, 1, 1], padding='SAME')  +
                           tf.nn.conv2d(hidden_state, self.W_hc, [1, 1, 1, 1], padding='SAME'),
                           self.b_c)
        #j = tf.contrib.layers.layer_norm(j)

        new_cell_state = (tf.multiply(f_t, cell_state, name='element_wise_multipy_ct1') + 
                         tf.multiply(i_t, tf.tanh( j ), name='element_wise_multipy_ct2'))
                         
        #new_cell_state = tf.contrib.layers.layer_norm(new_cell_state)

        o_t = tf.sigmoid( 
            tf.nn.bias_add(
                tf.nn.conv2d(inputs, self.W_xo, [1, 1, 1, 1], padding='SAME')  +
                tf.nn.conv2d(hidden_state, self.W_ho, [1, 1, 1, 1], padding='SAME') +
                tf.multiply(new_cell_state, self.W_co, name='element_wise_multipy_ot'), 
                self.b_o)
        )

        #o_t = tf.contrib.layers.layer_norm(o_t)

        new_hidden_state = tf.multiply(o_t, tf.tanh(new_cell_state), name='element_wise_multipy_it')

        state = tf.nn.rnn_cell.LSTMStateTuple(new_cell_state, new_hidden_state)

        return new_hidden_state, state

    
class FeedbackLSTMCell_stack2(tf.nn.rnn_cell.RNNCell):
    pass

# class FeedbackLSTMCell_stack2(tf.nn.rnn_cell.RNNCell):
#     '''
#     A feedback cell based on a convLSTM structure with a resnet-like set of 
#     convolutions. 
#     '''
    
    
#     def __init__(self, input_shape, feedback_iterations,
#                  forget_bias=1.0,
#                  activation=tf.tanh,
#                  is_training=True,
#                  initializer=tf.contrib.layers.xavier_initializer(),
#                  regularizer=None):

#         super().__init__(_reuse=None)

#         self._kernel = [3, 3]
#         self._input_shape = input_shape
#         self._feedback_iterations = feedback_iterations

#         self._forget_bias = forget_bias
#         self._activation = activation

#         self._is_training = is_training

#         with tf.variable_scope('feedback_cell'):
#             with tf.variable_scope('rnn'):
#                 with tf.variable_scope('convlstm'):

#                     input_nchannels = input_shape[-1]
#                     conv_nchannels = 4 * input_nchannels

#                     conv_kernel1_size = self._kernel + [input_nchannels, conv_nchannels]
#                     conv_kernel2_size = self._kernel + [input_nchannels, input_nchannels]

#                     self._x_kernel1 = tf.get_variable('x_kernel1', conv_kernel1_size, initializer=initializer, regularizer=regularizer)
#                     self._x_bias1 = tf.get_variable('x_bias1', [conv_nchannels], initializer=initializer, regularizer=regularizer)

#                     self._xi_kernel2 = tf.get_variable('xi_kernel2', conv_kernel2_size, initializer=initializer, regularizer=regularizer)
#                     self._xi_bias2 = tf.get_variable('xi_bias2', [input_nchannels], initializer=initializer, regularizer=regularizer)

#                     self._xf_kernel2 = tf.get_variable('xf_kernel2', conv_kernel2_size, initializer=initializer, regularizer=regularizer)
#                     self._xf_bias2 = tf.get_variable('xf_bias2', [input_nchannels], initializer=initializer, regularizer=regularizer)

#                     self._xc_kernel2 = tf.get_variable('xc_kernel2', conv_kernel2_size, initializer=initializer, regularizer=regularizer)
#                     self._xc_bias2 = tf.get_variable('xc_bias2', [input_nchannels], initializer=initializer, regularizer=regularizer)

#                     self._xo_kernel2 = tf.get_variable('xo_kernel2', conv_kernel2_size, initializer=initializer, regularizer=regularizer)
#                     self._xo_bias2 = tf.get_variable('xo_bias2', [input_nchannels], initializer=initializer, regularizer=regularizer)

#                     self._h_kernel1 = tf.get_variable('h_kernel1', conv_kernel1_size, initializer=initializer, regularizer=regularizer)

#                     self._W_ci = tf.get_variable('W_ci', input_shape, initializer=initializer, regularizer=regularizer)
#                     self._W_cf = tf.get_variable('W_cf', input_shape, initializer=initializer, regularizer=regularizer)
#                     self._W_co = tf.get_variable('W_co', input_shape, initializer=initializer, regularizer=regularizer)

                    
#     @property
#     def defined_variables(self):

#         return [self._x_kernel1, self._x_bias1,
#                 self._xi_kernel2, self._xf_kernel2, self._xc_kernel2, self._xo_kernel2,
#                 self._xi_bias2, self._xf_bias2, self._xc_bias2, self._xo_bias2,                
#                 self._h_kernel1, self._W_ci, self._W_cf, self._W_co]

    
#     @property
#     def state_size(self):

#         return tf.nn.rnn_cell.LSTMStateTuple(tf.TensorShape(self._input_shape), tf.TensorShape(self._input_shape))

    
#     @property
#     def output_size(self):
        
#         return tf.TensorShape(self._input_shape)

    
#     def apply_layer(self, x, sequence_length=None, initial_state=None, var_list=None):
#         '''
#         Input should be a python list of length `feedback_iterations`. It should
#         contain tensors with `input_shape`.
#         '''

#         assert len(x) == self._feedback_iterations, 'input should be {} elements long, but was {}'.format(self._feedback_iterations, len(x))
        
#         # add variables to var_list for model exporting
#         if var_list is not None:
#             for var in self.defined_variables:
#                 if var not in var_list:
#                     var_list.append(var)

#         with tf.variable_scope('feedback_cell', reuse=True):

#             outputs, state = tf.nn.static_rnn(
#                 self,
#                 x,
#                 dtype=tf.float32,
#                 sequence_length=None,
#                 initial_state=initial_state,
#             )
        
#         return outputs
    
    
#     def call(self, x_t, state):
#         '''
#         See equation 3 from https://arxiv.org/pdf/1506.04214.pdf
#         '''
        
#         c_tm1, h_tm1 = state
        
#         # skip_x_t = tf.contrib.layers.layer_norm(x_t, trainable=self._is_training)
#         # skip_h_tm1 = tf.contrib.layers.layer_norm(h_tm1, trainable=self._is_training)
#         skip_x_t = x_t
#         skip_h_tm1 = h_tm1
        
#         # -- basic resnet block --
#         # compute x (2-conv resnet block)
#         _x_t = tf.nn.conv2d(skip_x_t, self._x_kernel1, [1, 1, 1, 1], padding='SAME')
#         _x_t = tf.nn.relu( tf.nn.bias_add(_x_t, self._x_bias1) )

#         xi_t, xf_t, xc_t, xo_t = tf.split(_x_t, 4, axis=-1)
#         # xi_t = tf.contrib.layers.layer_norm(xi_t, trainable=self._is_training)
#         # xf_t = tf.contrib.layers.layer_norm(xf_t, trainable=self._is_training)
#         # xc_t = tf.contrib.layers.layer_norm(xc_t, trainable=self._is_training)
#         # xo_t = tf.contrib.layers.layer_norm(xo_t, trainable=self._is_training)
        
#         xi_t = tf.nn.conv2d(xi_t, self._xi_kernel2, [1, 1, 1, 1], padding='SAME')
#         xi_t = tf.nn.bias_add(xi_t + skip_x_t, self._xi_bias2)
#         xf_t = tf.nn.conv2d(xf_t, self._xf_kernel2, [1, 1, 1, 1], padding='SAME')
#         xf_t = tf.nn.bias_add(xf_t + skip_x_t, self._xf_bias2)
#         xc_t = tf.nn.conv2d(xc_t, self._xc_kernel2, [1, 1, 1, 1], padding='SAME')
#         xc_t = tf.nn.bias_add(xc_t + skip_x_t, self._xc_bias2)
#         xo_t = tf.nn.conv2d(xo_t, self._xo_kernel2, [1, 1, 1, 1], padding='SAME')
#         xo_t = tf.nn.bias_add(xo_t + skip_x_t, self._xo_bias2)
#         # -- end resnet block --
        
#         # compute h (single conv)
#         _h_tm1 = tf.nn.conv2d(skip_h_tm1, self._h_kernel1, [1, 1, 1, 1], padding='SAME')

#         hi_t, hf_t, hc_t, ho_t = tf.split(_h_tm1, 4, axis=-1)

#         # compute the components of c needed for i and f
#         ci_t = self._W_ci * c_tm1
#         cf_t = self._W_cf * c_tm1

#         # compute i
#         i_preactivation = xi_t + hi_t + ci_t
#         i_t = tf.sigmoid(i_preactivation)
#         # i_t = tf.contrib.layers.layer_norm(i_t, trainable=self._is_training)

#         # compute f
#         f_preactivation = xf_t + hf_t + cf_t
#         f_t = tf.sigmoid(f_preactivation + self._forget_bias)
#         # f_t = tf.contrib.layers.layer_norm(f_t, trainable=self._is_training)

#         # compute c
#         c_t = c_tm1 * f_t + i_t * self._activation(xc_t + hc_t)

#         # compute o
#         co_t = self._W_co * c_tm1
#         o_preactivation = xo_t + ho_t + co_t
#         o_t = self._activation(o_preactivation)
#         # o_t = tf.contrib.layers.layer_norm(o_t)

#         # compute h
#         h_t = o_t + self._activation(c_t)

#         state = tf.nn.rnn_cell.LSTMStateTuple(c_t, h_t)

#         return h_t, state
