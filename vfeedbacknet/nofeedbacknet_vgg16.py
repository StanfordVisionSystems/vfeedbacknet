import vfeedbacknet.vgg16_model as vgg16_model
import vfeedbacknet.convLSTM as convLSTM

import tensorflow as tf
import logging

logging.basicConfig(level=logging.DEBUG)

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

class ModelLogger:
    '''
    logging utility for debugging the construction of the CNN
    '''

    count = {}
    @staticmethod
    def log(var_name, var):
        if var_name in ModelLogger.count.keys():
            ModelLogger.count[var_name] += 1
            ModelLogger._log(var_name, var)
        else:
            ModelLogger.count[var_name] = 0
            ModelLogger._log(var_name, var)

    @staticmethod        
    def _log(var_name, var):
        maxwidth = 15
        padding = 4
        
        n = var_name[0:maxwidth]
        c = str(ModelLogger.count[var_name])
        p = ' ' * (maxwidth + padding - len(n) - len(c))

        if type(var) == list:
            logging.debug('{}-{}:{}{}x{}'.format(n, c, p, len(var), var[0].shape))
        else:
            logging.debug('{}-{}:{}{}'.format(n, c, p, var.shape))

def nofeedbacknet_vgg16(hparams):
    
    def model_generator(inputs, num_classes, vgg16_weights, fine_tune_vgg16=False, is_training=True):
        '''
        inputs: A tensor fo size [batch, video_length, video_height, video_width, channels]
        '''
        with tf.variable_scope('nofeedbacknet_vgg16'):
            ModelLogger.log('input', inputs)
            
            assert(inputs.shape[1:] == (40, 96, 96)) # specific model shape for now        
            inputs = tf.expand_dims(inputs, axis=4)
            assert(inputs.shape[1:] == (40, 96, 96, 1)) # specific model shape for now
            
            inputs = tf.unstack(inputs, axis=1)
            ModelLogger.log('input-unstack', inputs)
            
            logging.debug('--- begin model definition ---')
            
            # use VGG16 pretrained on imagenet as an initialization        
            vgg_layers = vgg16_model.VGG16(weights=vgg16_weights, sess=sess, trainable=fine_tune_vgg16)
            inputs = [ vgg_layers(inp) for inp in inputs ]
            ModelLogger.log('vgg16_conv', inputs)
            
            # use feedback network architecture below
            # ...
            
            logging.debug('--- end model definition ---')
            
            logits = inputs
            ModelLogger.log('logits', logits)
            
            return logits
        
    return model_generator

def basic_conv2d_generator(kernel_size, num_in_filters, num_out_filters, is_training):

    def generator(inputs, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
        
            with tf.variable_scope("conv0"):
                bn = batch_norm_generator(is_training)
                w = new_conv2dweight(kernel_size, kernel_size, num_in_filters, num_out_filters, reuse)
                
                inputs = tf.nn.conv2d(inputs, w, strides=[1,1,1,1], padding='SAME')
                inputs = bn(inputs, reuse, 'batch_norm')
                inputs = tf.nn.relu(inputs)
            
                return inputs

    return generator

def max_pool(x):
    """MaxPool
    tf.nn.max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
    )
    """
    return tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'SAME', data_format='NHWC')

def new_conv2dweight(xdim, ydim, input_depth, output_depth, reuse):
    weights = tf.get_variable('conv_v', shape=[xdim, ydim, input_depth, output_depth], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), regularizer=None)
    return weights

if __name__ == '__main__':
    sess = tf.Session()
    
    model_generator = nofeedbacknet_vgg16(None)

    x = tf.placeholder(tf.float32, [None, 40, 96, 96], name='inputs')
    logits = model_generator(x, 101, 'vgg16_weights.npz', sess)

    graph = tf.get_default_graph()
    
    for op in graph.get_operations():
        print((op.name))
