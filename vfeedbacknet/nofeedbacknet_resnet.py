import vfeedbacknet.resnet_model
import vfeedbacknet.convLSTM

import tensorflow as tf
import logging

logging.basicConfig(level=logging.DEBUG)

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

class ModelLogger:
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

def reuse_mask(l):
    '''
    reuse the weights created by a generator after declaration
    '''
    return [ None ] + [ True for _ in range(len(l)-1) ]
            
def resnet_50(hparams):

    def model_generator(inputs, num_classes_pretrain, num_classes_train, is_training):
        '''
        inputs: A tensor fo size [batch, video_length, video_height, video_width, channels]
        '''
        ModelLogger.log('input', inputs)
        
        inputs = tf.expand_dims(inputs, axis=4)
        #assert(inputs.shape[1:] == (40, 96, 96, 1)) # specific model shape for now

        inputs = tf.unstack(inputs, axis=1)
        ModelLogger.log('input-unstack', inputs)

        logging.debug('--- begin model definition ---')
        print(reuse_mask(inputs))
        with tf.variable_scope("layer0"):
            block = basic_conv2d_generator(5, 1, 64, is_training)
            inputs = [ block(inp, reuse, 'basic_conv_block') for inp, reuse in zip(inputs, reuse_mask(inputs)) ]
            ModelLogger.log('basic_conv_block', inputs)
            
            inputs = [ max_pool(inp) for inp in inputs ]
            ModelLogger.log('max_pool', inputs)
        
        with tf.variable_scope("layer1"):
            block = resnet_block_generator(2, 64, is_training)
            inputs = [ block(inp, reuse, 'resnet_block0') for inp, reuse in zip(inputs, reuse_mask(inputs)) ]
            ModelLogger.log('resnet_block', inputs)

            # add lstm here
            
        with tf.variable_scope("layer2"):
            block = resnet_block_generator(2, 128, is_training)
            inputs = [ block(inp, reuse, 'resnet_block1') for inp, reuse in zip(inputs, reuse_mask(inputs)) ]
            ModelLogger.log('resnet_block', inputs)

            # add lstm here

        with tf.variable_scope("layer3"):
            block = resnet_block_generator(2, 256, is_training)
            inputs = [ block(inp, reuse, 'resnet_block2') for inp, reuse in zip(inputs, reuse_mask(inputs)) ]
            ModelLogger.log('resnet_block', inputs)

            # add lstm here

        with tf.variable_scope("layer4"):
            block = resnet_block_generator(2, 512, is_training)
            inputs = [ block(inp, reuse, 'resnet_block3') for inp, reuse in zip(inputs, reuse_mask(inputs)) ]
            ModelLogger.log('resnet_block', inputs)

            # add lstm here
            
        with tf.variable_scope("dense_pretrain"):
            inputs_pretrain = inputs 

            inputs_pretrain = [ tf.layers.average_pooling2d(
                inputs=inp, pool_size=3, strides=1, padding='VALID',
                data_format='channels_last', name='ave_pool') for inp in inputs_pretrain ]

            inputs_pretrain = [ tf.reshape(inp, [-1, 512]) for inp in inputs_pretrain ]
            inputs_pretrain = [ tf.layers.dense(inputs=inp, units=num_classes_pretrain, reuse=reuse, name='fc') for inp, reuse in zip(inputs_pretrain, reuse_mask(inputs_pretrain)) ]
            
            output_pretrain = inputs_pretrain

        with tf.variable_scope("dense_train"):
            inputs_train = inputs 

            inputs_train = [ tf.layers.average_pooling2d(
                inputs=inp, pool_size=3, strides=1, padding='VALID',
                data_format='channels_last', name='ave_pool') for inp in inputs_train ]

            inputs_train = [ tf.reshape(inp, [-1, 512]) for inp in inputs_train ]
            inputs_pretrain = [ tf.layers.dense(inputs=inp, units=num_classes_pretrain, reuse=reuse, name='fc') for inp, reuse in zip(inputs_pretrain, reuse_mask(inputs_pretrain)) ]
            
            output_train = inputs_train

        logging.debug('--- end model definition ---')

        logits_pretrain = output_pretrain
        logits_train = output_train
        ModelLogger.log('logits_pretrain', logits_pretrain)
        ModelLogger.log('logits_train', logits_train)

        return output_pretrain, output_train
        
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

def resnet_block_generator(num_layers, num_filters, is_training, add_padding=True):
    assert num_filters % 2 == 0, 'number of layers must be a multiple of 2'
    
    def generator(inputs, reuse, name):
        with tf.variable_scope(name, reuse=reuse):

            shortcut = inputs
            with tf.variable_scope("block{}".format(0)):

                with tf.variable_scope("shortcut_padding"):
                    w = new_conv2dweight(3, 3, int(inputs.shape[-1]), num_filters, reuse)

                    shortcut = tf.nn.conv2d(shortcut, w, strides=[1,2,2,1], padding='SAME')

                with tf.variable_scope("conv0"):
                    bn = batch_norm_generator(is_training)
                    w = new_conv2dweight(3, 3, int(inputs.shape[-1]), num_filters, reuse)

                    inputs = tf.nn.conv2d(inputs, w, strides=[1,2,2,1], padding='SAME')
                    inputs = bn(inputs, reuse, 'batch_norm')
                    inputs = tf.nn.relu(inputs)

                with tf.variable_scope("conv1"):
                    bn = batch_norm_generator(is_training)
                    w = new_conv2dweight(3, 3, num_filters, num_filters, reuse)

                    inputs = tf.nn.conv2d(inputs, w, strides=[1,1,1,1], padding='SAME')

                    inputs = inputs + shortcut

                    inputs = bn(inputs, reuse, 'batch_norm')
                    inputs = tf.nn.relu(inputs)
                    shortcut = inputs

            for layer in range(1, num_layers):
                with tf.variable_scope("block{}".format(layer), reuse=reuse):

                    with tf.variable_scope("conv0"):
                        bn = batch_norm_generator(is_training)
                        w = new_conv2dweight(3, 3, num_filters, num_filters, reuse)

                        inputs = tf.nn.conv2d(inputs, w, strides=[1,1,1,1], padding='SAME')
                        inputs = bn(inputs, reuse, 'batch_norm')
                        inputs = tf.nn.relu(inputs)

                    with tf.variable_scope("conv1"):
                        bn = batch_norm_generator(is_training)
                        w = new_conv2dweight(3, 3, num_filters, num_filters, reuse)

                        inputs = tf.nn.conv2d(inputs, w, strides=[1,1,1,1], padding='SAME')

                        inputs = inputs + shortcut

                        inputs = bn(inputs, reuse, 'batch_norm')
                        inputs = tf.nn.relu(inputs)
                        shortcut = inputs

            return inputs
        
    return generator
    
def batch_norm_generator(is_training):

    def generator(inputs, reuse, name):
        inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=-1,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=is_training, trainable=True,
            fused=True, reuse=reuse, name='bn')
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
    
    model_generator = resnet_50(None)

    x = tf.placeholder(tf.float32, [None, 2, 96, 96], name='inputs')
    logits_pretrain, logit_train = model_generator(x, 1000, 101, True)

    graph = tf.get_default_graph()
    
    for op in graph.get_operations():
        print((op.name))
