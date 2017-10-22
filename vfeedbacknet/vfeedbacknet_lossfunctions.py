import tensorflow as tf

from vfeedbacknet.vfeedbacknet_utilities import ModelLogger

def loss_pred(inputs, inputs_sequence_length, inputs_sequence_maxlength, labels, zeros, last_loss_multiple=1):
    assert len(inputs) == inputs_sequence_maxlength, 'inputs must be the max sequence length'

    predictions = tf.stack([ tf.nn.softmax(logits=inp) for inp in inputs ], axis=1)
    ModelLogger.log('predictions', predictions)
    
    cross_entropies = [ tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=inp) for inp in inputs ]
    #ModelLogger.log('cross_entropies', cross_entropies)
    
    # only add the loss incurred during the sequence
    # [ 0 1 2 3 4 5 ] seqlen=3 -> [ 0 1 2 0 0 0 ]
    cross_entropies_truncated = [ tf.where(i > inputs_sequence_length-1, zeros, cross_entropies[i]) for i in range(len(inputs)) ]

    # boost the loss on the last output by last_loss_multiple
    # [ 0 1 2 0 0 0 ] seqlen=3 -> [ 0 0 2 0 0 0 ] 
    last_cross_entropy  = [ tf.where(i < inputs_sequence_length-1, zeros, cross_entropies_truncated[i]) for i in range(len(inputs)) ]
    cross_entropies_truncated = [ cross_entropies_truncated[i] + last_loss_multiple*last_cross_entropy[i] for i in range(len(inputs)) ]
    
    losses = tf.stack(cross_entropies_truncated, axis=1, name='loss')
    ModelLogger.log('losses', losses)
    
    total_loss = tf.reduce_sum(tf.reduce_sum(tf.stack(cross_entropies_truncated)) / tf.to_float(inputs_sequence_length), name='total_loss')
    ModelLogger.log('total_loss', total_loss)
        
    return losses, total_loss, predictions

