import tensorflow as tf

from vfeedbacknet.vfeedbacknet_utilities import ModelLogger
from vfeedbacknet.vfeedbacknet_base import VFeedbackNetBase

class VFeedbackNetModel1:


    def __init__(self, sess, num_classes,
                 train_featurizer='NO', train_main_model='FROM_SCRATCH', train_fc='FROM_SCRATCH',
                 weights_filename=None, is_training=True):


        self.sess = sess
        self.weights = np.load(weights) if weights_filename is not None else None
        self.num_classes = num_classes
        
        assert train_featurizer in ['NO', 'FINE_TUNE', 'FROM_SCRATCH'], 'train_featurizer must be either: NO, FINE_TUNE, or FROM_SCRATCH'
        self.train_featurizer = train_featurizer if is_training else 'NO'

        assert train_main_model in ['NO', 'FINE_TUNE', 'FROM_SCRATCH'], 'train_main_model must be either: NO, FINE_TUNE, or FROM_SCRATCH'
        self.train_main_model = train_main_model if is_training else 'NO'

        assert train_fc in ['NO', 'FINE_TUNE', 'FROM_SCRATCH'], 'train_fc must be either: NO, FINE_TUNE, or FROM_SCRATCH'
        self.train_fc = train_fc if is_training else 'NO'

        self.is_training = is_training

        self.main_model_variables = []

        self.vfeedbacknet_base = VFeedbackNetBase(sess, num_classes, train_vgg16=)
        self._declare_variables()

