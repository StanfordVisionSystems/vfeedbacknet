# util functions
from vfeedbacknet.vfeedbacknet_utilities import TrainingLogger
from vfeedbacknet.vfeedbacknet_utilities import ModelLogger

from vfeedbacknet.vfeedbacknet_utilities import pool_init
from vfeedbacknet.vfeedbacknet_utilities import prepare_video
from vfeedbacknet.vfeedbacknet_utilities import load_videos

import vfeedbacknet.vfeedbacknet_lossfunctions

# models
import vfeedbacknet.vfeedbacknet_vgg16 as vfeedbacknet_vgg

# import vfeedbacknet.vfeedbacknet_videoLSTM_1 as vfeedbacknet_videoLSTM_1
# import vfeedbacknet.vfeedbacknet_videoLSTM_2 as vfeedbacknet_videoLSTM_2

# import vfeedbacknet.vfeedbacknet_fb_base as vfeedbacknet_fb_base
# import vfeedbacknet.vfeedbacknet_fb_base1 as vfeedbacknet_fb_base1
# import vfeedbacknet.vfeedbacknet_fb_base2 as vfeedbacknet_fb_base2

# import vfeedbacknet.vfeedbacknet_model1 as vfeedbacknet_model1 # VGG16 + 2 x convLSTM
# import vfeedbacknet.vfeedbacknet_model2 as vfeedbacknet_model2 # VGG16 + deconv feedback (no fine tuning on pretrained VGG layers)
# import vfeedbacknet.vfeedbacknet_model3 as vfeedbacknet_model3 # VGG16 + deconv feedback (with some fine tuning on pretrained VGG layers)
# import vfeedbacknet.vfeedbacknet_model4 as vfeedbacknet_model4 
# import vfeedbacknet.vfeedbacknet_model5 as vfeedbacknet_model5 
# import vfeedbacknet.vfeedbacknet_model6 as vfeedbacknet_model6 
# import vfeedbacknet.vfeedbacknet_model7 as vfeedbacknet_model7 
# import vfeedbacknet.vfeedbacknet_model8 as vfeedbacknet_model8
# import vfeedbacknet.vfeedbacknet_model9 as vfeedbacknet_model9
# import vfeedbacknet.vfeedbacknet_model10 as vfeedbacknet_model10
# import vfeedbacknet.vfeedbacknet_model11 as vfeedbacknet_model11
# import vfeedbacknet.vfeedbacknet_model12 as vfeedbacknet_model12
# import vfeedbacknet.vfeedbacknet_model13 as vfeedbacknet_model13
# import vfeedbacknet.vfeedbacknet_model14 as vfeedbacknet_model14
# import vfeedbacknet.vfeedbacknet_model15 as vfeedbacknet_model15
# import vfeedbacknet.vfeedbacknet_model16 as vfeedbacknet_model16
# import vfeedbacknet.vfeedbacknet_model17 as vfeedbacknet_model17
# import vfeedbacknet.vfeedbacknet_model16_ucf as vfeedbacknet_model16_ucf
# import vfeedbacknet.vfeedbacknet_model16_ucf1 as vfeedbacknet_model16_ucf1
# import vfeedbacknet.vfeedbacknet_model18 as vfeedbacknet_model18
# import vfeedbacknet.vfeedbacknet_model19 as vfeedbacknet_model19
# import vfeedbacknet.vfeedbacknet_model20 as vfeedbacknet_model20
# import vfeedbacknet.vfeedbacknet_model21 as vfeedbacknet_model21
# import vfeedbacknet.vfeedbacknet_model22 as vfeedbacknet_model22
# import vfeedbacknet.vfeedbacknet_model23 as vfeedbacknet_model23
# import vfeedbacknet.vfeedbacknet_model24 as vfeedbacknet_model24
# import vfeedbacknet.vfeedbacknet_model25 as vfeedbacknet_model25
# import vfeedbacknet.vfeedbacknet_model26 as vfeedbacknet_model26
# import vfeedbacknet.vfeedbacknet_model27 as vfeedbacknet_model27
import vfeedbacknet.vfeedbacknet_model28 as vfeedbacknet_model28
import vfeedbacknet.vfeedbacknet_model29 as vfeedbacknet_model29
import vfeedbacknet.vfeedbacknet_eccv_model1 as vfeedbacknet_eccv_model1
import vfeedbacknet.vfeedbacknet_eccv_model2 as vfeedbacknet_eccv_model2
import vfeedbacknet.vfeedbacknet_eccv_model3 as vfeedbacknet_eccv_model3
import vfeedbacknet.vfeedbacknet_eccv_model4 as vfeedbacknet_eccv_model4
import vfeedbacknet.vfeedbacknet_eccv_model5 as vfeedbacknet_eccv_model5

# legacy models
# import vfeedbacknet.legacy.convLSTM
# import vfeedbacknet.legacy.convLSTM

# from vfeedbacknet.legacy.nofeedbacknet_singleimage_vgg16 import NoFeedbackNetVgg16
# from vfeedbacknet.legacy.nofeedbacknet_LSTM_vgg16 import NoFeedbackNetLSTMVgg16 as NoFeedbackNetLSTMVgg16_2LSTM

# from vfeedbacknet.legacy.nofeedbacknet_singleLSTM_vgg16_reg0_5 import NoFeedbackNetLSTMVgg16 as NoFeedbackNetLSTMVgg16_reg0_5

# from vfeedbacknet.legacy.nofeedbacknet_singleLSTM_vgg16_short import NoFeedbackNetLSTMVgg16 as NoFeedbackNetLSTMVgg16_short
# from vfeedbacknet.legacy.nofeedbacknet_singleGRU_vgg16_short import NoFeedbackNetGRUVgg16 as NoFeedbackNetGRUVgg16_short

# from vfeedbacknet.legacy.nofeedbacknet_singleLSTM_vgg16 import NoFeedbackNetLSTMVgg16
# from vfeedbacknet.legacy.nofeedbacknet_singleGRU_vgg16 import NoFeedbackNetGRUVgg16
