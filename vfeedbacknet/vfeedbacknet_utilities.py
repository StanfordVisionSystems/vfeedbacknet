import numpy as np
import multiprocessing as mp
import os
import random
import sys
                    
import PIL
from PIL import Image

import logging

logging.basicConfig(level=logging.DEBUG)

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

class TrainingLogger:
    '''
    logging utility for debugging during training
    '''

    @staticmethod
    def process_prediction(predictions, labels, lengths, losses=None, competition=False, competition_video_num=None, competition_labels=None):

        batch_size = predictions.shape[0]
        feedback_iterations = predictions.shape[1]

        analysis_per_feedback = []
        for feedback_idx in range(feedback_iterations):
            
            analysis_per_frame = []
            for video_idx in range(batch_size):
                video_labelnum = labels[video_idx]
                #video_labelstr = labels_num2str[video_labelnum]
                video_length = lengths[video_idx]

                fvec = []
                fmax = []
                fmax5 = []
                fsum = []
                floss = []
                for frame_idx in range(video_length):
                    predict = predictions[video_idx, feedback_idx, frame_idx, :]
                    fvec.append(predict)
                    fsum.append(sum(predict))
                    fmax.append(np.argmax(predict))

                    predict_copy = predict.copy()
                    fmax5.append(predict_copy.argsort()[-5:][::-1])

                    if losses is not None:
                        floss.append(losses[video_idx, feedback_idx, :])

                analysis_per_frame.append({
                    'frame_labelvec' : fvec,
                    'frame_labelmax' : fmax,
                    'frame_labelmax5' : fmax5,
                    'frame_probsum' : fsum,
                    'frame_loss' : floss,
                    'video_length' : video_length,
                    'video_labelnum' : video_labelnum,
                    #'video_labelstr' : video_labelstr
                })

            analysis_per_feedback.append(analysis_per_frame)

        analysis_per_feedback_len = len(analysis_per_feedback)
        analysis_per_video_len = len(analysis_per_feedback[0])

        for video_idx in range(analysis_per_video_len):

            first = True
            for feedback_idx in range(analysis_per_feedback_len):

                analysis = analysis_per_feedback[feedback_idx][video_idx]
                
                for a in analysis['frame_probsum']:
                    assert( abs(a - 1) < 0.00001 )

                fmax_str = list(map(lambda x: str(x).zfill(2), analysis['frame_labelmax']))
                fmax5_str = list(map(lambda x: str(x).zfill(2), analysis['frame_labelmax5'][-1]))
                tl = str(analysis['video_labelnum']).zfill(2)

                if competition and feedback_idx == analysis_per_feedback_len-1:
                    sys.stdout.write('{};{}\n'.format(competition_video_num[video_idx], competition_labels[analysis['frame_labelmax'][-1]]))
                    #sys.stdout.flush()
                    
                log_str = '{}                              ({}) {}'.format('T' if tl==fmax_str[-1] else 'F', fmax5_str, fmax_str)
                if first:
                    log_str = '{} true_label,prediction: {},{} ({}) {}'.format('T' if tl==fmax_str[-1] else 'F', tl, fmax_str[-1], fmax5_str, fmax_str)
                    first = False

                logging.debug(log_str)
                #logging.debug('{} true_label,prediction: {},{} ({}) {}'.format('T' if tl==fmax_str[-1] else 'F', tl, fmax_str[-1], fmax5_str, fmax_str))
                #logging.debug('{}'.format(analysis['frame_loss']))

        predictions_summary = []
        for feedback_idx in range(analysis_per_feedback_len):
            
            predicted_vals = np.asarray([ p['frame_labelmax'][-1] for p in analysis_per_feedback[feedback_idx] ])
            correct_predictions = sum(labels == predicted_vals)

            predicted_vals3 = [ l in p['frame_labelmax5'][-1][:3] for l,p in zip(labels, analysis_per_feedback[feedback_idx]) ]
            correct_predictions3 = sum(predicted_vals3)

            predicted_vals5 = [ l in p['frame_labelmax5'][-1] for l,p in zip(labels, analysis_per_feedback[feedback_idx]) ]
            correct_predictions5 = sum(predicted_vals5)
                
            predictions_summary.append({
                'correct_predictions' : correct_predictions,
                'correct_predictions3' : correct_predictions3,
                'correct_predictions5' : correct_predictions5
            })    

        return predictions_summary
            
def pool_init(shared_mem_):
    global shared_mem
    shared_mem = shared_mem_
    
def prepare_video(args):
    data_root, video_path, video_width, video_height, video_length, video_downsample_ratio, video_index, batch_size, shared_mem_idx, is_training, is_ucf101 = args

    video_mem = np.frombuffer(shared_mem[shared_mem_idx], np.ctypeslib.ctypes.c_float)
    video_mem = video_mem.reshape((batch_size, video_length, video_height, video_width))

    pathgen = lambda x : os.path.join(data_root, str(video_path), x)
    
    frames = None
    if os.path.isdir(pathgen('')):
        frames = sorted( os.listdir(pathgen('')) )
    else:
        frames = [ os.path.join(data_root, str(video_path)) ]
        
    flip_frames = bool(random.getrandbits(1)) and is_training and False # no fliping while training on 20bn
    flip_frames = bool(random.getrandbits(1)) and is_ucf101
    crop_frames = is_training
    add_noise = bool(random.getrandbits(1)) and is_training

    # choose a random time to start the video
    num_frames = len(frames)//video_downsample_ratio
    t_offset = 0
    stride_offset = 0
    if is_training and num_frames > video_length:
        t_offset = random.choice(range(num_frames - video_length))
        stride_offset = random.choice(range(video_downsample_ratio))
        
    num_frames = min(len(frames)//video_downsample_ratio, video_length)
    assert num_frames != 0, 'num frames in video cannot be 0: {}'.format(video_path)

    x1 = random.choice(list(range(20)))
    y1 = random.choice(list(range(20)))
    x2 = video_width - random.choice(list(range(20)))
    y2 = video_height - random.choice(list(range(20)))

    rotation_angle = random.choice(list(range(-10,10,1))) if is_training else 0
    
    video_mem[video_index,:,:,:] = 0
    for i in range(num_frames):
        image_idx = video_downsample_ratio * (i + t_offset)
        image_idx = min(image_idx + stride_offset, len(frames))
        image = Image.open(pathgen(frames[image_idx])) # in RGB order by default
        image = image.convert('L') # convert to YUV and grab Y-component

        image = image.resize((video_width, video_height), PIL.Image.BICUBIC)
        
        if flip_frames:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if rotation_angle != 0:
            image = image.rotate(rotation_angle)
            
        if crop_frames:
            image = image.crop(box=(x1,y1,x2,y2))
            image = image.resize((video_width, video_height), PIL.Image.BICUBIC)

        image = np.asarray(image, dtype=np.int32)
        image = image - 116  # center on mean value of 116 (as computed in preprocessing step)
        image = np.asarray(image, dtype=np.float32)
        image = np.clip(image, -128, 128)

        if add_noise:
            noise = np.random.normal(loc=0, scale=5, size=(video_height, video_width))
            image = image #+ noise
            image = np.clip(image, -128, 128)

        video_mem[video_index,i,:,:] = image
        
    return { 'num_frames' : num_frames, 'video_path' : video_path }

def load_videos(pool, data_root, data_labels, video_paths, video_width, video_height, video_length, video_downsample_ratio, is_training, batch_size, shared_mem, shared_mem_idx, is_ucf101=False):

    prepare_video_jobs = [ (data_root, video_paths[i], video_width, video_height, video_length, video_downsample_ratio, i, batch_size, shared_mem_idx, is_training, is_ucf101) for i in range(batch_size) ]
    prepared_videos_f = pool.map_async(prepare_video, prepare_video_jobs)

    def future():
        prepared_videos = prepared_videos_f.get()
        
        video_numframes = np.zeros((batch_size,), dtype=np.int32)
        video_labelnums = np.zeros((batch_size,), dtype=np.int32)

        for i in range(batch_size):
            video_numframes[i] = prepared_videos[i]['num_frames']
            video_labelnums[i] = data_labels[ prepared_videos[i]['video_path'] ]

        video_mem = np.frombuffer(shared_mem[shared_mem_idx], np.ctypeslib.ctypes.c_float)
        video_mem = video_mem.reshape((batch_size, video_length, video_height, video_width))

        batch = {
            'num_videos' : batch_size,
            'video_rawframes' : video_mem,
            'video_numframes' : video_numframes,
            'video_labelnums' : video_labelnums,
        }

        return batch

    return future
