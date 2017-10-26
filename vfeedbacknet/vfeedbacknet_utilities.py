import numpy as np
import multiprocessing as mp
import os
import random

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

def pool_init(shared_mem_):
    global shared_mem
    shared_mem = shared_mem_

def prepare_video(args):
    data_root, video_path, video_width, video_height, video_length, video_downsample_ratio, video_index, batch_size, is_training = args

    video_mem = np.frombuffer(shared_mem, np.ctypeslib.ctypes.c_float)
    video_mem = video_mem.reshape((batch_size, video_length, video_height, video_width))

    pathgen = lambda x : os.path.join(data_root, str(video_path), x)
    frames = sorted( os.listdir(pathgen('')) )

    flip_frames = bool(random.getrandbits(1)) and is_training and False # no fliping while training on 20bn
    #flip_frames = bool(random.getrandbits(1)) and is_training
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
    assert(num_frames != 0)

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

def load_videos(pool, data_root, data_labels, video_paths, video_width, video_height, video_length, video_downsample_ratio, is_training, batch_size, shared_mem):

    prepare_video_jobs = [ (data_root, video_paths[i], video_width, video_height, video_length, video_downsample_ratio, i, batch_size, is_training) for i in range(batch_size) ]
    prepared_videos = pool.map(prepare_video, prepare_video_jobs)

    video_numframes = np.zeros((batch_size,), dtype=np.int32)
    video_labelnums = np.zeros((batch_size,), dtype=np.int32)

    for i in range(batch_size):
        video_numframes[i] = prepared_videos[i]['num_frames']
        video_labelnums[i] = data_labels[ prepared_videos[i]['video_path'] ]

    video_mem = np.frombuffer(shared_mem, np.ctypeslib.ctypes.c_float)
    video_mem = video_mem.reshape((batch_size, video_length, video_height, video_width))

    batch = {
        'num_videos' : batch_size,
        'video_rawframes' : video_mem,
        'video_numframes' : video_numframes,
        'video_labelnums' : video_labelnums,
    }

    return batch
