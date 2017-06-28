#!/usr/bin/env python3

import argparse
import csv
import os

import h5py
import multiprocessing as mp
import numpy as np
import json

import PIL
from PIL import Image

def vfeedback_model_basic(args):
    pass

def main(args):
    
    model = vfeedback_model_basic(None)
    
    with h5py.File(os.path.join(args.output_hdf5, 'chunk_'+str(i).zfill(2)+'.hdf5'), 'w') as f:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    train the feedbacknet for the 20BN data set
    ''')
    
    parser.add_argument('train_root', type=str, nargs=None,
                        help='directory with the preprocessed training files')

    parser.add_argument('video_width', type=int, nargs=None,
                        help='the width to rescale all videos (recommended 176)')

    args = parser.parse_args()
    main(args)
