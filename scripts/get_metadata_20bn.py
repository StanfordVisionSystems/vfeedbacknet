#!/usr/bin/env python3

from PIL import Image

import argparse
import csv
import h5py
import json
import os
import multiprocessing as mp
import numpy as np

def get_video_stats(inp):
    data_root, data_num, data_labels, labels_num2str = inp

    pathgen = lambda x : os.path.join(args.data_root, str(data_num), x)
    frames = sorted( os.listdir(pathgen('')) )

    image = Image.open(pathgen(frames[0])); assert(image.mode == 'RGB')
    
    metadata = {
        'num_frames' : len(frames),
        'frame_width' : image.size[0],
        'frame_height' : image.size[1],
        'label_num' : data_labels[data_num],
        'label_str' : labels_num2str[ data_labels[data_num] ]
    }    
    return metadata

def main(args):

    with open(args.label_file, 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        labels_num2str = [ item[0].lower() for item in reader ]
        labels_str2num =  { label : idx  for idx,label in zip(range(len(labels_num2str)), labels_num2str) }

    with open(args.data_file, 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data_labels = { int(item[0]) : labels_str2num[item[1].lower()] for item in reader }

    pool = mp.Pool(25)
    jobs = [ (args.data_root, data_num, data_labels, labels_num2str) for data_num in data_labels.keys() ] 
    
    video_stats = pool.map(get_video_stats, jobs)

    metadata = {}
    metadata['label_file'] = os.path.abspath(args.label_file)
    metadata['data_file'] = os.path.abspath(args.data_file)
    metadata['label_num2str'] = labels_num2str
    metadata['videos'] = video_stats

    print( json.dumps(metadata, indent=4, sort_keys=True) )    

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='''
    collect metadata on videos the 20BN dataset and output to STDOUT in json format
    ''')
    
    parser.add_argument('label_file', type=str, nargs=None,
                        help='the something-something labels file')

    parser.add_argument('data_file', type=str, nargs=None,
                        help='the something-something train/validation file')
    
    parser.add_argument('data_root', type=str, nargs=None,
                        help='the root directory of the something-something data set')

    args = parser.parse_args()
    main(args)
