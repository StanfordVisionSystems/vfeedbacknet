#!/usr/bin/env python3

import argparse
import json
import os

def increment(d, key):
    if key not in d.keys():
        d[key] = 1
    else:
        d[key] += 1

def main(args):

    with open(args.metadata_file, 'r', encoding='utf8') as f:
        metadata = json.loads( f.read() )

    num_classes = len(metadata['label_num2str'])

    classes_count = [ { -2 : metadata['label_num2str'][i] } for i in range(num_classes) ]
    classes_lens = [ { -2 : metadata['label_num2str'][i] } for i in range(num_classes) ]
    classes_widths = [ { -2 : metadata['label_num2str'][i] } for i in range(num_classes) ]
    classes_heights = [ { -2 : metadata['label_num2str'][i] } for i in range(num_classes) ]
    global_lens = {} 
    global_widths = {}
    global_heights = {}
    
    for video in metadata['videos']:
        video_class = video['label_num']

        increment( classes_count[video_class], -1 )

        increment( classes_lens[video_class], video['num_frames'] )
        increment( classes_widths[video_class], video['frame_width'] )
        increment( classes_heights[video_class], video['frame_height'] )

        increment( global_lens, video['num_frames'] )
        increment( global_widths, video['frame_width'] )
        increment( global_heights, video['frame_height'] )

    data = {}
    data['classes_count'] = classes_count
    data['classes_num_frames'] = classes_lens
    data['classes_widths'] = classes_widths
    data['classes_heights'] = classes_heights
    data['global_num_frames'] = global_lens
    data['global_widths'] = global_widths
    data['global_heights'] = global_heights

    print( json.dumps(data, indent=4, sort_keys=True).replace('-2', 'class').replace('-1', 'count') )
    
if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='''
    summarize the metadata on videos the 20BN dataset and output to STDOUT in json format
    gives statistics about the following items:
        - number of videos in each class
        - distribution of video lengths for each class
        - distribution of video widths/heights for each class
        - global distribution of video lengths
        - global distribution of video widths/heights
    ''')
    
    parser.add_argument('metadata_file', type=str, nargs=None,
                        help='the something-something labels file')

    args = parser.parse_args()
    main(args)
