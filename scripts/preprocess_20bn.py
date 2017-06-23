#!/usr/bin/env python3

import sys
import argparse
import h5py

def taxonomy():
    pass

def preprocess(args):
        print(args)

    

if(__name__ == '__main__'):
    
    parser = argparse.ArgumentParser(description='''
    Preprocess the 20BN dataset and store as big hdf5 file on disk.
    ''')
    
    parser.add_argument('label_file', type=str, nargs=None,
                        help='the something-something labels file')

    parser.add_argument('data_file', type=str, nargs=None,
                        help='the something-something train/validation file')
    
    parser.add_argument('data_root', type=str, nargs=None,
                        help='the root directory of the something-something data set')

    args = parser.parse_args()
    preprocess(args)
