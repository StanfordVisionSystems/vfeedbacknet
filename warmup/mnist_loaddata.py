#!/usr/bin/env python

import numpy as np
import sys

def read_mnist(image_filename, label_filename):

    images = None
    with open(image_filename, 'rb') as f:
        images = np.fromfile(f, dtype=np.uint8)
        images = images[16:]
        images = images.reshape((28,28,len(images)/(28*28)))

    #print images[:,:,0].shape

    labels = None
    with open(label_filename, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = labels[8:]

    return (images, labels)
        
def main():
    if(len(sys.argv) != 3):
        print('Usage: ' + sys.argv[0] + ' INPUT.bin INPUT.labels')
        sys.exit(0)
        
    images, labels = read_mnist(sys.argv[1], sys.argv[2])
    assert(images.shape[0] == 28)
    assert(images.shape[1] == 28)
    assert(images.shape[2] == len(labels))

if __name__ == '__main__':
    main()
