#!/usr/bin/env python

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model

def main():
    batch_size = 32
    num_classes = 10
    epochs = 200

    # describe model below    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Flatten())    
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    plot_model(model, to_file='model.png', show_shapes=True)

    # load the data
    import mnist_loaddata
    images, labels = mnist_loaddata.read_mnist('/root/mnist/train-images-idx3-ubyte', '/root/mnist/train-labels-idx1-ubyte')

    # begin training

    # store the trained model

if(__name__ == '__main__'):
    main()
