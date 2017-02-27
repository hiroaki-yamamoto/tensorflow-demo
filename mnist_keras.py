#!/usr/bin/env python
# coding=utf-8

"""Mnist CNN based on keras' example."""

from keras.datasets import mnist as input_data
from keras import backend as K
from keras.models import Sequential
from keras.layers import (
    Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
)
from keras.utils import np_utils

nb_classes = 10  # That is the number of integers from 0 to 9 = 10
nb_epoch = 12  # Number of times to learn and refresh the network.

# For number of filters
# The filter should be larger than the actual image size, and needs to put
# multiple filters at different position.
# In general, # of filters is power of 2.

nb_filters = 32  # Number of Convolution filter i.e. # of neurons?

(width, height) = (28, 28)  # The size of the image i.e. 28 px x 28 px

# For Pool size
# poo_size should be defined as coeff. i.e. Pool Layer compress the image by
# re-sizing the output. Therefore, (Size of the layer) / (poo_size) is the
# result of the pool layer.

pool_size = (2, 2)  # Pooling co-eff.

kernel_size = (3, 3)  # The size of the filter. i.e. 3 px x 3 px.
batch_size = 128

# Load MNIST data
(X_train, y_train), (X_test, y_test) = input_data.load_data()

if K.image_dim_ordering() == 'th':
    # Perhaps, theano handles the data differently.
    X_train = X_train.reshape(X_train.shape[0], 1, height, width)
    X_test = X_test.reshape(X_test.shape[0], 1, height, width)
    input_shape = (1, height, width)
else:
    X_train = X_train.reshape(X_train.shape[0], height, width, 1)
    X_test = X_test.reshape(X_test.shape[0], height, width, 1)
    input_shape = (height, width, 1)

# Treat the data as float, and standardlize between 0.0 to 1.0.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save("mnist_cnn.h5")
print('The model data has been saved: mnist_cnn.h5')
