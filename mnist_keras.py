#!/usr/bin/env python
# coding=utf-8

"""Mnist CNN based on keras' example."""

from tensorflow.examples.tutorials.mnist import input_data

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
