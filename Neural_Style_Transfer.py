# Deep Learning: Advanced Computer Vision (GANs, SSD) - Lazy Programmer tutorial from Machine Learning A-Z - SuperDataScience
# Input by Ryan L Buchanan 17SEP20

from __future__ import print_function, division
from builtins import range, input

# Given an image, this script should be able to generate content, recreating the same image

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b



def VGG16_AvgPool(shape):
    # Account for features across the entire image
    # get rid of maxpool which throws away info
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    
    new_model = Sequential()
    for layer in vgg.layer:
        if layer.__class__ == MaxPooling2D:
            # replace it with average pooling
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)
    
    return new_model

def VGG16_AvgPool_CutOff(shape, num_convs):
    # there 13 convolutions in total 
    # pick any of them as "output"
    # of our content model
    
    if num_convs < 1 or num_convs > 13:
        print("num_convs must be in the range [1, 13]")
        return None
    
    model = VGG16_AvgPool(shape)
    new_model = Sequential()
    n = 0     