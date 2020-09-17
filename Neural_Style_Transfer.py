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
    # Get rid of maxpool which throws away info
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    
    new_model = Sequential()
    for layer in vgg.layer:
        if layer.__class__ == MaxPooling2D:
            # Replace it with average pooling
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)
    
    return new_model

def VGG16_AvgPool_CutOff(shape, num_convs):
    # There are 13 convolutions in total 
    # Pick any of them as "output" of our content model
    
    if num_convs < 1 or num_convs > 13:
        print("num_convs must be in the range [1, 13]")
        return None
    
    model = VGG16_AvgPool(shape)
    new_model = Sequential()
    n = 0     
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        new_model.add(layer)
        if n >= num_convs:
            break
        
    return new_model

def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x

if __name__ == '__main__':
    
    # Open an image













