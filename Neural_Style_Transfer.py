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



