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
    # Outputs pixel intensities between 0 and 1

if __name__ == '__main__':
    
    # Open an image
    path = 'images/curi_kit.jpg'
    img = image.load_img(path)

    # Convert image to array and preprocess for VGG (stands for Visual Geometry Group)    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)    
    x = preprocess_input(x)

    # Use this throughout the rest of the script
    batch_shape = x.shape
    shape = x.shape[1:]
    
    # To see the image
    # plt.imshow(img)
    # plt.show()
    
    
    # Make a content model
    # Try different cutoffs to see the image that results
    content_model = VGG16_AvgPool_CutOff(shape, 11)
    
    # Make the target
    target = K.variable(content_model.predict(x))
    
    
    # Try to match the image
    
    # Define our loss in Keras
    loss = K.mean(K.square(target - content_model.output))
    
    # Gradients which are needed by the optimizer
    grads = K.gradients(loss, content_model.input)
    
    # Like Theano.function
    get_loss_and_grads = K.function(
        inputs=[content_model.input],
        outputs=[loss] + grads
        )
    
    def get_loss_and_grads_wrapper(x_vec):
        # Scipy's minimizer allows us to pass back
        # function value f(x) and it's gradient f'(x)
        # Simultaneously, rather than using the fprime arg
        #
        # We cannot use the get_loss_and_grads() directly
        # Input minimizer func must be a 1-D array
        # Input to get_loss_and_grads must be [batch_of_images]
        #
        # Gradient must also be a 1-D array
        # And both loss and gradient must be np.float64
        # Will get an error otherwise
        
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)
    
    
    
    from datetime import datetime
    t0 = dateTime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i range(10):
        x, l, _ = fmin_l_bfgs_b(
            func=get_loss_and_grads_wrapper,
            x0=x,
            # bounds=[[-127, 127]]*len(x.flatten())
            maxfun=20
        )
        x = np.clip(x, -127, 127)
        # print("min:", x.min(), "max:", x.max())
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)
        
    print("duration:", dateTime.now() - t0)
    plt.plot(losses)
    plt.show()
    
    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    
    plt.imshow(scale_img(final_img[0]))
    plt.show()
        
        
        
        
        
        
        
        
        








