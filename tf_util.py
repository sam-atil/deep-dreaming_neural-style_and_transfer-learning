'''tf_util.py
Helper/utility functions related to using TensorFlow for Transfer Learning and working with images
Samuel Atilano
CS 343: Neural Networks
Project 4: Transfer Learning
Spring 2025
'''
import numpy as np
from PIL import Image
import tensorflow as tf


def load_pretrained_net(net_name='vgg19'):
    '''Loads the pretrained network (included in Keras) identified by the string variable `net_name`.

    Parameters:
    -----------
    net_name: str. Name of pretrained network to load. By default, this is VGG19.

    Returns:
    -----------
    The pretained net. Keras object.
    '''
    
    if net_name.lower() == 'vgg19':
        pretrained_net = tf.keras.applications.VGG19(weights = 'imagenet', include_top = False)
    pretrained_net.trainable = False

    return pretrained_net


def get_all_layer_strs(pretrained_net):
    '''returns the list of layer names from the pretrained net.

    Parameters:
    -----------
    pretrained_net: Keras object. The pretrained network.

    Returns:
    -----------
    Python list of str. Length is the number of layers in the pretrained network.
    '''
    layer_names = []
    for layer in pretrained_net.layers:
        layer_names.append(layer.name)

    return layer_names

def filter_layer_strs(layer_names, match_str='conv4'):
    '''Extracts the layer name strings from `layer_names` that have `match_str` in the name.

    Parameters:
    -----------
    layer_names: Python list of str. The complete list of layer names in the pretrained network
    match_str: str. Substring searched for within each layer name

    Returns:
    -----------
    Python list of str. The list of layers from `layer_names` that include the string `match_str`
    '''
    #extracting the layer names
    filtered_layers = []
    for name in layer_names:
        if match_str in name:
            filtered_layers.append(name)

    return filtered_layers


def preprocess_image2tf(img, as_var):
    '''Converts an image from numpy ndarray format to TensorFlow tensor format

    Parameters:
    -----------
    img: ndarray. shape=(Iy, Ix, n_chans). A single image
    as_var: bool. Do we represent the tensor as a tf.Variable?

    Returns:
    -----------
    tf tensor. dtype: tf.float32. shape=(1, Iy, Ix, n_chans)
    '''
    #Instance Variables
    Iy, Ix, n_chans = img.shape

    #Converting to tensor format
    if as_var == True:
        tensor = tf.Variable(img, dtype = tf.float32, trainable = True)
        tensor = tf.reshape(tensor, (1,Iy,Ix,n_chans))
        return tensor


def make_readout_model(pretrained_net, layer_names):
    '''Makes a tf.keras.Model object that returns the netAct (output) values of layers in the pretrained model
    `pretrained_net` that have the names in the list `layer_names` (the readout model).

    Parameters:
    -----------
    pretrained_net: Keras object. The pretrained network
    layer_names: Python list of str. Selected list of pretrained net layer names whose netAct values should be returned
        by the readout model.

    Returns:
    -----------
    tf.keras.Model object (readout model) that provides a readout of the netAct values in the selected layer list
        (`layer_names`).
    '''
    
    #Selecting the readout layers
    readout_layers = []
    for layer_name in layer_names:
        readout_layers.append(pretrained_net.get_layer(layer_name).output)

    #Creating the readout model    
    model = tf.keras.Model(inputs = pretrained_net.input, outputs = readout_layers)

    return model


def tf2image(tensor):
    '''Converts a TensorFlow tensor into a PIL Image object.

    Parameters:
    -----------
    tensor: tf tensor. dtype=tf.float32. shape=(1, Iy, Ix, n_chans). A single image. Values range from 0-1.

    Returns:
    -----------
    PIL Image object. dtype=uint8. shape=(Iy, Ix, n_chans). Image representation of the input tensor with pixel values
        between 0 and 255 (unsigned ints).
    '''
    
    #converting the tensor into numpy
    backToNp = tensor.numpy()

    #converting numpy into PIL image
    backToNp = (backToNp * 255).astype(np.uint8)

    #Checking if Batch Dimension is in it
    if backToNp.ndim > 3:
        backToNp = np.squeeze(backToNp)

    img = Image.fromarray(backToNp)
    return img