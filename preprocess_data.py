'''preprocess_data.py
Preprocessing data in STL-10 image dataset
Samuel Atilano
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np
import load_stl10_dataset


def preprocess_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N,)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9
    '''
    
    #Original Image Shape
    num_imgs, height, width, n_chans = imgs.shape

    #casting images into float and flattening array
    imgs_fl = imgs.astype(float)
    imgs_flat = imgs_fl.reshape(imgs_fl.shape[0], -1)

    #Standardizing the features
    feat_stand = (imgs_flat - imgs_flat.mean(axis=0)) / imgs_flat.std(axis = 0)

    #Fixing class labels
    new_labs = labels.copy()
    new_labs -= 1

    #Reformatting Image Shapes
    feat_stand = feat_stand.reshape(num_imgs, height, width, n_chans)

    return feat_stand, new_labs



def create_splits(data, y, n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Divides the dataset up into train/test/validation/development "splits" (disjoint partitions)

    Parameters:
    ----------
    data: float64 ndarray. Image data. shape=(Num imgs, height*width*chans)
    y: ndarray. int-coded labels.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)
    '''

    #Debug If-Statement
    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(data):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(data)}!')
        return

    #Local Variables
    test = n_train_samps + n_test_samps
    val = test + n_valid_samps
    dev = val + n_dev_samps

    #Setting up X-Variables
    x_train = data[0: n_train_samps]
    x_test = data[n_train_samps: test]
    x_val = data[test: val]
    x_dev = data[val: dev]

    #Setting up Y-Variables
    y_train = y[0: n_train_samps]
    y_test = y[n_train_samps: test]
    y_val = y[test: val]
    y_dev = y[val: dev]

    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev


def load_stl10(n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500, scale_fact=3):
    '''Automates the process of:
    - loading in the STL-10 dataset and labels
    - preprocessing
    - creating the train/test/validation/dev splits.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)
    '''
    
    #Loading in the STL-10 dataset(Also passing in scale_fact)
    stl_imgs, stl_labels = load_stl10_dataset.load(scale_fact=scale_fact)

    #Preprocessing
    feats, labs = preprocess_stl(stl_imgs, stl_labels)

    #Splitting the dataset
    x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev = create_splits(feats, labs, n_train_samps, n_test_samps, n_valid_samps, n_dev_samps)

    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev