#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 23:54:32 2021

@author: tianyu
"""

import numpy as np
from tqdm import tqdm


def single_class(y, class_num):
    return (y == class_num).astype('float32')


def combine_class(y, classes):
    maps = np.zeros(y.shape, dtype='float32')
    for i in classes:
        cur_map = (y == i).astype('float32')
        maps = ((maps + cur_map) > 0).astype('float32')
    return maps


def class_map(y, n_class, classes=None):
    if classes is None:
        c_range = range(1, n_class + 1)
    else:
        c_range = classes
    labels = np.empty((y.shape[0], 160, 208, 160, len(c_range)), dtype='float32')
    k = 0
    for i in tqdm(c_range):
        print('converting mask: ' + str(i))
        mask = (y == i).astype('float32')
        labels[..., k:k + 1] = mask
        k += 1
    print('Creating Final Mask ... ...')
    return labels


def load_train_data(n_class, label, x_train_path, y_train_path, classes=None, test_index=3):
    '''
    load the training data and ground truth data  
    '''

    x = np.load(x_train_path)
    y = np.load(y_train_path)
    print('Loading Data Done...')

    if label == 'multi':
        print('Converting label maps')
        y = class_map(y, n_class, classes)
    if test_index == 3:
        x_train = x[:21]
        y_train = y[:21]
        x_test = x[21:]
        y_test = y[21:]
    elif test_index == 1:
        x_train = x[9:]
        y_train = y[9:]
        x_test = x[:9]
        y_test = y[:9]
    else:
        x_train = np.concatenate((x[:9], x[18:]), axis=0)
        y_train = np.concatenate((y[:9], y[18:]), axis=0)
        x_test = x[9:18]
        y_test = y[9:18]

    return x_train, y_train, x_test, y_test


def load_test_data():
    x = np.load('...')
    y = np.load('...')
    imgs_train = x[21:]
    imgs_mask = y[21:]
    return imgs_train, imgs_mask
