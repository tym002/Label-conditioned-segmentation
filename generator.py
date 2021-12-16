#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 23:57:44 2021

@author: tianyu
"""

import numpy as np
from load_image import single_class, combine_class

def random_patch(x,y):
    #160,208,160,1
    xx = np.random.randint(0,80)
    yy = np.random.randint(0,104)
    zz = np.random.randint(0,80)
    return x[xx:xx+80, yy:yy+96, zz:zz+80,:], y[xx:xx+80, yy:yy+96, zz:zz+80,:]

def _transform(x,y):
    x = np.flip(x,axis=0)
    y = np.flip(y,axis=0)
    return x, y 

def _create_regular_sample(x,y,batch_size):
    examples = []
    labels = []
    total_sample = x.shape[0]
    for i in range(batch_size):
        index = np.random.randint(1, total_sample)
        image = x[index]
        label = y[index]
        image, label = random_patch(image, label)
        examples.append(image)
        labels.append(label)
    return np.array(examples),np.array(labels)

def _create_val_sample(at_img,at_lb, x, y, classes):

    total_sample = x.shape[0]
    class_arr = classes
    np.random.shuffle(class_arr)
    
    class_num = class_arr[0]
    index = np.random.randint(0, total_sample)
    image = x[index:index+1]
    label = single_class(y[index:index+1], class_num)
    
    at_img = at_img
    signal = single_class(at_lb,class_num)
    return image, at_img, signal, label

def _create_multi_input(x,y,batch_size,classes,n_class,random_atlas,combine_lb):
    examples = []
    labels = []
    atlas = []
    atlas_lb = []
    total_sample = x.shape[0]
    if classes is None:
        class_arr = np.arange(1,n_class+1)
    else:
        class_arr = classes
    if random_atlas:
        atlas_num = np.random.randint(total_sample)
        x[0], x[atlas_num] = x[atlas_num], x[0]  
        y[0], y[atlas_num] = y[atlas_num], y[0]
    np.random.shuffle(class_arr)
    is_flip = np.random.random()
    is_combine = np.random.random() > 0.5
    for i in range(batch_size):
        class_num = class_arr[i]

        index = np.random.randint(1, total_sample)

        image = x[index]
        if is_combine and combine_lb:
            class_num2 = class_arr[i+1]
            label = combine_class(y[index], [class_num,class_num2])
            signal = combine_class(y[0], [class_num,class_num2])
        else:
            label = single_class(y[index], class_num)
            signal = single_class(y[0],class_num)
        at_img = x[0]
        
        if is_flip > 0.5:
            image, label = _transform(image,label)
            at_img, signal = _transform(at_img,signal)
        
        examples.append(image)
        labels.append(label)
        atlas.append(at_img)
        atlas_lb.append(signal)
    return np.array(examples),np.array(atlas), np.array(atlas_lb) ,np.array(labels)

def validation_generator(at_img,at_lb, x, y, classes):
    
    while True:
        sample, atlas_img, atlas_label, label = _create_val_sample(at_img,at_lb, x, y, classes)
        yield [sample, atlas_img, atlas_label], label

def regular_generator(x, y, batch_size):
    while True:
        sample, label = _create_regular_sample(x,y,batch_size)
        yield sample, label

def multi_input_generator(x, y, batch_size, classes, n_class, random_atlas, combine_lb):
    while True:
        sample, atlas_img, atlas_label, label = _create_multi_input(x,y,batch_size,classes,n_class, random_atlas, combine_lb)
        yield [sample, atlas_img, atlas_label], label    
