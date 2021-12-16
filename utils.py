#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 23:50:08 2021

@author: tianyu
"""

import numpy as np
from skimage.util import random_noise
from scipy.ndimage import zoom


def dice_coef(y_true, y_pred):
    y_true = (y_true>0.5).astype('int')
    y_pred = (y_pred>0.5).astype('int')
    dice = []
    for i in range(len(y_pred)):
        true,pred = y_true[i],y_pred[i]
        intersection = np.sum(true * pred)
        union = np.sum(true) + np.sum(pred)
        dscore = (2. * intersection + 0.001) / (union + 0.001)
        dice.append(dscore)
    return np.mean(dice)

def case_dice(y_true, y_pred):
    y_true = (y_true>0.5).astype('int')
    y_pred = (y_pred>0.5).astype('int')
    intersection = np.sum(y_true * y_pred,axis=(1,2,3,4))
    union = np.sum(y_true,axis=(1,2,3,4)) + np.sum(y_pred,axis=(1,2,3,4))
    dscore = (2. * intersection + 0.001) / (union + 0.001)
    return dscore

def single_dice(y_true, y_pred):
    y_true = (y_true>0.5).astype('int')
    y_pred = (y_pred>0.5).astype('int')
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dscore = (2. * intersection + 0.001) / (union + 0.001)
    return np.array(dscore)

def _mse(x,y):
    return ((x - y)**2).mean(axis=(1,2,3,4))

def add_noise(x,mod,am):
    y = x.copy()
    for i in range(x.shape[0]):
        if mod == 's&p':
            y[i]=random_noise(x[i],clip=False,mode=mod,amount=am)
        else:
            y[i]=random_noise(x[i],clip=False,mode=mod,var=am)
    return y


def zooming(x,s1):
    if s1 == 0:
        return x
    for i in range(x.shape[0]):
        x[i,:,:,0] = zoom(zoom(x[i,:,:,0],s1),1/s1)
    return x

def union_prediction(x, classes):
    maps = np.zeros(x.shape[:-1],dtype='float32')
    for i in classes:
        cur_map = (x[...,i-1]>0.5).astype('float32')
        maps = ((maps + cur_map)>0).astype('float32')
    return maps

def ensemble_prediction(lst):
    emp = np.empty((9,160,208,160,1))
    for i in lst:
        emp += i
    emp = emp / len(lst)
    return emp





