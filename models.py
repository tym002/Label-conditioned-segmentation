#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 23:23:51 2020

@author: tianyu
"""

from keras.models import Model
from keras.layers import Input, Conv3D, UpSampling3D, Dropout, \
    MaxPooling3D, Concatenate, BatchNormalization, Activation, Conv3DTranspose
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K


def dice_coef(y_true, y_pred):
    y_pred = tf.cast((y_pred > 0.5), tf.float32)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2. * intersection + 0.01) / (union + 0.01)


def soft_dice_loss(y_true, y_pred):
    numerator = 2. * K.sum(y_pred * y_true) + 1.0
    denominator = K.sum(K.square(y_pred)) + K.sum(K.square(y_true)) + 1.0
    loss = 1 - (numerator / denominator)
    return loss


def conv_block(m, dim, acti='relu', bn=True, res=False, do=0):
    n = Conv3D(dim, 3, padding='same', dilation_rate=(1, 1, 1))(m)
    n = BatchNormalization()(n) if bn else n
    n = Activation(acti)(n)

    n = Dropout(do)(n) if do else n

    n = Conv3D(dim, 3, padding='same', dilation_rate=(1, 1, 1))(n)
    n = BatchNormalization()(n) if bn else n
    n = Activation(acti)(n)
    return Concatenate()([m, n]) if res else n


def level_block(m, l1, l2, start_ch, depth, inc, acti, do, bn, mp, up, res, lb):
    if depth > 0:
        n = conv_block(m, start_ch, acti, bn, res)
        m = MaxPooling3D()(n) if mp else Conv3D(start_ch, 3, strides=2, padding='same')(n)
        m = level_block(m, l1, l2, int(inc * start_ch), depth - 1, inc, acti, do, bn, mp, up, res, lb)
        if up:
            m = UpSampling3D()(m)
            m = Conv3D(start_ch, 3, activation=acti, padding='same')(m)
        else:
            m = Conv3DTranspose(start_ch, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, start_ch, acti, bn, res)
    else:

        if lb != 'multi':
            label = MaxPooling3D((16, 16, 16))(l2)
            m = Concatenate()([m, label])

        m = conv_block(m, int(start_ch), acti, bn, res, do)
    return m


def unet(img_shape, out_ch=1, start_ch=8, depth=3, inc_rate=2., activation='relu',
         dropout=0, batchnorm=False, maxpool=True, upconv=True, residual=False, label='multi'):
    i = Input(shape=img_shape, name='inp')
    if label != 'multi':
        l1 = Input(shape=img_shape, name='atlas')
        l2 = Input(shape=img_shape, name='atlas_lb')
        emb = Concatenate()([l1, l2])
    else:
        l1 = None
        l2 = None
        emb = None

    i1 = i

    o1 = level_block(i1, l1, emb, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv,
                     residual, label)

    o1 = Conv3D(out_ch, 1, activation='sigmoid')(o1)

    if label == 'multi':
        model = Model(inputs=[i], outputs=[o1])
    else:
        model = Model(inputs=[i, l1, l2], outputs=[o1])
    model.compile(optimizer=Adam(lr=5e-4), loss=soft_dice_loss, metrics=[dice_coef])
    return model
