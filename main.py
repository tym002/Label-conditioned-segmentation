#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 23:25:48 2020

@author: tianyu
"""
import tensorflow as tf
from models import unet
import os

import numpy as np
from keras.callbacks import ModelCheckpoint
import pandas as pd
from utils import dice_coef
from load_image import load_train_data, single_class
from generator import multi_input_generator, regular_generator, validation_generator
import argparse
import json
from sklearn.model_selection import train_test_split


def three_way(gen):
    for x, y in gen:
        yield x, [y, y]


def train_validate(folder_name, file_name, config_arg):
    save_folder = '...' + folder_name + '/'
    weight_path = save_folder + file_name + '.hdf5'

    if not os.path.exists(save_folder):
        print('making folder...')
        os.makedirs(save_folder)

    print('----- Creating and compiling model... -----')

    classes = config_arg["classes"]  # list of class labels
    n_class = config_arg["n_class"]  # number of training classes, should match the length of 'classes'
    label = config_arg["label"]  # multi: multiple output channel, unet baseline; single: single output channel, LCS
    b_size = config_arg["batch_size"]
    rand_atlas = config_arg["rand_atlas"]  # whether to use random atlas during training
    combine_label = config_arg["combine_label"]  # label augmentation (randomly combine labels) during training
    test_index = config_arg["test_index"]  # cross_validation index, 3 means last 1/3 data used as testing.
    image_size = (config_arg["img_shape_x"], config_arg["img_shape_y"], config_arg["img_shape_z"],
                  config_arg["channel"])
    if label == 'single':
        out_channel = 1
    else:
        if classes is None:
            out_channel = n_class
        else:
            out_channel = len(classes)

    model = unet(img_shape=image_size,
                 depth=config_arg["depth"],
                 dropout=config_arg["dropout"],
                 activation=config_arg["activation"],
                 out_ch=out_channel,
                 start_ch=config_arg["start_ch"],
                 residual=config_arg["residual"],
                 batchnorm=config_arg["batchnorm"],
                 label=label)
    print(model.summary())

    model_checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                       save_best_only=True, save_weights_only=True)

    x_train_path = config_arg["x_train_path"]
    y_train_path = config_arg["y_train_path"]
    x_train, y_train, x_test, y_test = load_train_data(n_class, label, x_train_path, y_train_path, classes, test_index)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

    print('Number of x train:', x_train.shape[0])
    print('Number of x val:', x_val.shape[0])
    print('Number of x test:', y_test.shape[0])

    if label == 'single':
        train_generator = multi_input_generator(x_train, y_train, b_size, classes, n_class, random_atlas=rand_atlas,
                                                combine_lb=combine_label)

        val_generator = validation_generator(x_train[0:1], y_train[0:1], x_val, y_val, classes)
    else:
        train_generator = regular_generator(x_train, y_train, b_size)
        val_generator = regular_generator(x_test, y_test, b_size)

    if label == 'multi':
        mtrain = model.fit_generator(train_generator, steps_per_epoch=20,
                                     epochs=2000, verbose=1, shuffle=True, callbacks=[model_checkpoint],
                                     validation_data=val_generator,
                                     validation_steps=10)
    else:
        mtrain = model.fit_generator(train_generator, steps_per_epoch=20,
                                     epochs=2000, verbose=1, shuffle=True, callbacks=[model_checkpoint],
                                     validation_data=val_generator,
                                     validation_steps=20)

    pd.DataFrame.from_dict(mtrain.history).to_csv(save_folder + 'history_' + file_name + '.csv', index=False)


def prediction(folder_name, file_name, config_arg, classes=None, class_start=1, n_class=4, model_type='conv',
               label='single',
               test_index=3):
    print('----- Loading and preprocessing test data... -----')
    print('Using GPU:', tf.test.is_gpu_available())
    x_train, y_train, x_test, y_test = load_train_data(model_type, n_class, label, classes, test_index)

    print("input size:", x_test.shape)

    if label == 'single' or label == 'combine':
        out_channel = 1
    else:
        if classes is None:
            out_channel = n_class
        else:
            out_channel = len(classes)

    print('----- Creating and compiling model... -----')

    ##############################
    # this is the prediction!!   #
    ##############################
    image_size = (config_arg["img_shape_x"], config_arg["img_shape_y"], config_arg["img_shape_z"],
                  config_arg["channel"])

    model = unet(img_shape=image_size,
                 depth=config_arg["depth"],
                 dropout=config_arg["dropout"],
                 activation=config_arg["activation"],
                 out_ch=out_channel,
                 start_ch=config_arg["start_ch"],
                 residual=config_arg["residual"],
                 batchnorm=config_arg["batchnorm"],
                 label=label)
    print(model.summary())

    save_folder = '...' + folder_name + '/'
    weight_path = save_folder + file_name + '.hdf5'
    save_path = save_folder + 'Prediction_' + file_name + '.npy'

    model.load_weights(weight_path)
    results = []

    if label == 'single':
        if classes is not None:
            predict_range = classes
        else:
            predict_range = range(class_start, n_class + 1)
        for i in predict_range:
            y_test_i = single_class(y_test, i)
            save_path = save_folder + 'Prediction_' + file_name + '_' + str(i) + '.npy'

            atlas_img = x_train[0:1]
            x_test_i1 = np.repeat(atlas_img, x_test.shape[0], axis=0)
            x_test_i2 = np.repeat(single_class(y_train[0:1], i), x_test.shape[0], axis=0)
            model_predict = model.predict([x_test, x_test_i1, x_test_i2], verbose=1, batch_size=2)

            print('Class', i, ': ', dice_coef(y_test_i, model_predict))

            results.append(round(dice_coef(y_test_i, model_predict), 3))
            np.save(save_path, model_predict)

    else:
        model_predict = model.predict([x_test], verbose=1, batch_size=1)
        np.save(save_path, model_predict)
        if classes is None:
            c_range = range(n_class)
        else:
            c_range = classes
        for i in c_range:
            predictions = model_predict[..., i:i + 1]
            score = dice_coef(y_test[..., i:i + 1], predictions)
            results.append(round(score, 3))
            print(score)

    print(results)
    print(np.mean(results))


def main(arg, config_arg):
    mode = arg.mode
    folder_name = arg.folder_name
    file_name = arg.file_name

    if mode == 'train':
        train_validate(folder_name, file_name, config_arg)

    elif mode == 'test':
        prediction(folder_name, file_name, config_arg,
                   classes=config_arg["classes"],
                   class_start=config_arg["class_start"],
                   n_class=config_arg["n_class"],
                   model_type=config_arg["model_type"],
                   label=config_arg["label"],
                   test_index=config_arg[
                       "test_index"])  # cross_validation index, 3 means last 1/3 data used as testing.
    else:
        print('No Such Mode, Quit Without Running ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--mode", default="train", help="train or test")
    parser.add_argument("--config_path", default="config.json", help="path to config file")
    parser.add_argument("--folder_name", default="train_results_1", help="name of the folder to save results")
    parser.add_argument("--file_name", default="train_results_1", help="name of the trained model file")
    parser.add_argument("--gpu", default=0, help="which gpu to use")

    args = parser.parse_args()
    config_args = json.load(open(args.config_path))
    main(args, config_args)
