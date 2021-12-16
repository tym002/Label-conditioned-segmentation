#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 23:25:48 2020

@author: tianyu
"""
import tensorflow as tf
from models import unet
import os
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
from keras.backend import tensorflow_backend
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from utils import dice_coef, union_prediction, ensemble_prediction, case_dice 
from load_image import load_train_data, load_test_data, single_class, combine_class
from generator import hyper_generator, multi_input_generator, regular_generator, validation_generator

def three_way(gen):
    for x,y in gen:
        yield x,[y,y]
        
def TrainandValidate(folder_name, file_name, classes = None, n_class=4, b_size = 4, model_type = 'conv', label = 'single', rand_atlas = False, combine_label = False, val_inx=2, image_size = (160,208,160,1), y_index = 49, test_index = 3):  

    save_folder = '...' + folder_name + '/'
    weight_path = save_folder + file_name + '.hdf5'
    
    if not os.path.exists(save_folder):
        print('making folder...')
        os.makedirs(save_folder)
      
    print('----- Creating and compiling model... -----')
    
    if label == 'single':
        out_channel = 1
    else:
        if classes is None:
            out_channel = n_class
        else: 
            out_channel = len(classes)

    model = unet(img_shape=image_size,
                depth=4,
                dropout=0.5,
                activation='relu',
                out_ch = out_channel,
                start_ch = 8,
                residual=False,
                batchnorm=True,
                ch_final_layer=16,
                n_class = n_class,
                label = label)
    print(model.summary())
    
    model_checkpoint = ModelCheckpoint(weight_path,monitor='val_loss',verbose=1, 
                                       save_best_only=True,save_weights_only = True)

    
    x_train,y_train,x_test,y_test = load_train_data(model_type, n_class, label, classes, y_index, test_index)
    x_val, y_val = x_train[16:], y_train[16:]
    
    print('Number of x train:',x_train.shape)
    print('Number of y val:',y_test.shape)


    if label == 'single':
        train_generator = multi_input_generator(x_train, y_train, b_size, classes, n_class, random_atlas = rand_atlas, combine_lb = combine_label)

        val_gen = validation_generator(x_train[0:1],y_train[0:1],x_val,y_val,classes)
    else:
        train_generator = regular_generator(x_train,y_train, b_size)
        val_generator = regular_generator(x_test,y_test, b_size)


    if label == 'multi':
        mtrain = model.fit_generator(train_generator, steps_per_epoch=20,
                             epochs=2000, verbose=1, shuffle=True,callbacks=[model_checkpoint],
                             validation_data=val_generator,
                             validation_steps=10)
    else:
        mtrain = model.fit_generator(train_generator, steps_per_epoch=20,
                             epochs=2000, verbose=1, shuffle=True,callbacks=[model_checkpoint],
                             validation_data=val_gen,
                             validation_steps = 20)            

    pd.DataFrame.from_dict(mtrain.history).to_csv(save_folder+'history_'+file_name+'.csv',index=False)
    
def prediction(folder_name, file_name, classes = None, class_start = 1, n_class=4, model_type = 'conv', label = 'single', y_index = 49, test_index = 3):

    print('----- Loading and preprocessing test data... -----')
    print('Using GPU:',tf.test.is_gpu_available())
    x_train,y_train,x_test,y_test = load_train_data(model_type, n_class, label, classes, y_index, test_index)

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
    ### this is the prediction!!##
    ##############################

    model = unet(img_shape=(160,208,160,1),
                depth=4,
                dropout=0.5,
                activation='relu',
                out_ch = out_channel,
                start_ch = 8,
                residual=False,
                batchnorm=True,
                ch_final_layer=16,
                n_class = n_class,
                label = label)
    print(model.summary())
    
    save_folder = '...' + folder_name + '/'
    weight_path = save_folder + file_name + '.hdf5'
    save_path = save_folder+'Prediction_'+file_name+'.npy'

    model.load_weights(weight_path)
    results = []


    if label == 'single':
        if classes is not None:
            predict_range = classes
        else:
            predict_range = range(class_start, n_class+1)
        for i in predict_range:
            y_test_i = single_class(y_test,i)
            save_path = save_folder+'Prediction_'+ file_name+ '_'+ str(i)+ '.npy'

            atlas_img = x_train[0:1]
            x_test_i1 = np.repeat(atlas_img,x_test.shape[0],axis=0)
            x_test_i2 = np.repeat(single_class(y_train[0:1],i),x_test.shape[0],axis=0)
            model_predict = model.predict([x_test, x_test_i1, x_test_i2], verbose =1 ,batch_size =2)

            print('Class',i,': ',dice_coef(y_test_i,model_predict))

            results.append(round(dice_coef(y_test_i,model_predict),3))
            np.save(save_path, model_predict)
            
    else:
        model_predict = model.predict([x_test], verbose =1 ,batch_size =1)
        np.save(save_path, model_predict)
        if classes is None:
            c_range = range(n_class)
        else:
            c_range = classes
        for i in c_range:
            prediction = model_predict[...,i:i+1]
            score = dice_coef(y_test[...,i:i+1],prediction)
            results.append(round(score,3))
            print(score)

    print(results)
    print(np.mean(results))


if __name__ == '__main__':
    
    mode = 'predict'
    
    folder_name = '...'
    file_name = '...'

    # multi: multiple output channel, unet baseline
    # single: single output channel, main method

    if mode == 'train':
       model = TrainandValidate(folder_name, file_name, 
               classes = np.arange(61,80,2), 
               n_class = 10, b_size = 2, model_type = 'conv', 
               label = 'multi',
               rand_atlas = False,
               combine_label = True,
               val_inx = 1,
               image_size = (160,208,160,1),
               y_index = 49, 
               test_index = 3)  

    elif mode == 'predict':
       model = prediction(folder_name, file_name, 
                            classes = np.arange(61,80,2), 
                            class_start = 1, 
                            n_class = 2, model_type = 'conv', 
                            label = 'single', 
                            y_index = 30, # y label data, number of classes 
                            test_index = 3) # cross_validation index, 3 means last 1/3 data used as testing. 
    else:
        print('No Such Mode, Quit Without Running ...')

