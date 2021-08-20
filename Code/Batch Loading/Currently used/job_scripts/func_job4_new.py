'''Functions for two channels for test_job --> works with the '''
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import csv

#global variables
cutout_dir = os.path.expandvars("$SCRATCH") + "/"
image_dir = "/home/anahoban/projects/rrg-kyi/astro/cfis/W3/"

#csv = pd.read_csv(cutout_dir + 'tiles_unbalanced.csv')
#tile_ids = list(csv.keys())
        
def get_cutouts(hf, cutout_ids, batch_size, cutout_size, bands="all"):
    ''' Input: hf file, tile indices, batch size, dimensions, band and bands
        Output: the img and weight cutouts for the test set as (batch_size, pix, pix, channels) '''
    b = 0 # counter for batch
    if bands == "all":
        band_indices = [0, 1, 2, 3, 4]
    elif bands == "cfis":
        band_indices = [0,2]
        l = len(band_indices)
    elif bands == "ps1":
        band_indices = [1, 3, 4]
    else:
        print('WARNING: unrecognized band')
        band_indices = [0]   
 
    l = len(band_indices)
    sources = np.zeros((batch_size, cutout_size, cutout_size, l))
    weights = np.zeros((batch_size, cutout_size, cutout_size, l))
    while True:
        for i in cutout_ids:
            (tile, cut) = i.split(' ')

            #get image    
            img = hf.get(tile + "/IMAGES/"  + cut)
            wt  = hf.get(tile + "/WEIGHTS/" + cut)
            
            sources[b,:,:,:] = np.array(img)[:,:,band_indices]
            weights[b,:,:,:] = np.array(wt)[:,:,band_indices]
            if np.isnan(np.sum(sources[b,...,0])): #u is nan
                sources[b,:,:,0] = np.random.normal(loc = 0.5,scale= 0.13, size=(cutout_size, cutout_size))
                weights[b,:,:,0] = np.zeros((cutout_size, cutout_size))
            if np.isnan(np.sum(sources[b,...,1])): #r is nan
                sources[b,:,:,1] = np.random.normal(loc = 0.5,scale= 0.13, size=(cutout_size, cutout_size))
                weights[b,:,:,1] = np.zeros((cutout_size, cutout_size))
            #extra step to normalize
            b += 1
            if b == batch_size:
                b = 0

                #yield (sources,sources)# no weights
                yield (np.concatenate((sources, weights), axis = -1), sources) #with weights

                    
def train_autoencoder(hf, train, validate, model, n_epochs, batch_size, cutout_size, all_callbacks = None, bands="all"):
    n_cutouts_train = len(train) 
    n_cutouts_val = len(validate)
    
    train_steps = n_cutouts_train // batch_size
    val_steps = n_cutouts_val // batch_size
    
    history = model.fit(get_cutouts(hf, train, batch_size, cutout_size, bands), 
                        epochs=n_epochs, steps_per_epoch=train_steps, 
                        validation_data=get_cutouts(hf, validate, batch_size, cutout_size, bands), 
                        validation_steps=val_steps, callbacks= all_callbacks)
    return model, history


def create_autoencoder2(shape):
    input_all = keras.Input(shape=shape)
    weights = input_all[...,shape[-1]//2:]
    input_imgs = input_all[...,:shape[-1]//2]
    
    x = keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding='same')(input_imgs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    y = keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(input_imgs)
    y = keras.layers.BatchNormalization()(y)
    encoded = keras.layers.Add()([x,y])
    
    x = keras.layers.Conv2DTranspose(32, kernel_size=4, activation='relu', padding='same')(encoded)
    x = keras.layers.Conv2DTranspose(16, kernel_size=4, activation='relu', padding='same')(x)
    
    #weights
    decoded_img = keras.layers.Conv2D(shape[2] // 2, kernel_size=3, activation='linear', padding='same')(x)
    decoded_all = tf.concat([decoded_img, weights], axis = -1)
    
    #no weights
    #decoded_all = keras.layers.Conv2D(shape[2], kernel_size=3,activation='relu', padding = 'same')(x)                                  
    
    return keras.Model(input_all, decoded_all)



bands = 2
def MSE_with_uncertainty(y_true, y_pred): 
    weights = y_pred[...,bands:] 
    y_pred_image = y_pred[...,:bands]
    
    loss = K.square(tf.math.multiply((y_true - y_pred_image), weights) )
    return loss #no weights