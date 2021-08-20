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

#with open(cutout_dir + 'tiles_5channel_41tiles.csv', newline='') as f:
 #   reader = csv.reader(f)
 #   data = list(reader)
    
#tile_ids = [i[0] for i in data]

def get_test_cutouts(index, tile_ids, n_cutouts, cutout_size, bands="cfis", start=0):
    n = 0
    if bands == "all":
        band_indices = [0, 1, 2, 3, 4]
        l = len(band_indices)
        sources = np.zeros((n_cutouts, cutout_size, cutout_size, l))
        weights = np.zeros((n_cutouts, cutout_size, cutout_size, l))
        
        
    elif bands == "cfis":
        band_indices = [2]
        l = len(band_indices)
        sources = np.zeros((n_cutouts, cutout_size, cutout_size, l))
        weights = np.zeros((n_cutouts, cutout_size, cutout_size, l))
        
    else: # PS1
        band_indices = [1, 3, 4]
        l = len(band_indices)
        sources = np.zeros((n_cutouts, cutout_size, cutout_size, l))
        weights = np.zeros((n_cutouts, cutout_size, cutout_size, l))
        
    img_group = hf.get(tile_ids[index] + "/IMAGES")
    wt_group = hf.get(tile_ids[index] + "/WEIGHTS")
    for i in range(start, len(img_group)):
        sources[n,:,:,:] = np.array(img_group.get(f"c{i}"))[:,:,band_indices]
        weights[n,:,:,:] = np.array(wt_group.get(f"c{i}"))[:,:,band_indices]
        n += 1
        if n == n_cutouts:
            #return sources #if no weights
            return np.concatenate((sources, weights),axis=-1) #if we train with weights
        
def get_cutouts(hf, tile_ids,tile_indices, batch_size, cutout_size, bands="all"):
    ''' Input: hf file, tile indices, batch size, dimensions, band and bands
        Output: the img and weight cutouts for the test set as (batch_size, pix, pix, channels) '''
    b = 0 # counter for batch
    if bands == "all":
        band_indices = [0, 1, 2, 3, 4]
    elif bands == "cfis":
        band_indices = [2]
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
        for i in tile_indices:
            img_group = hf.get(tile_ids[i] + "/IMAGES")
            wt_group = hf.get(tile_ids[i] + "/WEIGHTS")
            n_cutouts = len(img_group)
            for n in range(n_cutouts):
                sources[b,:,:,:] = np.array(img_group.get(f"c{n}"))[:,:,band_indices]
                weights[b,:,:,:] = np.array(wt_group.get(f"c{n}"))[:,:,band_indices]
                b += 1
                if b == batch_size:
                    b = 0
                    
                    #yield (sources,sources)# no weights
                    yield (np.concatenate((sources, weights), axis = -1), sources) #with weights
                    
                    
def train_autoencoder(hf, tile_ids, model, train_indices, val_indices, n_epochs, batch_size, cutout_size, all_callbacks = None, bands="all"):
    n_cutouts_train = 0
    for i in train_indices:
        img_group = hf.get(tile_ids[i] + "/IMAGES")        
        n_cutouts_train += len(img_group)
    
    n_cutouts_val = 0    
    for i in val_indices:
        img_group = hf.get(tile_ids[i] + "/IMAGES")        
        n_cutouts_val += len(img_group)
    
    train_steps = 1 #n_cutouts_train // batch_size
    val_steps = 1 #n_cutouts_val // batch_size
    
    history = model.fit(get_cutouts(hf, tile_ids, train_indices, batch_size, cutout_size, bands), 
                        epochs=n_epochs, steps_per_epoch=train_steps, 
                        validation_data=get_cutouts(hf, tile_ids, val_indices, batch_size, cutout_size, bands), 
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



bands = 1
def MSE_with_uncertainty(y_true, y_pred): 
    weights = y_pred[...,bands:] 
    y_pred_image = y_pred[...,:bands]
    
    return K.square(tf.math.multiply((y_true - y_pred_image), weights) )
    #return K.square(tf.math.multiply((y_true - y_pred), 1) ) #no weights
