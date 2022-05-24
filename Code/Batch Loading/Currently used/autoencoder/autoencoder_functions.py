import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import csv
import random


####

CROP_SIZE = 150
BATCH_SIZE  = 2 
bands = 5

#######

#global variables
#### PRE-PROCESSING #####
def rotation(tensor):
    rand = random.random()
    if rand < 1/3:
        return tf.image.rot90(tensor, k=1)
    elif rand > 2/3:
        return tf.image.rot90(tensor, k=2)
    else:
        return tensor
    
def normalization(cutout): #apply same image normalization to corresponding weight channel *****
#     Say we want to rescale a cutout I and weight W, what is currently done:
#     a = 1 / (percentile_99.999(I) - percentile_0.001(I) )
#     b = min(I)
#     I' = a(I-b), and no-scaling of W
#      What should be probably done on W:
#     Var(I') = a²Var(I), w = 1/Var(I)
#     W' = W / a²
    
    image, weight = cutout[...,:5], cutout[...,5:]
    #image percentiles
    img_lower = tfp.stats.percentile(image, 0.001, axis = [0,1])
    img_upper = tfp.stats.percentile(image, 99.999,axis = [0,1])

    lower_broadcast = tf.ones(tf.shape(image), dtype=image.dtype) * img_lower
    arr = tf.where(tf.less(image, img_lower), lower_broadcast, image)

    lower_broadcast = tf.ones(tf.shape(image), dtype=image.dtype) * img_upper
    arr = tf.where(tf.less(image, img_lower), lower_broadcast, image)
    
    #images
    img_norm = (image - tf.reduce_min(image, axis = [0,1])) / (img_upper - img_lower)
    
    #images
    weight_norm = weight/ (img_upper - img_lower)**2
    
    return tf.concat([img_norm, weight_norm],axis = -1)

def preprocess_image(data, crop_size = CROP_SIZE, masking_weights = False):
    '''This function applies the following transformation on the data:
    1. normalizing
    2. cropping
    3. rotation
    4. flipping
    5. masking'''     
    # 1. normalization
    norm_cutout = normalization(data)

    # 2. cropping & translating
    crop_cutout = tf.image.random_crop(norm_cutout, size = [crop_size,crop_size,10])

    # 3. rotating
    rot_cutout = rotation(crop_cutout)

    # 4. flipping
    flip_cutout = tf.image.random_flip_left_right(rot_cutout)
    flip_cutout = tf.image.random_flip_up_down(flip_cutout)
    
    no_nan_cutout = tf.where(tf.math.is_nan(flip_cutout), tf.cast(0, tf.float64), flip_cutout)

    # 5. masking #do we need to maks the weights?***
    if masking_weights is True:
        mask = tf.random.uniform(shape=(crop_size,crop_size,10), minval=0, maxval=2, dtype=tf.int32)
    if masking_weights is False:
        img_mask = tf.cast(tf.random.uniform(shape=(crop_size,crop_size,5), minval=0, maxval=2, dtype=tf.int32), tf.float64)
        weight_mask = tf.cast(tf.ones(shape=(crop_size,crop_size,5)), tf.float64)
        mask = tf.concat([img_mask, weight_mask],axis = -1)
        
    masked_cutout = no_nan_cutout*tf.cast(mask, no_nan_cutout.dtype)
    #masked_cutout = tf.where(tf.math.is_nan(masked_cutout), tf.cast(0, tf.float64), masked_cutout)
        
    # 6. add noise in missing channels #just sering them to 0 for now
#     for i in range(10):
#         if tf.math.is_nan(tf.reduce_sum(masked_cutout[...,i])) == True:
#             noise_mask[...,i] == tf.random.normal((crop_size,crop_size))
        
    return masked_cutout, no_nan_cutout #outputing tuple mask-label!!

def create_autoencoder(shape):
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
    #set nan to 0 and flag it + counter
    return keras.Model(input_all, decoded_all)



bands = 5
def MSE_with_uncertainty(y_true, y_pred): #loss with weights
    weights = y_pred[...,bands:] 
    y_pred_image = y_pred[...,:bands]
    y_true_image = y_true[...,:bands]
    
    #set nan to 0 and flag it + counter
    loss = K.square(tf.math.multiply((y_true_image - y_pred_image), weights) )
    #print(tf.reduce_sum(loss).numpy())
    return tf.where(tf.math.is_nan(loss), tf.cast(0, tf.float32), loss)

def plot_loss_curves(history, n, figname):
    '''Plots loss curves and saves it in ../Loss Curves subdir'''
    fig, axes = plt.subplots(1,2, figsize=(16,8))
    plt.suptitle("Loss Curves for Training/Validation Sets")
    
    axes[0].plot(history["loss"], color="g", label="Training")
    axes[0].plot(history["val_loss"], color="b", label="Validation")
    axes[0].set_title('All history')
    axes[1].plot(history["loss"][-n:], color="g", label="Training")
    axes[1].plot(history["val_loss"][-n:], color="b", label="Validation")
    axes[1].set_title(f'Last {n} epochs')
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[0].legend()
    plt.savefig("./" + figname)
    
def prepare_dataset(dataset, batch_size):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID calculation
    return dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10 * batch_size).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)