''' Training auto-encoder with one band and weigths for 50 epochs with ~ 250 000 cutouts'''

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from func_job1 import *
from tensorflow.keras.callbacks import ModelCheckpoint

tile_list = open(image_dir + "tiles.list", "r")
tile_ids = []

for tile in tile_list:
    tile = tile[:-1] # Remove new line character
    channels = tile.split(" ")
    if len(channels) == 5: # Order is u,g,r,i,z
        tile_ids.append(channels[0][5:12]) # XXX.XXX id
tile_list.close()

#cutouts file
hf = h5py.File(cutout_dir + "cutouts_filtered_64p.h5", "r")

##TRAINING PREP##
BATCH_SIZE  = 256 
CUTOUT_SIZE = 64
N_EPOCHS    = 50 

# tiles for val and training
train_indices = range(17)
val_indices = [19]

#autosave
model_checkpoint_file = "../Models/job4.h5"
model_checkpoint_callback = ModelCheckpoint(model_checkpoint_file, monitor='val_loss', mode='min',verbose=1, save_best_only=True)


#training
bands = 1
autoencoder_cfis = create_autoencoder2((CUTOUT_SIZE, CUTOUT_SIZE, bands*2)) #last is the number of channels
autoencoder_cfis.compile(optimizer="adam", loss=MSE_with_uncertainty)

(autoencoder_cfis, history_cfis) = train_autoencoder(hf, tile_ids, autoencoder_cfis, train_indices,  val_indices, batch_size=BATCH_SIZE, cutout_size=CUTOUT_SIZE, n_epochs= N_EPOCHS,all_callbacks = [model_checkpoint_callback], bands="cfis")

hf.close()

#saving model
autoencoder_cfis.save("../Models/job4")
hist_df = pd.DataFrame(history_cfis.history) 

hist_csv_file = '../Histories/job4.csv'
with open(hist_csv_file, mode='a') as f:
    hist_df.to_csv(f)