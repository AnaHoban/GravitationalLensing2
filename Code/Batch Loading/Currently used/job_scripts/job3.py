''' Training auto-encoder with two bands and weigths for 50 epochs with ~ 250 000 cutouts'''

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from func_job2 import *
from tensorflow.keras.callbacks import ModelCheckpoint
import csv
from itertools import chain

cutout_dir = os.path.expandvars("$SCRATCH") + '/'
image_dir = "/home/anahoban/projects/rrg-kyi/astro/cfis/W3/"

#cutouts file
print('opening file')
hf = h5py.File(cutout_dir + "testing_create_cuts_noisy.h5", "r")

#to get random order of tiles
#csv = pd.read_csv(cutout_dir + 'tiles_unbalanced.csv')
#tile_ids = list(csv.keys())

tile_ids = list(hf.keys())
print(tile_ids)


##TRAINING PREP##
BATCH_SIZE  = 256
CUTOUT_SIZE = 64
N_EPOCHS    = 100

# tiles for val and training
#train_indices = range(5)
train_indices = chain(range(11), range(12, 15))
val_indices = [11] #both channels

#autosave
model_checkpoint_file = "../Models/job8.h5"
model_checkpoint_callback = ModelCheckpoint(model_checkpoint_file, monitor='val_loss', mode='min',verbose=1, save_best_only=True)

print('starting training')
#training
bands = 2
autoencoder_cfis = create_autoencoder2((CUTOUT_SIZE, CUTOUT_SIZE, bands*2)) #last is the number of channels
autoencoder_cfis.compile(optimizer="adam", loss=MSE_with_uncertainty)

(autoencoder_cfis, history_cfis) = train_autoencoder(hf, tile_ids, autoencoder_cfis, train_indices,  val_indices, batch_size=BATCH_SIZE, cutout_size=CUTOUT_SIZE, n_epochs= N_EPOCHS, all_callbacks = [model_checkpoint_callback], bands="cfis")

hf.close()

#saving model
autoencoder_cfis.save("../Models/job8")
hist_df = pd.DataFrame(history_cfis.history) 

hist_csv_file = '../Histories/job8.csv'
with open(hist_csv_file, mode='w', newline='') as f:
    hist_df.to_csv(f)
