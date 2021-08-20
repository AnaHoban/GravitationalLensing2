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
from func_job4_new import *
from tensorflow.keras.callbacks import ModelCheckpoint
import csv
import random
from itertools import chain

print('start')

cutout_dir = os.path.expandvars("$SCRATCH") + '/'
image_dir = "/home/anahoban/projects/rrg-kyi/astro/cfis/W3/"
file_name = "confirmed_cfis_64p.h5"

#cutouts file
print('opening file')
hf = h5py.File(cutout_dir + file_name, "r")

#tile ids
with open('./cutouts/cutouts_adresses.csv', newline='') as f:
    reader = csv.reader(f)
    cutout_ids = list(reader)[0]
print(cutout_ids)

#shuffle
random.Random(4).shuffle(cutout_ids)
train, validate, test = np.split(cutout_ids, [int(.7*len(cutout_ids)), int(.9*len(cutout_ids))])
print(len(train), len(validate),len(test))

##TRAINING PREP##
BATCH_SIZE  = 256
CUTOUT_SIZE = 64
N_EPOCHS    = 30


#autosave
model_checkpoint_file = "../Models/job15.h5"
model_checkpoint_callback = ModelCheckpoint(model_checkpoint_file, monitor='val_loss', mode='min',verbose=1, save_best_only=True)

print('starting training')
#training
bands = 2 
#continue training :
autoencoder_cfis = keras.models.load_model("../Models/job14.h5", custom_objects={'MSE_with_uncertainty': MSE_with_uncertainty})
#new autoencoder:
#autoencoder_cfis = create_autoencoder2((CUTOUT_SIZE, CUTOUT_SIZE, bands*2)) #last is the number of channels
autoencoder_cfis.compile(optimizer="adam", loss=MSE_with_uncertainty)

(autoencoder_cfis, history_cfis) = train_autoencoder(hf = hf, train = train, validate= validate, model = autoencoder_cfis, batch_size=BATCH_SIZE, cutout_size=CUTOUT_SIZE, n_epochs= N_EPOCHS, all_callbacks = [model_checkpoint_callback], bands="cfis")

hf.close()

#saving model
autoencoder_cfis.save("../Models/job15")
hist_df = pd.DataFrame(history_cfis.history) 

hist_csv_file = '../Histories/job15.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
