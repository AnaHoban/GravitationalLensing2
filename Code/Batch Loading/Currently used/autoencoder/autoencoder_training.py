''' Training auto-encoder '''

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from autoencoder_functions import *
from tensorflow.keras.callbacks import ModelCheckpoint
import csv
from itertools import chain
import tensorflow_probability as tfp
import random

############## PARAMS #############
n_train = 10000                   #
n_val   = 1000                    #
                                  #
BATCH_SIZE  = 2                   #
CUTOUT_SIZE = 150                 #
N_EPOCHS    = 25                  #
###################################

print('start')

#define useful directories
scratch = os.path.expandvars('$HOME') +'/projects/rrg-kyi/astro/cfis/'
h5_names = ['Dataset_run2_'+ str(i+1) + '.h5' for i in range(3)]

#cutouts file
hf = h5py.File(scratch+ h5_names[0], "r")

#corresponding master catalogue
df =  pd.read_csv(scratch + 'mastercat_run2.csv')
keys =  df["tile"].astype(str) + '_' + df["index"].str.slice_replace(stop=1, repl='')

#gener
# generator + dataset

class generator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf:
                yield hf[im]
       
        #for im in range(len(arr)):
            #yield arr[im]

ds = tf.data.Dataset.from_generator(
    generator(scratch + 'Dataset_run2_1.h5'), 
    tf.float64, 
    tf.TensorShape([200,200,10]))


#whole data prep
print('prepping dataset')
dataset = prepare_dataset(ds, BATCH_SIZE)


train = dataset.take(n_train)
val = dataset.skip(n_train).take(n_val)
#test = dataset.skip(n_train + n_val).take(n_test)


#autosave
model_checkpoint_file = "./job211.h5"
model_checkpoint_callback = ModelCheckpoint(model_checkpoint_file, monitor='val_loss', mode='min',verbose=1, save_best_only=True)

print('starting training')
#training
autoencoder_cfis = keras.models.load_model("./job21.h5", custom_objects={'MSE_with_uncertainty': MSE_with_uncertainty})
#autoencoder_cfis = create_autoencoder((CUTOUT_SIZE, CUTOUT_SIZE, 10)) #last is the number of channels
autoencoder_cfis.compile(optimizer="adam", loss=MSE_with_uncertainty)

history_cfis = autoencoder_cfis.fit(train, epochs=N_EPOCHS, validation_data=val, callbacks= model_checkpoint_callback)

hf.close()

plot_loss_curves(history_cfis.history, 20, figname = 'autoencoder_job211')

#saving model
#autoencoder_cfis.save("./job2")
hist_df = pd.DataFrame(history_cfis.history) 

hist_csv_file = './job211.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
