import os
import shutil
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#define useful directories
scratch = os.path.expandvars('$SCRATCH') + '/'
h5_names = ['Dataset_run2_'+ str(i+1) + '.h5' for i in range(3)]

#creating the dataset
for ds in h5_names:
    hf = h5py.File(scratch + ds, "r")

    gen = lambda: (tf.expand_dims(hf.get(key), axis=0) for key in hf.keys())

    print('starting dataset creation')
    dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float64))
    
    dataset.concatenate
#save dataset
tf.data.experimental.save(dataset, path = scratch + 'class_dataset')