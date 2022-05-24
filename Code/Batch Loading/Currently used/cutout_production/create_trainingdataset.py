import os
import shutil
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#define useful directories
scratch = '/home/anahoban/projects/rrg-kyi/astro/cfis/'
h5_names = ['Dataset_run2_'+ str(i+1) + '.h5' for i in range(3)]

class generator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf:
                yield hf[im]
       
        #for im in range(len(arr)):
            #yield arr[im]

dataset_1 = tf.data.Dataset.from_generator(
    generator(scratch + 'Dataset_run2_1.h5'), 
    tf.float64, 
    tf.TensorShape([200,200,10]))

#creating the dataset
for ds in h5_names:
    hf = h5py.File(scratch + ds, "r")

    gen = lambda: (tf.expand_dims(hf.get(key), axis=0) for key in hf.keys())

    
    new_ds = tf.data.Dataset.from_generator(
    generator(scratch + ds), 
    tf.float64, 
    tf.TensorShape([200,200,10]))

    
    dataset_1.concatenate(new_ds)
#save dataset
tf.data.experimental.save(dataset_1, path = '/home/anahoban/projects/rrg-kyi/astro/cfis/' + 'class_dataset')