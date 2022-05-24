import os
import sys
import h5py
import shutil
import numpy as np
import tensorflow as tf

#define useful directories
project_dir = '/home/anahoban/projects/rrg-kyi/astro/cfis/'
scratch = os.path.expandvars('$SCRATCH') + '/'
h5_names = ['Dataset_run2_'+ str(i+1) + '.h5' for i in range(3)]

hf_1 = h5py.File(scratch + h5_names[0], "r")
hf_2 = h5py.File(scratch + h5_names[1], "r")
hf_3 = h5py.File(scratch + h5_names[2], "r")
hf_4 = h5py.File(scratch + h5_names[2], "r")

gen_1 = lambda: (tf.expand_dims(hf_1.get(key), axis=0) for key in hf_1.keys())
gen_2 = lambda: (tf.expand_dims(hf_2.get(key), axis=0) for key in hf_2.keys())
gen_3 = lambda: (tf.expand_dims(hf_3.get(key), axis=0) for key in hf_3.keys())
gen_4 = lambda: (tf.expand_dims(hf_4.get(key), axis=0) for key in hf_4.keys())

print('starting dataset creation')
shapes = (1,200,200,10)
dataset_1 = tf.data.Dataset.from_generator(gen_1, output_types=(tf.float64),output_shapes=shapes)
dataset_2 = tf.data.Dataset.from_generator(gen_2, output_types=(tf.float64),output_shapes=shapes)
dataset_3 = tf.data.Dataset.from_generator(gen_3, output_types=(tf.float64),output_shapes=shapes)
dataset_4 = tf.data.Dataset.from_generator(gen_4, output_types=(tf.float64),output_shapes=shapes)

combined_dataset = dataset_1.concatenate(dataset_2).concatenate(dataset_3).concatenate(dataset_4)
#save dataset
tf.data.experimental.save(combined_dataset, path = project_dir + 'autoencoder_full_dataset_w')

hf_1.close()
hf_3.close()
hf_2.close()
hf_4.close()

print('done')