import os
import shutil
import h5py
import numpy as np
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from astropy.wcs import WCS,utils
import umap.umap_ as umap
import seaborn as sns


scratch = os.path.expandvars('$SCRATCH') + '/'

####### 1 ####### this will be part of the script
#Input
#getting cutouts
file_directory = scratch + 'class_dataset'
cutouts = tf.data.experimental.load(file_directory, element_spec = tf.TensorSpec(shape=(1,64,64,4), dtype=tf.float64))

####### 2 ######
#classification
classifier = keras.models.load_model("../Models/binary_classifier_equalsets")
for i in range(len(classifier.layers)):
    classifier.layers[i].trainable = False

predict_labels = classifier.predict(cutouts)


###### 3 ######
#stats
plt.hist(predict_labels)
plt.title('Histogram of predicted labels on all cutouts') #*** add batch number 
plt.savefig('../Classification/' + 'hist')
###### 4 ######
#lenses
lenses_indices, lens_score =  np.where(predict_labels > 0.5)

print(f"found {len(lenses_indices)} lenses")

#store lenses in array
all_found_lenses = np.ones((len(lenses_indices),64,64,4))
i = 0 
lens_count=0
if len(lenses_indices) > 0:
    for elem in cutouts:
        i += 1
        if i in lenses_indices:
            all_found_lenses[lens_count,...] =  elem[0]
            lens_count += 1
              

for i, score in enumerate(lens_score):
    f, ax = plt.subplots(1,2)
    ax[0].imshow(all_found_lenses[i,...,0])
    ax[0].set_title('u band ')

    ax[1].imshow(all_found_lenses[i,...,1])
    ax[1].set_title('r band')

    f.suptitle('score: ' + str(score))

    #plt.savefig(f'../Classification/Lenses/lens_{int(all_found_lenses[i,0,0,2])}')


####### 5 #######
#latent space rep
## ****put draw umap in functions file
'''
reducer = umap.UMAP(random_state=42, n_neighbors = 5, min_dist = 0.1, metric = 'euclidian')
mapper = reducer.fit(test_x.reshape(len(cutouts),64*64*1))
embedding = reducer.transform(test_x.reshape(len(cutouts),64*64*4))

plt.scatter(embedding[:,0], embedding[:,1], c = pred_label)
plt.legend()
plt.savefig('../Classification/Lenses/umap_classification')'''
