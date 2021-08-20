'''This script takes given tiles and runs the classifier on the cutouts in their r catalogue. It downloads in the scratch directory the r catalogue and 
appends a new column with the classification given to each cutout. Furthermore, cutouts are written in a separate csv file. '''

import os
import sys
import shutil
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


cutout_size = 64

#classifier
classifier = keras.models.load_model("../Models/binary_classifier")
for i in range(len(classifier.layers)):
    classifier.layers[i].trainable = False


######################## FUNCTIONS ##########################
#create cutouts function
def create_cutout(img, x, y, classifier_model = classifier):
    ''' Creates the image and weight cutouts given a tile, the position of the center and the band '''
    
    img_cutout = Cutout2D(img[0].data, (x, y), cutout_size, mode="partial", fill_value=0).data
    
    if np.count_nonzero(np.isnan(img_cutout)) >= 0.05*cutout_size**2 or np.count_nonzero(img_cutout) == 0: # Don't use this cutout
        return np.zeros((cutout_size,cutout_size))
    
    img_cutout[np.isnan(img_cutout)] = 0
    
    img_lower = np.percentile(img_cutout, 0.001)
    img_upper = np.percentile(img_cutout, 99.999)
    
    img_cutout[img_cutout<img_lower] = img_lower
    img_cutout[img_cutout>img_upper] = img_upper
    
    if img_lower == img_upper:
        img_norm = np.zeros((cutout_size, cutout_size))
    else:
        img_norm = (img_cutout - np.min(img_cutout)) / (img_upper - img_lower)
        #inorms.append(1/(img_upper - img_lower))

        
    return img_norm


def classify(cat, cat_name, u_image= None, r_image = None):
    #go through each cutout and classify them
    for i in range(len(cat)): #each cutout in tile
        if cat["FLAGS"][i] != 0 or cat["MAG_AUTO"][i] >= 99.0 or cat["MAGERR_AUTO"][i] <= 0 or cat["MAGERR_AUTO"][i] >= 1:
            continue

        x = cat[i]['X_IMAGE']
        y = cat[i]['Y_IMAGE']

        cut = np.random.normal(loc = 0.5,scale= 0.13, size= (1, cutout_size, cutout_size, 4)) 
        
        if u_image != None:
            cut[...,0] =  create_cutout(u_image, x, y)
        if r_image != None:
            cut[...,1] =  create_cutout(r_image, x, y)

        if cut.shape == (1,64,64,4):
            cut_pred = classifier.predict(cut)
            #add to catalogue
            cat[i]['pred'] = cut_pred
        else:
            continue

        if cut_pred >= 0.5:
            f, ax = plt.subplots(1,2)
            ax[0].imshow(cut[...,0])
            ax[0].set_title('u band ')

            ax[1].imshow(cut[...,1])
            ax[1].set_title('r band')

            f.suptitle('score: ' + str(cut_pred))

            plt.savefig(scratch + f'Lenses/lens_{tile_id}_{i}')

    cat.write(scratch + 'catalogues/' + cat_name , format='pandas.csv')    
    
    if u_image != None:
        u_image.close()
    if r_image != None:
        r_image.close()

    print(f'tile {tile_id} done')
#############################################################

#get tile id
tile = sys.argv[1] 
print('read in ', tile)

#directories
scratch = os.path.expandvars("$SCRATCH") + '/'
tmp_dir = os.path.expandvars("$SLURM_TMPDIR") + '/'
files_dir = sys.argv[2] #tmp_dir + tile + '/'


tile_id = tile[5:] #get only the XXX.XXX id (by cropping CFIS.)

#file names
rcat_name = 'CFIS.'+ tile_id + '.r' + ".cat"
ucat_name = 'CFIS.'+ tile_id + '.u' + ".cat"

u_name =  'CFIS.'+ tile_id + '.u' + ".fits"
r_name =  'CFIS.'+ tile_id + '.r' + ".fits"

u_tile = None
r_tile = None

#go through tiles
available_files = os.listdir(files_dir)
print(available_files)

if rcat_name in available_files:
    cat = Table.read(files_dir + '/' + rcat_name, format="ascii.sextractor")
    r_tile = fits.open(files_dir + '/' + r_name, memmap=True) 
    cat_name = rcat_name
    
    if u_name in available_files:
        u_tile = fits.open(files_dir + '/' + u_name, memmap=True)
    

elif ucat_name in available_files:
    cat = Table.read(files_dir + '/' + ucat_name, format="ascii.sextractor")
    u_tile = fits.open(files_dir + '/' + u_name, memmap=True)
    cat_name = ucat_name
    
#initialize classification column
cat['pred'] = 2.0 

classify(cat=cat, cat_name = cat_name, u_image= u_tile, r_image = r_tile)




