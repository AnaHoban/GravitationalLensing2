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
import umap
from sklearn.preprocessing import StandardScaler
from astropy.wcs import WCS,utils

######################## FUNCTIONS ##########################
#create cutouts function
def create_cutout(img, x, y):
    ''' Creates the image and weight cutouts given a tile, the position of the center and the band '''
    
    img_cutout = Cutout2D(img.data, (x.values[0], y.values[0]), cutout_size, mode="partial", fill_value=0).data
    
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

def make_hf(hf):
    #for every tile
    for tile_id in master_catalogue['TILE'].unique():
        cut = np.zeros((cutout_size, cutout_size, 4)) 
        u_name =  'CFIS.'+ str(tile_id) + '.u' + ".fits"
        r_name =  'CFIS.'+ str(tile_id) + '.r' + ".fits"

            # Copy tiles to $SLURM_TMPDIR
        shutil.copy2(image_dir + u_name, tmp_dir)
        shutil.copy2(image_dir + r_name, tmp_dir)

        #open fits file
        u_image = fits.open(tmp_dir + u_name, memmap=True)
        r_image = fits.open(tmp_dir + r_name, memmap=True)   
        
        #for every cutout in a tile
        for cutout in master_catalogue[master_catalogue['TILE'] == tile_id]['CUTOUT']:
            cut = np.zeros((cutout_size, cutout_size, 4))

            x = master_catalogue[master_catalogue['CUTOUT'] == cutout]['X_IMAGE']
            y = master_catalogue[master_catalogue['CUTOUT'] == cutout]['Y_IMAGE']

            cut[...,0] = create_cutout(u_image[0], x, y)
            cut[...,1] = create_cutout(r_image[0], x, y)           
            cut[0,0,2] = int(cutout[1:]) #tracking
            print(cutout)
            hf.create_dataset(f"{cutout}", data=cut)
            
            if int(cutout[1:]) > nb_max:
                break
            
        if int(cutout[1:]) > nb_max:
            break
        u_image.close()
        r_image.close()
    hf.close()


#########################################################


#sizes
cutout_size = 64
nb_max = 500000

#directories
scratch = os.path.expandvars("$SCRATCH") + '/'
tmp_dir = os.path.expandvars("$SLURM_TMPDIR") + '/'
image_dir = "/home/anahoban/projects/rrg-kyi/astro/cfis/W3/"

#all tiles
all_tiles = os.listdir(image_dir)

#tiles we can use
unused_tiles_file = './cutouts/list_unused_tiles.csv'
unused_tiles = list( (pd.read_csv(unused_tiles_file, dtype= str))['0'])


#create all cutout adresses and store them in master catalogue
#all unused cutouts with both r and u channels
available_tiles = []
for tile in unused_tiles:
    r_tile = 'CFIS.' + tile + '.r.fits'
    u_tile = 'CFIS.' + tile + '.u.fits'
    if u_tile in all_tiles and r_tile in all_tiles: #taking cutouts with u and r bands
        available_tiles.append(tile)


#Loading mastercat
print('loading mastercat')
master_catalogue = pd.read_csv(scratch + 'classify_catalogue.csv')[1:]


#fix tiles
print('fixing tiles strings')
need_fix = list(master_catalogue['TILE'].unique())
fixed = available_tiles[:len(need_fix)]

fix_dict = {j: fixed[i] for i,j in enumerate(need_fix)}
master_catalogue['TILE'] = master_catalogue['TILE'].replace(fix_dict)

print('starting cutout creation')
#creating and storing the cutouts    
hf_file_name = 'class_cuts.h5'
os.remove(scratch + hf_file_name)
hf = h5py.File(scratch + hf_file_name, "w")
make_hf(hf)

#creating the dataset
hf = h5py.File(scratch + hf_file_name, "r")

gen = lambda: (tf.expand_dims(hf.get(key), axis=0) for key in hf.keys())

print('starting dataset creation')
dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float64))

#save dataset
tf.data.experimental.save(dataset, path = scratch + 'class_dataset')

print('dataset saved')