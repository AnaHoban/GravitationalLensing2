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
import csv
#sizes
cutout_size = 64
nb_cutouts = 1000000

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
ex_cat = image_dir + 'CFIS.' + available_tiles[0] + '.u.cat'
example = Table.read(ex_cat, format="ascii.sextractor")
keys = example.keys()
master_catalogue = pd.DataFrame(index = [0], columns = keys + ['TILE'] + ['BAND'] + ['CUTOUT'] + ['NB'])
master_catalogue.to_csv(scratch + 'classify_catalogue.csv') 

#master_catalogue = pd.read_csv(scratch + 'classify_catalogue.csv')
keys = master_catalogue.keys()
print('master cat created \n')
#populate master cat

nb = 0
for tile_id in available_tiles: #single and both channels

    rcat = Table.read(image_dir + 'CFIS.'+ tile_id + '.r' + ".cat", format="ascii.sextractor")
    count = 0
    for i in range(len(rcat)): #each cutout in tile
        if rcat["FLAGS"][i] != 0 or rcat["MAG_AUTO"][i] >= 99.0 or rcat["MAGERR_AUTO"][i] <= 0 or rcat["MAGERR_AUTO"][i] >= 1:
            continue

        #keep track
        '''new_cutout = pd.DataFrame(index = [i], data=np.array(rcat[i]), columns = keys + ['TILE'] + ['BAND'] + ['CUTOUT'] + ['NB'])
        new_cutout['BAND'] = 'r'
        new_cutout['TILE'] = tile_id
        new_cutout['CUTOUT'] = f"{count}"
        new_cutout['NB'] = f"c{nb}"'''

        #master_catalogue = master_catalogue.append(new_cutout)
        new_cutout = ([i] + list(rcat[i])+[tile_id,'r',f"c{nb}", f"{count}"])
        # open the file in the write mode
        with open(scratch + 'classify_catalogue.csv', 'a') as f:
            # create the csv writer
            writer = csv.writer(f)

            # write a row to the csv file
            writer.writerow(new_cutout)
        
        count += 1
        nb += 1
        if nb == nb_cutouts:
            break
    if nb == nb_cutouts:
        break
            
print('master cat filled')   

#save
#master_catalogue[1:].to_csv(scratch + 'classify_catalogue.csv') 
