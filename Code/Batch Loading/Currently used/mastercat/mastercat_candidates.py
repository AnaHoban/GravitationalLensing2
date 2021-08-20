import os
import h5py
import shutil
import pandas as pd
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from astropy import table
import numpy as np
import random
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
from astropy.visualization import make_lupton_rgb
from collections import Counter
from tensorflow import keras
from astropy.wcs import WCS,utils


print('Starting \n' )

#### FILE + SETUP ####

file_name = "/confirmed_cfis_64p.h5"

#useful directories
os.path.expandvars("$SLURM_TMPDIR")

scratch = os.path.expandvars("$SCRATCH")
tmp_dir = os.path.expandvars("$SLURM_TMPDIR")

image_dir = "/home/anahoban/projects/rrg-kyi/astro/cfis/W3/"
label_dir = "../labels/"

#hf = h5py.File(scratch + file_name, "w")
#hf.close()

hf = h5py.File(scratch + file_name, "r")

hf_ids = list(hf.keys())

#### Useful lists #####
label_subdirs = ["stronglensdb_confirmed_unige/", "stronglensdb_candidates_unige/", "canameras2020/",
                 "huang2020a_grade_A/", "huang2020a_grade_B/", "huang2020a_grade_C/", 
                 "huang2020b_grade_A/", "huang2020b_grade_B/", "huang2020b_grade_C/"]
filters = ["CFIS u/", "PS1 g/", "CFIS r/", "PS1 i/", "PS1 z/"]
filter_dict = {k:v for v,k in enumerate(filters)}

#### Create cutouts ######

cutout_size = 64
tile_ids_r = []
tile_ids_u = []
for label_subdir in [label_subdirs[0]]:
    for f in [filters[2]]:
        subdir = label_dir + label_subdir + f
        for csv in os.listdir(subdir):
            if csv == '.ipynb_checkpoints':
                 del csv
            else:
                tile_ids_r.append(csv[:7]) # XXX.XXX id
    for f in [filters[0]]:
        subdir = label_dir + label_subdir + f
        for csv in os.listdir(subdir):
            if csv == '.ipynb_checkpoints':
                 del csv
            else:
                tile_ids_u.append(csv[:7]) # XXX.XXX id
##########################

#master catalogue
example = table.Table.read(image_dir + 'CFIS.' + '004.180.u' + '.cat', format="ascii.sextractor")
keys = example.keys()
master_catalogue = pd.DataFrame(index = [0], columns = keys + ['TILE'] + ['BAND'] + ['CUTOUT'])

#catalogue of r cutouts
print('r')
for tile in hf_ids:
    print('tile', tile)
    
    if tile in tile_ids_r:
        cat_r = table.Table.read(image_dir + 'CFIS.' + tile + '.r.cat', format="ascii.sextractor")
        for cutout in list(hf.get(tile+ '/IMAGES')): 
            print('tile in u', cutout)
            if cutout[-1] in ['u', 'r']:
                #do stuff later
                continue
            else:
                data = cat_r[int(cutout[1:])]
                print(tile, cutout)
            #store info into mastercat
            #r
                new_cutout = pd.DataFrame(index = [int(cutout[1:])], data=np.array(data), columns = keys + ['TILE'] + ['BAND'] + ['CUTOUT'])
                new_cutout['BAND'] = 'r'
                new_cutout['TILE'] = tile
                new_cutout['CUTOUT'] = cutout
                
                master_catalogue = master_catalogue.append(new_cutout)
            
            
    if tile in tile_ids_u and tile not in tile_ids_r:
        print('this tile is only in u', tile)
        cat_u = table.Table.read(image_dir + 'CFIS.' + tile + '.u.cat', format="ascii.sextractor")
        for cutout in list(hf.get(tile + '/IMAGES')):       
            print('tile in u', cutout)
            if cutout[-1] in ['u', 'r']:
                #do stuff later
                continue
            else:
                print('storing', cutout)
                print(int(cutout[1:]))
                data = cat_u[int(cutout[1:])] 
            #store info into mastercat
            #u
                new_cutout = pd.DataFrame(index = [int(cutout[1:])], data=np.array(data), columns = keys + ['TILE'] + ['BAND'] + ['CUTOUT'])
                new_cutout['BAND'] = 'u'
                new_cutout['TILE'] = tile
                new_cutout['CUTOUT'] = cutout
            
            
                master_catalogue = master_catalogue.append(new_cutout)
hf.close()
#save
master_catalogue.to_csv(scratch + '/master_catalogue.csv') 