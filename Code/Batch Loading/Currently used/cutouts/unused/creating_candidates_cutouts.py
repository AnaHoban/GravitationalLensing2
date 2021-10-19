import os
import sys
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

sys.path.append('../')
from functions import create_cutouts

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

shutil.copy2(scratch + file_name, tmp_dir)


hf = h5py.File(scratch + file_name, "w")

##########################

#### Useful lists #####
tile_file = open(image_dir + "tiles_cand.list", "r")
tile_list = tile_file.readlines()
for i in range(len(tile_list)):
    tile_list[i] = tile_list[i][:-1] # Remove new line characters
    print(tile_list[i])
tile_file.close()

label_subdirs = ["stronglensdb_confirmed_unige/", "stronglensdb_candidates_unige/", "canameras2020/",
                 "huang2020a_grade_A/", "huang2020a_grade_B/", "huang2020a_grade_C/", 
                 "huang2020b_grade_A/", "huang2020b_grade_B/", "huang2020b_grade_C/"]
filters = ["CFIS u/", "PS1 g/", "CFIS r/", "PS1 i/", "PS1 z/"]
filter_dict = {k:v for v,k in enumerate(filters)}

##### Clean directory ####
for label_subdir in [label_subdirs[0]]:
    for f in [filters[2],filters[0]]:
        subdir = label_dir + label_subdir + f
        print(subdir)
        z = 1
        for csv in os.listdir(subdir):
            if csv == '.ipynb_checkpoints':
                del csv
            else:
                print(csv)
################                

#### Create cutouts ######

broken_tiles = []
prev_conf = {'197.271': None} #dictionary of previously seen tiles
cutout_size = 64
for label_subdir in [label_subdirs[0]]:
    for f in [filters[2],filters[0]]:
        subdir = label_dir + label_subdir + f
        print(subdir)
#         z = 0
        for csv in os.listdir(subdir):
            if csv == '.ipynb_checkpoints':
                 del csv
            else:
#               if z <1:
#                 z+=1
                tile_id = csv[:7] # XXX.XXX id
                print('Tile', tile_id)    
    #                 img_group_name = label_subdir + tile_id + "/" + f + "IMAGES"
    #                 wt_group_name = label_subdir + tile_id + "/" + f + "WEIGHTS"

                img_group_name =  tile_id + "/" + "IMAGES"
                wt_group_name  =  tile_id + "/" + "WEIGHTS"
                tile_name = f.split(" ")[0] + "." + tile_id + "." + f.split(" ")[1][0]
               
                #FILES
                #weight
                if "CFIS" in f:
                    wt_name = ".weight.fits.fz"
                    wt_index = 1
                else:
                    wt_name = ".wt.fits"
                    wt_index = 0
                
                #files
                img_fits = fits.open(image_dir + tile_name + ".fits", memmap=True)
                wt_fits  = fits.open(image_dir + tile_name + wt_name, memmap=True)
                cat = table.Table.read(image_dir + tile_name + '.cat', format="ascii.sextractor")
                    
                new_group = False
                if img_group_name not in hf:
                    new_group = True
                    img_group = hf.create_group(img_group_name)
                    wt_group = hf.create_group(wt_group_name)

                    img_cutout_all = np.random.normal(loc = 0.5,scale= 0.13, size=(cutout_size, cutout_size,5))
                    #img_cutout = np.zeros((cutout_size, cutout_size, 5)) 
                    wt_cutout_all  = np.zeros((cutout_size, cutout_size, 5))

                    #generate list of random cutouts
                    rand_cutouts = random.sample(range(len(cat)), 400)
#                     rand_cutouts = random.sample(range(len(cat)), 2)

                else:
                    img_group = hf[img_group_name]
                    wt_group = hf[wt_group_name]

                    #want the same cutouts as previous, with previous catalogue (r)
                    list_cutouts = list(hf[img_group_name].keys())
                    #prev random cutouts
                    rand_cutouts = [int((i[1:]).split("-")[0]) for i in list_cutouts if  'r' not in (i[1:]).split("-")] 
                    #prev confirmed cutouts
                    cat_prev = table.Table.read(image_dir + tile_name[:-1] + 'r' + '.cat', format="ascii.sextractor")


                #### make cutouts
                count = 0 
                #candidates cutouts
                df = pd.read_csv(subdir + csv)
                nlabels = len(df)
                
            #----create candidate cutout from previous catalogue----#    
                if (tile_id + '-0') in prev_conf:
                    j = 0 
                    try_tile = tile_id + '-0' #first cutout is labelled with 0 for sure
                    if try_tile in prev_conf:
                        (x,y) =prev_conf[tile_id + '-' + str(count)] 
                        try:
                            (img_cutout, wt_cutout) = create_cutouts(img_fits[0], wt_fits[wt_index], x, y, tile_name[-1])
                            hf[img_group_name + f"/c{j}-r"][...,filter_dict[f]] = img_cutout
                            hf[wt_group_name + f"/c{j}-r"][...,filter_dict[f]] = wt_cutout
                        except:
                            broken_tiles.append(tile_name) 
                        count += 1
                   
                        
                        j += 1
                        try_tile = tile_id + f'-{j}'
                        print('done r cutout in u')
                    else:
                        print(f'done all {j} old cutout(s) in new (u) band')
                        break
                    
                    print('onto new confirmed u cutouts')
                    
            #----create candidate cutout from current catalogue----#    
                count = 0
                for n in range(nlabels): 
                    x = df["x"][n]
                    y = df["y"][n]
                    #save x,y for r channel
                    if tile_name[-1] == 'r':
                        prev_conf[tile_id + '-' + str(count)] = (x,y)
                    #make cutout
                    try:
                        (img_cutout, wt_cutout) = create_cutouts(img_fits[0], wt_fits[wt_index], x, y, tile_name[-1])
                    except:
                        broken_tiles.append(tile_name) 
                        pass

                    count += 1
                    
                    print('new candidate')
                    img_cutout_all[:,:,filter_dict[f]] = img_cutout
                    wt_cutout_all[:,:,filter_dict[f]]  = wt_cutout

                    img_group.create_dataset(f"c{n}-"+tile_name[-1], data=img_cutout_all)
                    wt_group.create_dataset(f"c{n}-"+tile_name[-1], data=wt_cutout_all)  
                print(f'done {count} new candidates')
                
            #----create extra random cutouts from current catalogue----#   
                for n in rand_cutouts:
                    #if cat["FLAGS"][n] != 0 or cat["MAG_AUTO"][n] >= 99.0 or cat["MAGERR_AUTO"][n] <= 0 or cat["MAGERR_AUTO"][n] >= 1:
                    #    continue
                
                    count += 1
                    if new_group is True: #create cutout from current catalogue if new group
                        print('new random')
                        x = cat["X_IMAGE"][n]
                        y = cat["Y_IMAGE"][n]
                        try:
                            (img_cutout, wt_cutout) = create_cutouts(img_fits[0], wt_fits[wt_index], x, y, f[-2])
                        except:
                            broken_tiles.append(tile_name) 
                            pass
                
                        img_cutout_all[:,:,filter_dict[f]] = img_cutout
                        wt_cutout_all[:,:,filter_dict[f]]  = wt_cutout
                        
                        img_group.create_dataset(f"c{n}", data=img_cutout_all)
                        wt_group.create_dataset(f"c{n}", data=wt_cutout_all)
                    
                    else:                 #add cutout to existing group
                        print('adding to previous random')
                        x = cat_prev["X_IMAGE"][n]
                        y = cat_prev["Y_IMAGE"][n]
                        try:
                            (img_cutout, wt_cutout) = create_cutouts(img_fits[0], wt_fits[wt_index], x, y, f[-2])
                        except:
                            broken_tiles.append(tile_name) 
                            pass

                        if img_cutout is None:
                            img_cutout = np.random.normal(loc = 0.5,scale= 0.13, size=(cutout_size, cutout_size))
                            wt_cutout  = np.zeros((cutout_size, cutout_size))
                        else:
                            hf[img_group_name + f"/c{n}"][...,filter_dict[f]] = img_cutout
                            hf[wt_group_name + f"/c{n}"][...,filter_dict[f]] = wt_cutout
                    print(len(rand_cutouts),n, count)
                print(f'done {count}')
       
                img_fits.close()
                wt_fits.close()
        print(f"Finished {label_subdir}")
        
hf.close()
print('done')
print(broken_tiles)
np.savetxt("broken_tiles.csv", broken_tiles, delimiter =", ", fmt ='% s')
#shutil.copy2(tmp_dir + file_name, scratch)
