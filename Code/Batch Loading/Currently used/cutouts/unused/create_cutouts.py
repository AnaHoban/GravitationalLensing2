'''Creates cutouts for u, r and u+r (with r catalogue)
   Don't forget to specify the # of tiles you need'''


import os
import h5py
import shutil
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from astropy import table
import numpy as np
import matplotlib.pyplot as plt

import csv
import pandas as pd

#tweakable
cutout_size = 64
start = 0
end = 5 #NUMBER OF TILES TAKEN FROM EACH CHANNEL!

#directories
scratch = os.path.expandvars("$SCRATCH") + '/'
image_dir = "/home/anahoban/projects/rrg-kyi/astro/cfis/W3/"
cutout_dir = scratch

src = os.path.expandvars("$SCRATCH") #+ "/test2.h5"
dest = os.path.expandvars("$SLURM_TMPDIR") + "/"

#hf file
hf = h5py.File(scratch + "testing_create_cuts_noisy.h5", "w")

tile_list = open(image_dir + "tiles.list", "r")

# makes diff id lists for each channel
tile_list = open(image_dir + "tiles.list", "r")
tile_ids_r = []
tile_ids_u = []
tile_ids_g = []
tile_ids_i = []
tile_ids_z = []

tile_ids_u_r = []

i=0
for tile in tile_list:
    tile = tile[:-1] #remove new line character 
    tile_bands = tile.split(" ") 
    if tile_bands == ['']:
        continue
    elif len(tile_bands) == 1 : #just 1 band
        for num_band in range(len(tile_bands)):
            if tile_bands[num_band][-1] == 'r' and (len(tile_ids_r)< end):
                tile_ids_r.append(tile_bands[num_band][5:12])
            if tile_bands[num_band][-1] == 'u' and (len(tile_ids_u)< end):
                tile_ids_u.append(tile_bands[num_band][5:12])
            if tile_bands[num_band][-1] == 'g' and (len(tile_ids_g)< end):
                tile_ids_g.append(tile_bands[num_band][4:11])
            if tile_bands[num_band][-1] == 'i' and (len(tile_ids_i)< end):
                tile_ids_i.append(tile_bands[num_band][4:11])
            if tile_bands[num_band][-1] == 'z' and (len(tile_ids_z)< end):
                tile_ids_z.append(tile_bands[num_band][4:11])

tile_list.close()
tile_list = open(image_dir + "tiles.list", "r")               
i=0
for tile in tile_list: 
    tile = tile[:-1] # Remove new line character
    channels = tile.split(" ") #tile.channel tile.channel 
    if ( i < end ):    #to avoid putting other two-channel combination
        if len(channels) == 2: # Order is u,g,r,i,z
            tile_ids_u_r.append(tile[5:12])
            i+=1
                 
tile_list.close()

#tiles
u_images = [];  u_weights = []
r_images = [];  r_weights = []
ur_images= []; ur_weights = [] 
u_cat = []
r_cat = []
u_r_cat = []

for tile_id in tile_ids_r:
    r_cat.append(image_dir + 'CFIS.'+ tile_id + '.r' + ".cat") 
    r_images.append(image_dir + 'CFIS.'+ tile_id + '.r' + ".fits")
    r_weights.append(image_dir + 'CFIS.'+ tile_id + '.r' + ".weight.fits.fz")   
    
for tile_id in tile_ids_u:
    u_cat.append(image_dir + 'CFIS.'+ tile_id + '.u' + ".cat")
    u_images.append(image_dir + 'CFIS.'+ tile_id + '.u' + ".fits")
    u_weights.append(image_dir + 'CFIS.'+ tile_id + '.u' + ".weight.fits.fz")   
    
for tile_id in tile_ids_u_r:
    u_r_cat.append(image_dir + 'CFIS.'+ tile_id + '.r' + ".cat")
    ur_images.append(image_dir + 'CFIS.'+ tile_id + '.r' + ".fits")
    ur_weights.append(image_dir + 'CFIS.'+ tile_id + '.r' + ".weight.fits.fz")
    
for tile_id in tile_ids_u_r: #so that u and r same tile are n = end-start indices apart
    u_r_cat.append(image_dir + 'CFIS.'+ tile_id + '.u' + ".cat")  
    ur_images.append(image_dir + 'CFIS.'+ tile_id + '.u' + ".fits")
    ur_weights.append(image_dir + 'CFIS.'+ tile_id + '.u' + ".weight.fits.fz")   

print(r_cat , '\n' , u_cat, '\n', u_r_cat)    
    
# Copy tiles to $SLURM_TMPDIR
for n in range(start, end):
    shutil.copy2(u_cat[n], dest)
    shutil.copy2(r_cat[n], dest)
    
    u_cat[n] = os.path.abspath(dest + os.path.basename(u_cat[n]))
    r_cat[n] = os.path.abspath(dest + os.path.basename(r_cat[n]))
    u_r_cat[n] = os.path.abspath(dest + os.path.basename(u_r_cat[n]))
       

#create cutouts function
def create_cutouts(img, wt, x, y, band):
    ''' Creates the image and weight cutouts given a tile, the position of the center and the band '''
    
    img_cutout = Cutout2D(img.data, (x, y), cutout_size, mode="partial", fill_value=0).data
    if np.count_nonzero(np.isnan(img_cutout)) >= 0.05*cutout_size**2 or np.count_nonzero(img_cutout) == 0: # Don't use this cutout
        return (None, None)
    wt_cutout = Cutout2D(wt.data, (x, y), cutout_size, mode="partial", fill_value=0).data
    img_cutout[np.isnan(img_cutout)] = 0
    wt_cutout[np.isnan(wt_cutout)] = 0
    
    img_lower = np.percentile(img_cutout, 0.001)
    #img_lower = np.min(img_cutout)
    img_upper = np.percentile(img_cutout, 99.999)
    
    img_cutout[img_cutout<img_lower] = img_lower
    img_cutout[img_cutout>img_upper] = img_upper
    
    
    if img_lower == img_upper:
        img_norm = np.zeros((cutout_size, cutout_size))
    else:
        img_norm = (img_cutout - np.min(img_cutout)) / (img_upper - img_lower)
        #inorms.append(1/(img_upper - img_lower))
        
    if band in ["i", "g", 'z']: # Alter weights for PS1
        wt_norm = (wt_cutout - np.min(wt_cutout)) / (img_upper - img_lower)
        #wnorms.append(1/(img_upper - img_lower))
    else: # Do not alter weights for CFIS
        wt_norm = wt_cutout
        
    return (img_norm, wt_norm)

#making the cutouts
for n in range(start, end): 
    #cats
    rcat = table.Table.read(r_cat[n], format="ascii.sextractor")
    ucat = table.Table.read(u_cat[n], format="ascii.sextractor")
    urcat= table.Table.read(u_cat[n], format="ascii.sextractor")
    #suggestion: can read off master cat instead!
    
                                            ####### R #######
    img_cutout = np.random.normal(loc = 0.5,scale= 0.13, size=(64,64,5))
    #np.zeros((cutout_size, cutout_size, 5)) #this can be changed to random noise if necessary, don't need to append it in loop
    wt_cutout  = np.zeros((cutout_size, cutout_size, 5))
    #tiles
    r_image  = fits.open(r_images[n] , memmap=True)
    r_weight = fits.open(r_weights[n], memmap=True)
    
    root = os.path.basename(r_images[n])[5:12]  # XXX.XXX id
    img_group = hf.create_group(root + "/IMAGES")
    wt_group  = hf.create_group(root + "/WEIGHTS")
    
    count=0
    for i in range(len(rcat)): #each cutout in tile
        if rcat["FLAGS"][i] != 0 or rcat["MAG_AUTO"][i] >= 99.0 or rcat["MAGERR_AUTO"][i] <= 0 or rcat["MAGERR_AUTO"][i] >= 1:
            continue
            
        x = rcat["X_IMAGE"][i]
        y = rcat["Y_IMAGE"][i]
        
        (r_img, r_wt) = create_cutouts(r_image[0], r_weight[1], x, y, "r")
        
        if r_img is None:
            continue
        img_cutout[:,:,2] = r_img  
        wt_cutout[:,:,2]  = r_wt
        count += 1
        

        img_group.create_dataset(f"c{count}", data=img_cutout)
        wt_group.create_dataset(f"c{count}", data=wt_cutout)
    
    print(f'tile {root} in r done')
    r_image.close()
    r_weight.close()
    
    
                                            ####### U #######
    img_cutout = np.random.normal(loc = 0.5,scale= 0.13, size=(64,64,5))
    #img_cutout = np.zeros((cutout_size, cutout_size, 5)) #this can be changed to random noise if necessary, don't need to append it in loop
    wt_cutout  = np.zeros((cutout_size, cutout_size, 5))
    #tiles
    u_image  = fits.open(u_images[n] , memmap=True)
    u_weight = fits.open(u_weights[n], memmap=True)
    
    count=0
    #cutouts from u band
    root = os.path.basename(u_images[n])[5:12]  # XXX.XXX id
    
    img_group = hf.create_group(root + "/IMAGES")
    wt_group = hf.create_group(root + "/WEIGHTS")    
    
    for j in range(len(ucat)): #each cutout in tile
        if ucat["FLAGS"][j] != 0 or ucat["MAG_AUTO"][j] >= 99.0 or ucat["MAGERR_AUTO"][j] <= 0 or ucat["MAGERR_AUTO"][j] >= 1:
            continue
        x = ucat["X_IMAGE"][j]
        y = ucat["Y_IMAGE"][j]
        
        (u_img, u_wt) = create_cutouts(u_image[0], u_weight[1], x, y, "u")
        if u_img is None:
            continue
            
        img_cutout[:,:,0] = u_img
        wt_cutout[:,:,0] = u_wt            
       
        img_group.create_dataset(f"c{count}", data=img_cutout)
        wt_group.create_dataset(f"c{count}", data=wt_cutout)            
        count += 1    
    u_image.close()
    u_weight.close()
    
    print(f'tile {root} in u done')
    
                                            ####### U & R #######    
    img_cutout = np.zeros((cutout_size, cutout_size, 5)) #this can be changed to random noise if necessary, don't need to append it in loop
    wt_cutout  = np.zeros((cutout_size, cutout_size, 5))
    #tiles
    ur_r_image = fits.open(ur_images[n] , memmap=True)
    ur_r_weight = fits.open(ur_weights[n] , memmap=True)
    ur_u_image = fits.open(ur_images[n+end] , memmap=True)
    ur_u_weight = fits.open(ur_weights[n+end] , memmap=True)    
    
    count = 0
    root = os.path.basename(ur_images[n])[5:12]  # XXX.XXX id
    img_group = hf.create_group(root + "/IMAGES")
    wt_group = hf.create_group(root + "/WEIGHTS") 
    
    for k in range(len(rcat)): #each cutout in tile
        if rcat["FLAGS"][k] != 0 or rcat["MAG_AUTO"][k] >= 99.0 or rcat["MAGERR_AUTO"][k] <= 0 or rcat["MAGERR_AUTO"][k] >= 1:
            continue
        
        x = rcat["X_IMAGE"][k]
        y = rcat["Y_IMAGE"][k]
        
        (r_img, r_wt) = create_cutouts(ur_r_image[0], ur_r_weight[1], x, y, "r")
        if r_img is None:
            continue
        (u_img, u_wt) = create_cutouts(ur_u_image[0], ur_u_weight[1], x, y, "u")
        if u_img is None:
            continue
  

        img_cutout[:,:,0] = u_img
        img_cutout[:,:,2] = r_img
        
        wt_cutout[:,:,0] = u_wt
        wt_cutout[:,:,2] = r_wt

        img_group.create_dataset(f"c{count}", data=img_cutout)
        wt_group.create_dataset(f"c{count}", data=wt_cutout)
        count += 1
    
    ur_r_image.close()
    ur_r_weight.close()
    ur_u_image.close()
    ur_u_weight.close
    print(f'tile {root} in u and r done')
    
    
    print(f"Tile {n+1} completed")
    
    
hf.close()
