import os
import h5py
import shutil
from astropy.io import fits
from astropy import table
import numpy as np
import matplotlib.pyplot as plt

import csv
import pandas as pd

print('Starting \n' )
#tweakable
cutout_size = 64
start = 0
end = 4 #NUMBER OF TILES TAKEN FROM EACH CHANNEL!

#directories
scratch = os.path.expandvars("$SCRATCH") + '/'
image_dir = "/home/anahoban/projects/rrg-kyi/astro/cfis/W3/"
cutout_dir = scratch

src = os.path.expandvars("$SCRATCH") #+ "/test2.h5"
dest = os.path.expandvars("$SLURM_TMPDIR") + "/"


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

u_cat = []
r_cat = []
u_r_cat = []

for tile_id in tile_ids_r:
    r_cat.append(image_dir + 'CFIS.'+ tile_id + '.r' + ".cat")     
    
for tile_id in tile_ids_u:
    u_cat.append(image_dir + 'CFIS.'+ tile_id + '.u' + ".cat")
    
for tile_id in tile_ids_u_r:
    r_cat.append(image_dir + 'CFIS.'+ tile_id + '.r' + ".cat")
    u_cat.append(image_dir + 'CFIS.'+ tile_id + '.u' + ".cat")
    
print(r_cat, u_cat)    
    
# Copy tiles to $SLURM_TMPDIR
for n in range(start, end+end):
    shutil.copy2(u_cat[n], dest)
    shutil.copy2(r_cat[n], dest)
    
    u_cat[n] = os.path.abspath(dest + os.path.basename(u_cat[n]))
    r_cat[n] = os.path.abspath(dest + os.path.basename(r_cat[n]))
    
print(r_cat,u_cat)    
#prepare master cat
example = table.Table.read(r_cat[1], format="ascii.sextractor")
keys = example.keys()
master_catalogue = pd.DataFrame(index = [0], columns = keys + ['TILE'] + ['BAND'] + ['CUTOUT'])

print('master cat created \n')
#populate master cat

for n in range(start, end+end): #single and both channels
    rcat = table.Table.read(r_cat[n], format="ascii.sextractor")
    ucat = table.Table.read(u_cat[n], format="ascii.sextractor")
    
    #cutouts from r band
    root = os.path.basename(r_cat[n])[5:12]  # XXX.XXX id
    
    count=0
    for i in range(len(rcat)): #each cutout in tile
        if rcat["FLAGS"][i] != 0 or rcat["MAG_AUTO"][i] >= 99.0 or rcat["MAGERR_AUTO"][i] <= 0 or rcat["MAGERR_AUTO"][i] >= 1:
            continue
            
        new_cutout = pd.DataFrame(index = [i], data=np.array(rcat[i]), columns = keys + ['TILE'] + ['BAND'] + ['CUTOUT'])
        new_cutout['BAND'] = 'r'
        new_cutout['TILE'] = f"{root}"
        new_cutout['CUTOUT'] = f"c{count}"
        
        master_catalogue = master_catalogue.append(new_cutout)
        count += 1
    
    print(f'tile {root} in r done')
    
    count=0
    #cutouts from u band
    root = os.path.basename(u_cat[n])[5:12]  # XXX.XXX id
        
    for j in range(len(ucat)): #each cutout in tile
        if ucat["FLAGS"][j] != 0 or ucat["MAG_AUTO"][j] >= 99.0 or ucat["MAGERR_AUTO"][j] <= 0 or ucat["MAGERR_AUTO"][j] >= 1:
            continue
            
        new_cutout = pd.DataFrame(index = [j], data=np.array(ucat[j]), columns = keys + ['TILE'] + ['BAND'] + ['CUTOUT'])
        new_cutout['BAND'] = 'u'
        new_cutout['TILE'] = f"{root}"
        new_cutout['CUTOUT'] = f"c{count}"
        
        master_catalogue = master_catalogue.append(new_cutout)
        count += 1
    print(f'tile {root} in u done')
    
    print(f"Tile {n+1} completed")
    
    
#save
master_catalogue.to_csv(scratch + 'master_catalogue.csv') 
