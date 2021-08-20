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

#useful directories
scratch = os.path.expandvars("$SCRATCH")
tmp_dir = os.path.expandvars("$SLURM_TMPDIR")

print(tmp_dir)

image_dir = "/home/anahoban/projects/rrg-kyi/astro/cfis/W3/"
label_dir = "../labels/"

cut_dir = scratch + '/all_cand/'

##########################

#### Useful lists #####
label_subdirs = ["stronglensdb_confirmed_unige/", "stronglensdb_candidates_unige/", "canameras2020/",
                 "huang2020a_grade_A/", "huang2020a_grade_B/", "huang2020a_grade_C/", 
                 "huang2020b_grade_A/", "huang2020b_grade_B/", "huang2020b_grade_C/"]
filters = ["CFIS u/", "PS1 g/", "CFIS r/", "PS1 i/", "PS1 z/"]
filter_dict = {k:v for v,k in enumerate(filters)}

all_tiles = os.listdir(image_dir)

##### Clean directory ####
for label_subdir in label_subdirs:
    for f in [filters[2],filters[0]]:
        subdir = label_dir + label_subdir + f
        #print(subdir)
        z = 1
        for csv in os.listdir(subdir):
            if csv == '.ipynb_checkpoints':
                del csv
            else:
                continue
                #print(csv)
################                

#### Create cutouts ######

broken_tiles = []
prev_conf = {'197.271': None} #dictionary of previously seen tiles
cutout_size = 124
for label_subdir in label_subdirs:
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

                tile_name = f.split(" ")[0] + "." + tile_id + "." + f.split(" ")[1][0]
                print('TILE', tile_name)
               
                #FILES
                #weight
                if "CFIS" in f:
                    wt_name = ".weight.fits.fz"
                    wt_index = 1
                else:
                    wt_name = ".wt.fits"
                    wt_index = 0
                
                #files
                # Load the image and the WCS
                shutil.copy2(image_dir + tile_name + ".fits", tmp_dir)
                print('copied to slurm')
                img_slurm = os.path.abspath(tmp_dir +'/'+ tile_name + ".fits")
                img_fits = fits.open(img_slurm, memmap=True)[0]
                print('opened file')
                wcs = WCS(img_fits.header)
                
                #wt_fits  = fits.open(image_dir + tile_name + wt_name, memmap=True)
                #cat = table.Table.read(image_dir + tile_name + '.cat', format="ascii.sextractor")
                
                g_band = False
                i_band = False
                z_band = False
                o_band = False
                
                #see if it's available in other bands
                if 'PS1.' + tile_id +'.z.fits' in all_tiles:
                    z_band = True
                    shutil.copy2(image_dir + 'PS1.' +  tile_id + '.z'  + ".fits", tmp_dir)
                    img_slurm_z = os.path.abspath(tmp_dir +'/'+ 'PS1.' + tile_id + '.z' + ".fits")
                    img_fits_z = fits.open(img_slurm_z, memmap=True)[0]
                    wcs_z = WCS(img_fits_z.header)
                    print('z found')
                    
                if 'PS1.' + tile_id +'.g.fits' in all_tiles:
                    g_band = True
                    shutil.copy2(image_dir + 'PS1.' +  tile_id + '.g'  + ".fits", tmp_dir)
                    img_slurm_g = os.path.abspath(tmp_dir +'/'+ 'PS1.' + tile_id + '.g' + ".fits")
                    img_fits_g = fits.open(img_slurm_g, memmap=True)[0]
                    wcs_g = WCS(img_fits_g.header)
                    print('g found')
                
                if 'PS1.' + tile_id +'.i.fits' in all_tiles:
                    i_band = True
                    shutil.copy2(image_dir + 'PS1.' + tile_id + '.i'  + ".fits", tmp_dir)
                    img_slurm_i = os.path.abspath(tmp_dir +'/'+ 'PS1.' + tile_id + '.i' + ".fits")
                    img_fits_i = fits.open(img_slurm_i, memmap=True)[0]
                    wcs_i = WCS(img_fits_i.header)
                    print('i found')
                    
                if 'CFIS.' + tile_id + '.r_cutout_0.fits' not in os.listdir(cut_dir) and tile_id + '.r' != tile_name and 'CFIS.' + tile_id + '.r_cutout_0.fits' in all_tiles:
                    o_band = True
                    shutil.copy2(image_dir + 'CFIS.' + tile_id + '.r'  + ".fits", tmp_dir)
                    img_slurm_o = os.path.abspath(tmp_dir +'/'+ 'PS1.' + tile_id + '.r' + ".fits")
                    img_fits_o = fits.open(img_slurm_o, memmap=True)[0]
                    wcs_o = WCS(img_fits_o.header)
                    print('r found')
                    
                elif 'CFIS.' + tile_id + '.u_cutout_0.fits' not in os.listdir(cut_dir) and tile_id + '.u' != tile_name and 'CFIS.' + tile_id + '.u_cutout_0.fits' in all_tiles:
                    o_band = True
                    shutil.copy2(image_dir + 'CFIS.'+tile_id + '.u'  + ".fits", tmp_dir)
                    img_slurm_o = os.path.abspath(tmp_dir +'/'+'CFIS.' +  tile_id + '.u' + ".fits")
                    img_fits_o = fits.open(img_slurm_o, memmap=True)[0]
                    wcs_o = WCS(img_fits_o.header)
                    print('u found')

                #candidates cutouts
                df = pd.read_csv(subdir + csv)
                nlabels = len(df)

            #----create candidate cutout from current catalogue----#    
                print('creating candidates')
                count = 0
                for n in range(nlabels): 
                    x = df["x"][n]
                    y = df["y"][n]
                    #save x,y for r channel
                    #make cutout
                    try:
                        # Load the image and the WCS
                        wcs = WCS(img_fits.header)

                        # Make the cutout, including the WCS
                        cutout = Cutout2D(img_fits.data, position=(x,y), size=cutout_size, wcs=wcs)

                        # Put the cutout image in the FITS HDU
                        img_fits.data = cutout.data

                        # Update the FITS header with the cutout WCS
                        img_fits.header.update(cutout.wcs.to_header())

                        # Write the cutout to a new FITS file
                        cutout_filename = cut_dir + tile_name + f'_cut_{count}.fits'
                        print(cutout_filename)
                        img_fits.writeto(cutout_filename, overwrite=True)
                        print('cutout created')
                        os.remove(img_slurm)
                        
                    except:
                        broken_tiles.append(tile_name) 
                        pass
                    
                    if g_band == True:
                        try:
                            # Load the image and the WCS
                            wcs = WCS(img_fits_g.header)

                            # Make the cutout, including the WCS
                            cutout = Cutout2D(img_fits_g.data, position=(x,y), size=cutout_size, wcs=wcs)

                            # Put the cutout image in the FITS HDU
                            img_fits.data = cutout.data

                            # Update the FITS header with the cutout WCS
                            img_fits.header.update(cutout.wcs.to_header())

                            # Write the cutout to a new FITS file
                            cutout_filename = cut_dir + tile_id + '.g'+ f'_cut_{count}.fits'
                            print(cutout_filename)
                            img_fits.writeto(cutout_filename, overwrite=True)
                            print('g cutout created')
                            os.remove(img_slurm_g)
                        except:
                            broken_tiles.append(tile_id + '.g') 
                            pass
                        
                        
                    if i_band == True:
                        try:
                            # Load the image and the WCS
                            wcs = WCS(img_fits_i.header)

                            # Make the cutout, including the WCS
                            cutout = Cutout2D(img_fits_i.data, position=(x,y), size=cutout_size, wcs=wcs)

                            # Put the cutout image in the FITS HDU
                            img_fits.data = cutout.data

                            # Update the FITS header with the cutout WCS
                            img_fits.header.update(cutout.wcs.to_header())

                            # Write the cutout to a new FITS file
                            cutout_filename = cut_dir + tile_id + '.i'+ f'_cut_{count}.fits'
                            print(cutout_filename)
                            img_fits.writeto(cutout_filename, overwrite=True)
                            print('i cutout created')
                            os.remove(img_slurm_i)
                        except:
                            broken_tiles.append(tile_id + '.i') 
                            pass
                        
                        
                    if z_band == True:
                        try:
                            # Load the image and the WCS
                            wcs = WCS(img_fits_z.header)

                            # Make the cutout, including the WCS
                            cutout = Cutout2D(img_fits_z.data, position=(x,y), size=cutout_size, wcs=wcs)

                            # Put the cutout image in the FITS HDU
                            img_fits.data = cutout.data

                            # Update the FITS header with the cutout WCS
                            img_fits.header.update(cutout.wcs.to_header())

                            # Write the cutout to a new FITS file
                            cutout_filename = cut_dir + tile_id + '.z' + f'_cut_{count}.fits'
                            print(cutout_filename)
                            img_fits.writeto(cutout_filename, overwrite=True)
                            print('z cutout created')
                            os.remove(img_slurm_z)
                        except:
                            broken_tiles.append(tile_id + '.z' ) 
                            pass
                        
                        
                    if o_band == True:
                        try:
                            # Load the image and the WCS
                            wcs = WCS(img_fits_o.header)

                            # Make the cutout, including the WCS
                            cutout = Cutout2D(img_fits_o.data, position=(x,y), size=cutout_size, wcs=wcs)

                            # Put the cutout image in the FITS HDU
                            img_fits.data = cutout.data

                            # Update the FITS header with the cutout WCS
                            img_fits.header.update(cutout.wcs.to_header())

                            # Write the cutout to a new FITS file
                            if tile_id + '.u' != tile_name:
                                tile_name = tile_id + '.u'
                            elif tile_id + '.r' != tile_name:
                                tile_name = tile_id + '.r'
                            cutout_filename = cut_dir + tile_name + f'_cut_{count}.fits'
                            print(cutout_filename)
                            img_fits.writeto(cutout_filename, overwrite=True)
                            print('u/r cutout created')
                            os.remove(img_slurm_o)
                        except:
                            broken_tiles.append(tile_id) 
                            pass
                       
                    
                    count += 1
            
        print(f"Finished {label_subdir}")

print('done')
print(broken_tiles)
np.savetxt("broken_tiles.csv", broken_tiles, delimiter =", ", fmt ='% s')

