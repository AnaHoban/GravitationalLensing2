import os
import shutil
import h5py
import random
import numpy as np
from astropy.nddata.utils import Cutout2D
import fitsio
from astropy.io import fits
from astropy.table import Table
import pandas as pd
from astropy.visualization import (ZScaleInterval, ImageNormalize)
from astropy.wcs import WCS,utils
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel

import warnings
warnings.filterwarnings("ignore")

#important parameters:
cutout_size = 200
cut_per_tile = 200
mag_mode = 24.665777 - 0.5 #based on r catalogue on 76 million cutouts
hf_file_name = 'Dataset_run2_4.h5'

#directories
label_dir = "/home/anahoban/projects/def-sfabbro/anahoban/lensing/GravitationalLensing/Code/Batch Loading/Currently used/labels/"
lenses_list_dir = "/home/anahoban/projects/def-sfabbro/anahoban/lensing/GravitationalLensing/Code/Batch Loading/Currently used/cutouts/"
scratch = os.path.expandvars("$SCRATCH") + '/'
files_dir = scratch + 'all_lenses/'
tmp_dir = os.path.expandvars("$SLURM_TMPDIR") + '/'

#data files
lenses_in_survey = pd.read_csv('lenses_in_survey.csv', dtype={'Tile':str} )
tiles_with_lenses = list(lenses_in_survey['Tile'].unique())

#keys for master catalogue
ex_cat = files_dir + 'CFIS.005.246.u.cat'
example = Table.read(ex_cat, format="ascii.sextractor")
keys_mastercat = example.keys() + ['tile', 'cat_band', 'index']


#other important dicts and keys
keys = ['CFIS_u', 'CFIS_r', 'PS1_g', 'PS1_i', 'PS1_z','HSC_g']
key_dict = {'CFIS_u': 'u', 'CFIS_r': 'r', 'PS1_g' : 'G', 'PS1_i': 'i', 'PS1_z': 'z','HSC_g': 'g'}
band2cam = {v: k[:-2] for k, v in key_dict.items()}
band2index = {'u':0, 'r':1, 'g':2, 'i':3, 'z':4}

band_to_mask = { 'u' : '.weight.fits.fz',
                 'r' : '.weight.fits.fz',
                 'i' : '.wt.fits',
                 'z' : '.wt.fits',
                 'G' : '.wt.fits',
                 'g' : '.weight.fits.fz'}

band_to_dir = { 'u' : 'vcp -v vos:cfis/tiles_DR3/CFIS.',
                'r' : 'vcp -v vos:cfis/tiles_DR3/CFIS.',
                'i' : 'vcp -L -v vos:cfis/ps_tiles/PS1.',
                'z' : 'vcp -L -v vos:cfis/ps_tiles/PS1.',
                'G' : 'vcp -L -v vos:cfis/ps_tiles/PS1.',
                'g' : 'vcp -v vos:cfis/hsc/stacks2/HSC.'}

######################## FUNCTIONS ##########################
#create cutouts function
def make_cutout(img, x, y):
    ''' Creates the image and weight cutouts given a tile, the position of the center and the band '''
    
    img_cutout = Cutout2D(img.data, (x, y), cutout_size, mode="partial", fill_value=0).data
        
    if np.count_nonzero(np.isnan(img_cutout)) >= 0.05*cutout_size**2 or np.count_nonzero(img_cutout) == 0: # Don't use this cutout
        return np.zeros((cutout_size,cutout_size))
    
    img_cutout[np.isnan(img_cutout)] = 0
        
    return img_cutout

def get_nonlens_list(catalogue_filename, nb_cutouts):
    '''Making a cutouts list for each catalogue'''
    cat = Table.read(catalogue_filename, format="ascii.sextractor") 
    tile_id = catalogue_filename[-13:-6]
    
    #make magnitude cut
    sample_cat = cat[cat['MAG_AUTO'] <= mag_mode].to_pandas().sample(400)
    
    data= []  
    count=0
    for index, row in sample_cat.iterrows(): #each cutout in tile
        if row["FLAGS"].item() != 0 or row["MAGERR_AUTO"].item() <= 0 or row["MAGERR_AUTO"].item() >= 1:
            continue

        new_cutout = list(row) + [f"{tile_id}", catalogue_filename[-5], f"c{count}"] #append all row, tile_id, band, index 
        data.append(new_cutout)
        count +=1 
        if count == nb_cutouts:
            return data
        
def ra_dec_to_xy(img_filename, ra, dec, band):
    fits = fitsio.FITS(img_filename)
    if band in ['i','z']:
        head = fits[1].read_header()
    else:
        head = fits[0].read_header()
    w = WCS(head)
    return skycoord_to_pixel(SkyCoord(ra, dec, unit="deg"), w)



############ CODE STARTS HERE ################
#list to create master catalogue in the end
all_non_lenses = []
#os.remove(scratch+hf_file_name)
hf = h5py.File(scratch + hf_file_name, "w")

print('starting cutout extraction')

for tile in tiles_with_lenses[1500:]:    
    try: #in case there is a problem somewhere for this tile
        #candidates coordindates for a tile
        print(tile)

        #ra_dec list of candidates
        ra_dec_list = [(row.RA, row.DEC) for index, row in lenses_in_survey[lenses_in_survey.Tile == tile][['RA','DEC']].iterrows()]
        n_cand = len(ra_dec_list)
        n = 0

        #available bands for this tile   
        bands = list(lenses_in_survey.loc[lenses_in_survey.Tile == tile, 'bands_list'].iloc[0]) #array of bands

        cat_name = None #set it to none in case there is no u,r,g band for that cutout
    
        #get r,g, or u catalogue
        if len(bands) > 0:
            n = cut_per_tile
            if 'r' in bands:
                cat_name = 'CFIS.' + tile + '.r.cat'
            elif 'g' in bands:
                cat_name =  'HSC.' + tile + '.g.cat'
            elif 'u' in bands:
                cat_name = 'CFIS.' + tile + '.u.cat'


            shutil.copy2(files_dir + cat_name, tmp_dir)

            non_lens = get_nonlens_list(tmp_dir + cat_name, n)
            all_non_lenses += non_lens   

        cut_array = np.zeros((n+n_cand, cutout_size, cutout_size, 10)) 

        #make cutouts
        for band in bands:
            print(band)
            if band == 'G': #not using ps1 g band
                break
                
            cam = band2cam[band]

            #get img and weight files
            img_name = cam + '.' + tile + '.' +  band + '.fits'
            wt_name =  cam + '.' + tile + '.' +  band + band_to_mask[band]     

            # Copy tiles to $SLURM_TMPDIR
            shutil.copy2(files_dir + img_name, tmp_dir)
            shutil.copy2(files_dir +  wt_name, tmp_dir)

            #open fits file
            image = fits.open(tmp_dir + img_name, memmap=True)
            weight = fits.open(tmp_dir + wt_name, memmap=True)

            for count, ra_dec in enumerate(ra_dec_list):
                ra,dec = ra_dec        
                X, Y = ra_dec_to_xy(tmp_dir + img_name, ra, dec, band)

                if band in ['i','z']:
                    img = make_cutout(image[1], X.item(), Y.item())
                else:
                    img = make_cutout(image[0], X.item(), Y.item())

                wt  = make_cutout(weight[1], X.item(), Y.item())

                #add to cutout collection
                cut_array[count,:,:,band2index[band]] = img 
                cut_array[count,:,:,band2index[band] + 5] = wt

            for count, row in enumerate(non_lens):
                count += n_cand #resume numbering after candidates numbers
                X,Y = row[1], row[2]

                if band in ['i','z']:
                    img = make_cutout(image[1], X, Y)
                else:
                    img = make_cutout(image[0], X, Y)

                wt  = make_cutout(weight[1], X, Y)

                #add to cutout collection
                cut_array[count, :,:,band2index[band]] = img 
                cut_array[count, :,:,band2index[band] + 5] = wt
                
            #clean up files related to this band
            image.close()
            weight.close()  

        #store all cutouts in all avaiable bands for this tile in hf only if there was more than one band and that band 
        if np.all((cut_array == 0)) == False:
            for count in range(len(cut_array)):
                hf.create_dataset(tile + '_'f"{count}", data=cut_array[count])      

            print(tile, 'done')

            #delete from slurm tmp dir
            for f in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, f))
        
    except:
        print('ERROR: SOMETHING WENT WRONG FOR TILE' + tile)
 
        
hf.close()
print('hf done')

#master catalogue
pd.DataFrame(np.array(all_non_lenses),columns = keys_mastercat).to_csv(scratch + 'mastercat_run2_4.csv')
