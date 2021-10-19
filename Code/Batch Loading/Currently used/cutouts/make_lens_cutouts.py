import os
import sys
import h5py
import shutil
import csv
import pandas as pd
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from astropy import table,coordinates
from astropy import units as u
from astropy.wcs.utils import skycoord_to_pixel
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
from astropy.visualization import make_lupton_rgb
from astroquery.vizier import Vizier
from collections import Counter
from astropy.wcs import WCS,utils
import matplotlib.pyplot as plt
from astropy.io import ascii
import fitsio
from esutil import wcsutil


#directories
label_dir = "/home/anahoban/projects/def-sfabbro/anahoban/lensing/GravitationalLensing/Code/Batch Loading/Currently used/labels/"
lenses_list_dir = "/home/anahoban/projects/def-sfabbro/anahoban/lensing/GravitationalLensing/Code/Batch Loading/Currently used/cutouts/"
lens_dir_scratch = os.path.expandvars("$SCRATCH") + '/all_lenses/'
cuts_dir_scratch = os.path.expandvars("$SCRATCH") + '/all_lenses_cutouts/'
tmp_dir = os.path.expandvars("$SLURM_TMPDIR") + '/'

lenses_in_survey = pd.read_csv(lenses_list_dir+'lenses_in_survey.csv', dtype={'Tile':str} )
tiles_table = pd.read_csv(label_dir + 'tiles_summary.csv', dtype={'Tile':str})
tiles_with_lenses = list(lenses_in_survey['Tile'].unique())

#make the urgz list for each tile
keys = ['CFIS_u', 'CFIS_r', 'PS1_g', 'PS1_i', 'PS1_z','HCS_g']
key_dict = {'CFIS_u': 'u', 'CFIS_r': 'r', 'PS1_g' : 'G', 'PS1_i': 'i', 'PS1_z': 'z','HCS_g': 'g'}
band2cam = {v: k[:-2] for k, v in key_dict.items()}

band_to_mask = { 'u' : '.weight.fits.fz ',
                 'r' : '.weight.fits.fz ',
                 'i' : '.wt.fits ',
                 'z' : '.wt.fits ',
                 'G' : '.wt.fits ',
                 'g' : '.weight.fits.fz '}

bands_list = []
for index, row in tiles_table.iterrows():
    bands = ''
    for key in keys:
        if row[key] == 1:
            bands += key_dict[key]
    bands_list.append(bands)
    
tiles_table['bands_list'] = bands_list


def make_cutout(ra, dec, filename,  cutout_size = 200):
    #copy to tmpdir
    shutil.copy2(lens_dir_scratch + filename, tmp_dir)
    print('copied to slurm')
    
    img_slurm = os.path.abspath(tmp_dir + filename)
    img_fits = fits.open(img_slurm, memmap=True)[0]
    print('opened file')
    wcs = WCS(img_fits.header)
    
    
    #get X,Y
    fits = fitsio.FITS(lens_dir_scratch + filename)
    head = fits[0].read_header()
    w = wcsutil.WCS(head)
    x,y = w.sky2image(ra,dec)

    nx = head['NAXIS1']
    ny = head['NAXIS2']
    fits.close()
    
    # Make the cutout, including the WCS
    cutout = Cutout2D(img_fits.data, position=(x,y), size=cutout_size, wcs=wcs)

    # Put the cutout image in the FITS HDU
    img_fits.data = cutout.data

    # Update the FITS header with the cutout WCS
    img_fits.header.update(cutout.wcs.to_header())
    
    os.remove(img_slurm)
    
    return img_fits


for tile in tiles_with_lenses[:10]:
    #candidates coordindates for a tile
    print(tile)
    ra_dec_list = [(row.RA, row.DEC) for index, row in lenses_in_survey[lenses_in_survey.Tile == tile][['RA','DEC']].iterrows()]
    
    #available bands    
    bands = tiles_table[tiles_table.Tile == tile]['bands_list']
        
    #make cutouts
    for band in bands:
        print(band)
        cam = band2cam[band]
        for ra_dec in ra_dec_list:
            print(ra_dec)
            ra,dec = ra_dec
            img_name = cam + '.' + tile + '.' +  band + '.fits'
            wt_name =  cam + '.' + tile + '.' +  band + band_to_mask[band]     
            
            #make cutout for given tile+band combination
            
            
            cutout_filename = cam + '.' + tile + '_' + band + str(ra) + '_' + str(dec) + '.fits'
            #cutout_filename = cam + '.' + tile + '_' + band + str(ra) + '_' + str(dec) + '.fits'
            
            cutout_fits = make_cutout(ra, dec, img_name)
            #cutout_fits = make_cutout(ra, dec, wt_name)
            
            cutout_fits.writeto(cuts_dir_scratch + cutout_filename, overwrite=True)
            



