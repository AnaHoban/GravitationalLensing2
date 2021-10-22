import os
import sys
import pandas as pd
import numpy as np

#directories
label_dir = "/home/anahoban/projects/def-sfabbro/anahoban/lensing/GravitationalLensing/Code/Batch Loading/Currently used/labels/"
lens_dir_scratch = os.path.expandvars("$SCRATCH") + '/all_lenses'
tmp_dir = os.path.expandvars("$SLURM_TMPDIR") + '/'

prev_dowloaded_files = os.listdir(lens_dir_scratch)

#input from command line in the form: XXX.YYY urgiz
#tile = sys.argv[1] 
#bands = sys.argv[2] #one or many bands

lenses_in_survey = pd.read_csv('lenses_in_survey.csv',dtype={'Tile':str} )
tiles_table = pd.read_csv(label_dir + 'tiles_summary.csv', dtype={'Tile':str})

tiles_with_lenses = list(lenses_in_survey['Tile'].unique())

band_to_dir = { 'u' : 'vcp -v vos:cfis/tiles_DR3/CFIS.',
                'r' : 'vcp -v vos:cfis/tiles_DR3/CFIS.',
                'i' : 'vcp -L -v vos:cfis/ps_tiles/PS1.',
                'z' : 'vcp -L -v vos:cfis/ps_tiles/PS1.',
                'G' : 'vcp -L -v vos:cfis/ps_tiles/PS1.',
                'g' : 'vcp -v vos:cfis/hsc/stacks2/HSC.'}


band_to_mask = { 'u' : '.weight.fits.fz ',
                 'r' : '.weight.fits.fz ',
                 'i' : '.wt.fits ',
                 'z' : '.wt.fits ',
                 'G' : '.wt.fits ',
                 'g' : '.weight.fits.fz '}


#make the urgz list for each tile
keys = ['CFIS_u', 'CFIS_r', 'PS1_g', 'PS1_i', 'PS1_z','HCS_g']
key_dict = {'CFIS_u': 'u', 'CFIS_r': 'r', 'PS1_g' : 'G', 'PS1_i': 'i', 'PS1_z': 'z','HCS_g': 'g'}
band2cam = {v: k[:-2] for k, v in key_dict.items()}

bands_list = []
for index, row in tiles_table.iterrows():
    bands = []
    for key in keys:
        if row[key] == 1:
            bands.append(key_dict[key])
    bands_list.append(bands)
    
tiles_table['bands_list'] = bands_list


for tile in tiles_with_lenses:
    #download tiles + weight map for each tile containing at least one candidate
    bands = tiles_table[tiles_table.Tile == tile]['bands_list'].item()
    print(bands)
    for band in bands:
        try:
            if (band2cam[band] + '.' + tile + '.' +  band + '.fits') not in prev_dowloaded_files:
                os.system(band_to_dir[band] + tile + '.' +  band + '.fits {}'.format(lens_dir_scratch) )
            
            if (band2cam[band] + '.' + tile + '.' +  band + band_to_mask[band][:-1]) not in prev_dowloaded_files:
                os.system(band_to_dir[band] + tile + '.' +  band + band_to_mask[band] + str(lens_dir_scratch) )
        except:
            continue
        if band in ['u','r','g']:
            if (band2cam[band] + '.' + tile + '.' +  band + '.cat')  not in prev_dowloaded_files:
                os.system(band_to_dir[band] + tile + '.' +  band + '.cat {}'.format(lens_dir_scratch) )
                


