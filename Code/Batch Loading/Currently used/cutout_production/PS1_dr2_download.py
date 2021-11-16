#some tiles are missing: the DR2 ps1 tiles!

import os
import sys
import pandas as pd
import numpy as np

label_dir = "/home/anahoban/projects/def-sfabbro/anahoban/lensing/GravitationalLensing/Code/Batch Loading/Currently used/labels/"
lens_dir_scratch = os.path.expandvars("$SCRATCH") + '/all_lenses'

lenses_in_survey = pd.read_csv('lenses_in_survey.csv', dtype={'Tile':str} )
tiles_table = pd.read_csv(label_dir + 'tiles_summary.csv', dtype={'Tile':str})
tiles_with_lenses = list(lenses_in_survey['Tile'].unique())

ps1_tiles = []
for tile in tiles_with_lenses:
    if tiles_table[tiles_table.Tile == tile].PS1_i.item() == 1.0:
        print(tile)
        os.system('vcp -v vos:cfis/panstarrs/DR2/skycell.' + tile[:3] + '/CFIS.V0.skycell.' + tile + '.stk.???????.unconv' + '.fits ' + lens_dir_scratch )
        os.system('vcp -v vos:cfis/panstarrs/DR2/skycell.' + tile[:3] + '/CFIS.V0.skycell.' + tile + '.stk.???????.unconv' + '.wt.fits ' + lens_dir_scratch )