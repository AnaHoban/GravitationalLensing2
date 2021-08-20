import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel

image_dir = "/home/anahoban/projects/rrg-kyi/astro/cfis/W3/"
lense_dir = "/home/anahoban/projects/rrg-kyi/astro/lenses/"
label_dir = "/home/anahoban/scratch/lenses/"
label_subdirs = ["stronglensdb_confirmed_unige/", "stronglensdb_candidates_unige/", "canameras2020/", 
                 "huang2020a_grade_A/", "huang2020a_grade_B/", "huang2020a_grade_C/", 
                 "huang2020b_grade_A/", "huang2020b_grade_B/", "huang2020b_grade_C/"]

tile_file = open(image_dir + "tiles_cand.list", "r")
tile_list = tile_file.readlines()
for i in range(len(tile_list)):
    tile_list[i] = tile_list[i][:-1] # Remove new line characters
tile_file.close()

for label_subdir in label_subdirs:
    if label_subdir.startswith("huang"):
        ra_str = "RA"
        dec_str = "Dec"
    else:
        ra_str = "ra"
        dec_str = "dec"
    for tile in tile_list:
        tile_id = tile.split(".")[1] + "." + tile.split(".")[2] # XXX.XXX id
        # Determine subdirectory to output csv file
        if tile[-1] == "u":
            subdir = label_dir + label_subdir + "CFIS u/"
        elif tile[-1] == "g":
            subdir = label_dir + label_subdir + "PS1 g/"
        elif tile[-1] == "r" and tile.startswith("CFIS"):
            subdir = label_dir + label_subdir + "CFIS r/"
        elif tile[-1] == "r" and tile.startswith("PS1"):
            subdir = label_dir + label_subdir + "PS1 r/"
        elif tile[-1] == "i":
            subdir = label_dir + label_subdir + "PS1 i/"
        else:
            subdir = label_dir + label_subdir + "PS1 z/"

        # Convert (ra, dec) -> (x,y) using header from fits file
        fits_file = fits.open(image_dir + tile + ".fits")
        head = fits_file[0].header
        w = WCS(head)
        df = pd.read_csv(lense_dir + label_subdir[:-1] + '.csv')
        x,y = skycoord_to_pixel(SkyCoord(df[ra_str], df[dec_str], unit="deg"), w)
        df['x'] = x
        df['y'] = y
        nx = head['NAXIS1']
        ny = head['NAXIS2']
        tile_df = df[(df['x']>0) & (df['y']>0) & (df['x']<nx) & (df['y']<ny)]
        if len(tile_df) > 0: # Labels were found in this fits file
            tile_df.to_csv(subdir + tile_id + '_labels.csv')
        fits_file.close()
    print(f"Finished {label_subdir}")
