import glob
import pandas as pd

lense_dir = "/home/eyvorch9/projects/rrg-kyi/astro/lenses/"
ra_list = []
dec_list = []
for csv in glob.glob(lense_dir + "*.csv"):
    df = pd.read_csv(csv)
    if "huang" in csv:
        ra_str = "RA"
        dec_str = "Dec"
    else:
        ra_str = "ra"
        dec_str = "dec"
    (ra, dec) = (df[ra_str], df[dec_str])
    ra_list.extend(ra)
    dec_list.extend(dec)
ra_dec_dict = {"ra": ra_list, "dec": dec_list}
ra_dec_df = pd.DataFrame(ra_dec_dict)
ra_dec_df.to_csv("labels/ra_dec.csv")
