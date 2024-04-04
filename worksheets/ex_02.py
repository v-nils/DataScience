import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
from src.data_models import SDSS


########################################
# GLOBAL VARIABLES
########################################

# Here we define the tasks that we want to run
tasks = [2]

# Load SDSS as class
sdss = SDSS(pd.read_csv('../data/raw_data/sdss_cutout.csv'))

# Set redshift slice
lower_redshift: float = 0.09
upper_redshift: float = 0.11

# Select redshift slice
sdss.select_redshift_slice(z_min=lower_redshift, z_max=upper_redshift)


########################################
# TASK 1
########################################

if 1 in tasks:
    save_path_angular_map: str = '../data/results/ex_02/angular_map/angular_map.png'
    sdss.plot_maps(save_path=save_path_angular_map, content='angular')

if 2 in tasks:
    save_path_histograms: str = '../data/results/ex_02/histograms/2d_histograms.png'

    _bins = np.linspace(100, 500, 15)
    bins = np.round(_bins).astype(int)
    sdss.plot_2d_histograms(bins, save_path=save_path_histograms)