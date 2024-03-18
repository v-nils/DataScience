import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
from src.data_models import SDSS


########################################
# TASK 1
########################################

# Load SDSS as class
sdss = SDSS(pd.read_csv('../data/raw_data/sdss_cutout.csv'))


# Plot
sdss.plot_ecdf()
sdss.plot_rband_redshift()

# Filter
sdss.filter_params()

# Plot
sdss.plot_rband_redshift(xlim=(0, 0.2))


########################################
# TASK 2
########################################


sdss.plot_colors()


########################################
# TASK 3
########################################

mean_red = sdss.mean_red
mean_blue = sdss.mean_blue
std_red = sdss.std_red
std_blue = sdss.std_blue

print(mean_red, mean_blue, std_red, std_blue)


########################################
# TASK 4
########################################

sdss.plot_maps()


########################################
# TASK 5
########################################

