import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
from src.data_models import SDSS


########################################
# GLOBAL VARIABLES
########################################

tasks = [4]

# Load SDSS as class
sdss = SDSS(pd.read_csv('../data/raw_data/sdss_cutout.csv'))


########################################
# TASK 1
########################################

if 1 in tasks:

    out_file_ecdf = '../data/results/ex_01/ecdf/ecdf.pdf'
    out_file_rband_redshift = '../data/results/ex_01/rband_redshift/rband_redshift.png'

    # Plot
    sdss.plot_ecdf(save_path=out_file_ecdf)
    sdss.plot_rband_redshift(save_path=out_file_rband_redshift)

    # Filter
    sdss.filter_params()

    # Plot
    sdss.plot_rband_redshift(xlim=(0, 0.2))

else:

    sdss.filter_params()


########################################
# TASK 2
########################################

if 2 in tasks:

    out_file_colors = '../data/results/ex_01/colors/colors.pdf'

    # Plot
    sdss.plot_colors(save_path=out_file_colors)


########################################
# TASK 3
########################################

if 3 in tasks:
    mean_red = sdss.mean_red
    mean_blue = sdss.mean_blue
    std_red = sdss.std_red
    std_blue = sdss.std_blue

    print(mean_red, mean_blue, std_red, std_blue)


########################################
# TASK 4
########################################

if 4 in tasks:
    out_file_maps = '../data/results/ex_01/maps/maps.pdf'

    sdss.plot_maps(save_path=out_file_maps)


########################################
# TASK 5
########################################

if 5 in tasks:
    # Initialize the SDSS class with data from the CSV file
    sdss = SDSS(pd.read_csv('../data/raw_data/sdss_cutout.csv'))

    # Define parameter for the two point correlation function

    iterations = 1000
    sample_size = 250

    # Filter the data based on the parameters defined in the filter_params method
    sdss.filter_params()
    sdss.two_point_correlation(iterations=iterations, m_samples=sample_size, plot=True)

