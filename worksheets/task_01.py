import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
from src.data_models import SDSS


########################################
# GLOBAL VARIABLES
########################################

# Here we define the tasks that we want to run
tasks = [5]

# Load SDSS as class
sdss = SDSS(pd.read_csv('../data/raw_data/sdss_cutout.csv'))


########################################
# TASK 1
########################################

if 1 in tasks:

    out_file_ecdf = '../data/results/ex_01/ecdf/ecdf.png'
    out_file_rband_redshift = '../data/results/ex_01/rband_redshift/rband_redshift.png'

    # Plot
    sdss.plot_ecdf(save_path=out_file_ecdf)
    sdss.plot_rband_redshift(save_path=out_file_rband_redshift)

    out_file_rband_redshift_filtered = '../data/results/ex_01/rband_redshift/rband_redshift_filtered.png'

    # Filter
    sdss.filter_params()

    # Plot
    sdss.plot_rband_redshift(xlim=(0, 0.2), save_path=out_file_rband_redshift_filtered)

else:

    sdss.filter_params()


########################################
# TASK 2
########################################

if 2 in tasks:

    out_file_colors = '../data/results/ex_01/colors/colors.png'

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
    out_file_maps = '../data/results/ex_01/maps/maps.png'

    sdss.plot_maps(save_path=out_file_maps)


########################################
# TASK 5
########################################

if 5 in tasks:

    # Define parameter for the two point correlation function

    iterations = 10_000
    sample_size = 400

    out_file_correlation = \
        f'../data/results/ex_01/two_point_correlation/results_{str(iterations)}_{str(sample_size)}.png'

    sdss.two_point_correlation(iterations=iterations, m_samples=sample_size, plot=True, save_path=out_file_correlation)

