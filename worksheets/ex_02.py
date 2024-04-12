import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
from src.data_models import SDSS


########################################
# GLOBAL VARIABLES
########################################

# Here we define the tasks that we want to run
tasks = ['1', '2.1', '2.2', '2.3', '2.5', '2.4']

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

if '1' in tasks:
    save_path_angular_map: str = '../data/results/ex_02/angular_map/angular_map.png'
    sdss.plot_maps(save_path=save_path_angular_map, content='angular')

if '2.1' in tasks:

    # Define save paths
    save_path_histograms: str = '../data/results/ex_02/histograms/2d_histograms.png'

    _bins = np.linspace(100, 500, 15)
    bins = np.round(_bins).astype(int)
    sdss.plot_2d_histograms(bins, save_path=save_path_histograms)

if '2.2' in tasks:
    save_path_kde: str = '../data/results/ex_02/kde/kde.png'
    save_path_kde_std: str = '../data/results/ex_02/kde/kde_std.png'

    kernel_bins = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    kernel_bins_standardized = np.array([0.001, 0.005, 0.01, 0.025, 0.05, 0.1])

    sdss.kernel_density_estimation(kernel_bins,
                                   use_standardizes_vals=False,
                                   save_path=save_path_kde)  # Not standardized
    sdss.kernel_density_estimation(kernel_bins_standardized,
                                   use_standardizes_vals=True,
                                   save_path=save_path_kde_std)  # Standardized

if '2.3' in tasks:
    save_path_nn: str = '../data/results/ex_02/nn/nn.png'
    save_path_nn_std: str = '../data/results/ex_02/nn/nn_std.png'

    n_neighbors = np.array([5, 10, 15, 20, 25, 30])
    #n_neighbors_standardized = np.array([1, 2])

    sdss.nearest_neighbor_estimation(n_neighbors,
                                     use_standardizes_vals=False,
                                     save_path=save_path_nn)  # Not standardized
    #sdss.nearest_neighbor_estimation(n_neighbors_standardized, use_standardizes_vals=True, save_path=save_path_nn_std)  # Standardized

if '2.4' in tasks:

    save_path_voronoi: str = '../data/results/ex_02/voronoi/voronoi.png'

    sdss.voroni_volumes(save_path=save_path_voronoi)

if '2.5' in tasks:

    save_path_delaunay: str = '../data/results/ex_02/delaunay/delaunay.png'
    sdss.delaunay_triangulation(save_path=save_path_delaunay)
