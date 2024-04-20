import numpy as np
import pandas as pd
from src.data_models.sdss import SDSS, compute_percentiles

########################################
# GLOBAL VARIABLES
########################################

# Here we define the tasks that we want to run
tasks = ['3']

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

########################################
# TASK 2
########################################

if '2.1' in tasks:
    # Define save paths
    save_path_histograms: str = '../data/results/ex_02/histograms/2d_histograms.png'

    _bins = np.linspace(100, 250, 6)
    bins = np.round(_bins).astype(int)
    sdss.plot_2d_histograms(bins, save_path=save_path_histograms)

if '2.2' in tasks:
    save_path_kde: str = '../data/results/ex_02/kde/kde.png'
    save_path_kde_std: str = '../data/results/ex_02/kde/kde_std.png'

    kernel_bins: np.ndarray = np.array([0.01, 0.05, 0.1])
    kernel_bins_standardized: np.ndarray = np.array([0.001, 0.005, 0.01])

    sdss.compute_kde(kernel_bins, use_standardizes_vals=False, method='both', save_path=save_path_kde, plot_one_point_stats=False)
    sdss.compute_kde(kernel_bins_standardized, use_standardizes_vals=True, method='both', save_path=save_path_kde_std, plot_one_point_stats=False)

if '2.3' in tasks:
    save_path_nn: str = '../data/results/ex_02/nn/nn.png'
    save_path_nn_std: str = '../data/results/ex_02/nn/nn_std.png'

    n_neighbors = np.array([5, 10, 20])

    sdss.nearest_neighbor_estimation(n_neighbors, use_standardizes_vals=True, save_path=save_path_nn_std)

if '2.4' in tasks:
    save_path_voronoi_f: str = '../data/results/ex_02/voronoi/voronoi_filled.png'
    save_path_voronoi: str = '../data/results/ex_02/voronoi/voronoi.png'

    sdss.voroni_volumes(plot_colors=True, save_path=save_path_voronoi_f)
    #sdss.voroni_volumes(plot_colors=False, save_path=save_path_voronoi)

if '2.5' in tasks:
    save_path_delaunay: str = '../data/results/ex_02/delaunay/delaunay.png'
    sdss.delaunay_triangulation(save_path=save_path_delaunay)


########################################
# TASK 3
########################################


if '3' in tasks:
    save_path_kde_std: str = '../data/results/ex_02/kde_one_point_stats/kde_vol_with_stats_std.png'
    kernel_bins_standardized: np.ndarray = np.array([0.001, 0.005, 0.01])

    sdss.compute_kde(kernel_bins_standardized,
                     n_bins=50,
                     use_standardizes_vals=True,
                     save_path=save_path_kde_std,
                     method='both',
                     plot_one_point_stats=True)
## Plot mit achsenbeschriftung

########################################
# TASK 4
########################################


if '4' in tasks:
    save_path_ks = '../data/results/ex_02/ks/ks.png'

    sdss.kolmogorov_smirnoff(n_bins=22, bandwidths=0.005,  n_slices=4, method='mass', use_standardized_vals=True,
                             save_path=save_path_ks, kde_per_subplot=3)


########################################
# TASK 5
########################################


if '5' in tasks:
    save_path_ks_quantiles = '../data/results/ex_02/ks/ks_quantiles.png'

    results = sdss.kolmogorov_smirnoff(n_bins=50, bandwidths=0.75,  n_slices=1, method='mass',
                                       plot=False, use_standardized_vals=True)
    compute_percentiles(*results, save_path=save_path_ks_quantiles)


## Plots Algemein