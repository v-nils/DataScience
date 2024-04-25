import copy
import matplotlib.pyplot as plt
from src.data_models.desi import DESI

# Load the data
data_path = '../data/raw_data/DECals_galaxies.hdf5'
desi_main = DESI(data_path)

task = [4]

wavelets = None
svds = None
###############################################################################
# Exercise 1
# Create a new DESI object with the data_path as the data_path attribute.


###############################################################################

if 1 in task:

    desi_copy = copy.deepcopy(desi_main)

    galaxy_list = [0, 17, 23, 25, 86, 99, 105, 200]


    desi_copy.remove_by_index(galaxy_list, inverse=True)
    # Galaxies of interest
    p_vals = [1, 2, 4, 8, 16, 32]
    # Compute the average of all the galaxies
    desi_copy.average_all_galaxies(overwrite=False)

    desi_copy.plot_images(save_path='../data/results/ex_03/desi_rgb_avg/desi_plot.png')


    desi_copy.create_downsampled_dataset(p_vals,
                                         plot=True,
                                         save_path='../data/results/ex_03/desi_rgb_avg/desi_downsampled.png')


###############################################################################
# Exercise 2

if 2 in task:
    galaxy_idx = 233

    desi_copy = copy.deepcopy(desi_main)

    galaxy = desi_copy.galaxies[galaxy_idx]

    galaxy.average_bands(overwrite=True)
    wavelets = galaxy.compute_wavelets(level=2,
                                       save_path='../data/results/ex_03/galaxy_wavelets/desi_wavelets.png')

if 3 in task:
    galaxy_idx = 233

    desi_copy = copy.deepcopy(desi_main)

    galaxy = desi_copy.galaxies[galaxy_idx]

    galaxy.average_bands(overwrite=True)

    svds = galaxy.singular_value_decomposition(ranks=[128, 64, 32, 16, 8, 4, 2],
                                               save_path='../data/results/ex_03/galaxy_svd/galaxy_svd_2.png',
                                               plot_magnitude=True,
                                               hard_threshold=True)

    if wavelets is None:
        wavelets = galaxy.compute_wavelets(level=2, save_path='../data/results/ex_03/galaxy_wavelets/galaxy_wavelets.png')
        pass

    galaxy.plot_compare_wavelet_svd(wavelets, svds, save_path='../data/results/ex_03/galaxy_wavelets_svd/desi_wavelets_svd.png')

if 4 in task:
    all_galaxies = copy.deepcopy(desi_main)

    all_galaxies.average_all_galaxies(overwrite=True)

    all_galaxies.sample_all_galaxies(4)

