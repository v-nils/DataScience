from typing import List, Any

import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import pywt
from numpy import ndarray, dtype, bool_, unsignedinteger, signedinteger, floating, complexfloating
from pywt._doc_utils import draw_2d_wp_basis, wavedec2_keys
import scienceplots
from src.math_functions import average_galaxy_bands
from src.util_functions import process_plot
from skimage.measure import block_reduce

data_path = '../../data/raw_data/DECals_galaxies.hdf5'


class Galaxy:

    def __init__(self, data: np.array, bands: None | list = None):
        """
        Initialize the Galaxy object with the image data and the bands.
        :param data: The image data.
        :param bands: Bands of the image data.
        """

        self.wavelet_data = None
        if bands is not None:
            self.bands = bands
        else:
            self.bands = ['g', 'r', 'z']

        coords = {'x': np.arange(data.shape[0]), 'y': np.arange(data.shape[1]), 'bands': self.bands}
        self.data = xr.DataArray(data, dims=['x', 'y', 'bands'], coords=coords)

    def average_bands(self, new_band: str | list = 'avg', **kwargs) -> None:
        """
        Average the bands of the galaxy data.
        new_band: str: The name of the new band.
        kwargs: dict: Additional arguments.
            overwrite: bool: If True, the data is initialized and overwritten by the new band. If False, the averaged band
            is appended to the data.
        :return: The averaged bands.
        """

        if isinstance(new_band, str):
            new_band = [new_band]
        overwrite = kwargs.get('overwrite', False)

        avg_band = self.data.mean(dim='bands')
        avg_band = avg_band.expand_dims({'bands': new_band})
        avg_band = avg_band.transpose('x', 'y', 'bands')
        if overwrite:
            self.__init__(avg_band, new_band)
            return

        self.data = xr.concat([self.data, avg_band], dim='bands')

    def compute_wavelets(self, save_path: str | None = None, **kwargs) -> list[Any]:
        """
        Compute the wavelet coefficients for the galaxy data.

        :param save_path: str: The path to save the plot.
        :param kwargs: dict: Additional arguments.
            level: int: The level of decomposition.
        """

        data = self.data[:, :, 0]
        coef = 2

        max_lev = kwargs.get('level', 2)

        fig, axes = plt.subplots(2, max_lev + 1, figsize=[14, 8])
        re_images = []

        for level in range(max_lev + 1):
            if level == 0:
                # show the original image before decomposition
                axes[0, 0].set_axis_off()
                axes[0, 0].imshow(data, cmap=plt.cm.gray)
                axes[0, 0].set_title('Image')
                axes[0, 0].set_axis_off()
                axes[1, 0].set_visible(False)
                continue

            coef = coef ** 2
            c = pywt.wavedec2(data, 'db2', mode='symmetric', level=level)

            # normalize each coefficient array independently for better visibility
            c[0] /= np.abs(c[0]).max()
            for detail_level in range(level):
                c[detail_level + 1] = [d / np.abs(d).max() for d in c[detail_level + 1]]

            cA = c[0]
            cH, cV, cD = c[1][:]
            re_image = pywt.waverec2(c, 'db2', mode='smooth')

            re_images.append(re_image)

            # show the normalized coefficients
            arr, slices = pywt.coeffs_to_array(c)
            axes[0, level].imshow(re_image, cmap=plt.cm.gray_r)
            axes[0, level].set_title(f'\n 1/{coef} of wavelet coefficients')
            axes[0, level].set_axis_off()

            axes[1, level].imshow(re_image, cmap=plt.cm.gray_r)
            axes[1, level].set_axis_off()

        plt.tight_layout()
        process_plot(plt, save_path=save_path)

        return re_images

    def singular_value_decomposition(self,
                                     ranks: list | None = None,
                                     **kwargs) -> list[Any]:
        """
        Compute the singular value decomposition of the galaxy data.

        :param ranks:
        :param kwargs: dict: Additional arguments.
            save_path: str: The path to save the plot.
        """

        plt.style.use(['science', 'ieee'])

        svds = []
        if ranks is None:
            ranks = [32, 64, 128, 256]

        save_path: str | None = kwargs.get('save_path', None)

        data = self.data[:, :, 0]

        U, S, V = np.linalg.svd(data, full_matrices=False)
        S: np.array = np.diag(S)

        hard_threshold = kwargs.get('hard_threshold', False)

        if hard_threshold:
            ht: float = 2.858 * np.median(S.diagonal())
            print(ht)

            S_ht = np.where(S > ht, S, 0)
            approx_ht = U @ S_ht @ V
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.imshow(approx_ht, cmap='gray_r')
            ax.set_title("Hard Thresholding", fontsize=21)
            ax.axis('off')

            if save_path is not None:
                ht_save_path = '..' + save_path.split('.')[-2] + '_ht.png'
                process_plot(plt, save_path=ht_save_path)

        if kwargs.get('plot_magnitude', False):
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            magnitudes = S.diagonal()
            ax.plot(np.linspace(0, len(magnitudes), len(magnitudes)), magnitudes)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r'$\sigma_r$', fontsize=12)
            ax.set_xlabel(r'RANK $r$', fontsize=12)

            if hard_threshold:
                fraction_of_svd_above_ht = np.sum(S.diagonal() > ht) / len(S.diagonal())
                ax.axhline(y=ht, linestyle='--', label=f'Hard Threshold at {ht:.2f}')
                ax.legend()
                ax.set_title(f'Percentage of SVD above HT: {fraction_of_svd_above_ht*100:.2f} percent', fontsize=21)

            if save_path is not None:
                magnitude_save_path = '..' + save_path.split('.')[-2] + '_magnitude.png'
                process_plot(plt, save_path=magnitude_save_path)

        fig, ax = plt.subplots(len(ranks), 2, figsize=(8, len(ranks)*4))

        for i, r in enumerate(ranks):
            approx = U[:, :r] @ S[:r, :r] @ V[:r, :]
            ax[i][0].imshow(approx, cmap='gray_r')
            ax[i][0].set_title("r = " + str(r), fontsize=21)
            ax[i, 0].axis('off')
            ax[i][1].set_title("Original Image", fontsize=21)
            ax[i][1].imshow(data, cmap='gray_r')
            ax[i, 1].axis('off')
            i += 1
            svds.append(approx)

        plt.tight_layout()

        process_plot(plt, save_path=save_path)

        return svds



    def downsample(self, p: int, bands: str | list | None = None, overwrite: bool = True, **kwargs):
        """
        Downsample the image data by averaging over p x p pixel blocks.

        :param overwrite:
        :param bands:
        :param p: The down-sampling factor.
        :return: The down-sampled image data.
        """

        if bands is None:
            bands = ['avg']
        elif isinstance(bands, str):
            bands = [bands]

        if overwrite:

            downsampled_data = block_reduce(self.data.sel(bands=bands).to_numpy(),
                                            block_size=(p, p, len(bands)),
                                            func=np.mean)

            self.__init__(downsampled_data, bands)

        else:

            downsampled_data = block_reduce(self.data.sel(bands=bands).to_numpy(),
                                            block_size=(p, p, len(bands)),
                                            func=np.mean)

            return Galaxy(downsampled_data, bands)

    def plot_compare_wavelet_svd(self, wavelet_data: list[Any], svd_data: list[Any], **kwargs) -> None:

        plot_cols = len(wavelet_data)

        figure, axis = plt.subplots(2, plot_cols, figsize=(plot_cols * 4, 10))

        for c_idx, (wavelet, svd) in enumerate(zip(wavelet_data, svd_data)):
            axis[0, c_idx].imshow(wavelet, cmap='gray_r')
            axis[0, c_idx].set_title('Wavelet')
            axis[0, c_idx].axis('off')
            axis[1, c_idx].imshow(svd, cmap='gray_r')
            axis[1, c_idx].set_title('SVD')
            axis[1, c_idx].axis('off')

        save_path = kwargs.get('save_path', None)
        process_plot(plt, save_path=save_path)


class DESI:
    """
    A class which holds the  data for one or more Galaxy objects from the DESI Model.
    """

    def __init__(self, file_path: str):
        self.data_path = file_path
        self.galaxies: list[Galaxy] = self._load_data()

    def sample_all_galaxies(self, p: int, **kwargs):
        """
        Downsample all the galaxies in the dataset by averaging over p x p pixel blocks.
        :param p:
        :return:
        """

        for galaxy in self.galaxies:
            galaxy.downsample(p, **kwargs)

    def average_all_galaxies(self, **kwargs):
        """
        Average the bands of all the galaxies in the dataset.
        :return:
        """
        print('###############################################')
        print('## Averaging all the galaxies in the dataset ##')
        print('###############################################')

        for galaxy in self.galaxies:
            galaxy.average_bands(**kwargs)

    def remove_by_index(self, index: int | list[int], inverse: bool = False) -> None:
        """
        Remove galaxies from the dataset by index.
        :param index: The index of the galaxies to remove.
        :param inverse: If True, the galaxies with the specified index are removed.
        :return:
        """
        if isinstance(index, int):
            index = [index]

        if inverse:
            bool_mask = [False if i not in index else True for i in range(len(self.galaxies))]
        else:
            bool_mask = [False if i in index else True for i in range(len(self.galaxies))]

        print(bool_mask)

        self.galaxies = self.galaxies[bool_mask]


    def _load_data(self, which='images_spirals'):

        with h5py.File(self.data_path) as F:
            images = np.array(F[which])
            galaxies = np.array([Galaxy(i) for i in images])
        return galaxies

    def create_downsampled_dataset(self,
                                   p_values: int | list[int],
                                   plot: bool = False,
                                   save_path: str | None = None) -> None:
        """
        Create a down-sampled dataset for the galaxies in the dataset.

        :param p_values: (int | list[int]) The down-sampling factors.
        :param plot: (bool) If True, the down-sampled images are plotted.
        :param save_path: (str) The path to save the plot.
        :return:
        """

        print('################################################')
        print('## Creating down-sampled dataset for galaxies ##')

        if isinstance(p_values, int):
            p_values = [p_values]

        down_samples = []

        for p in p_values:
            down_sample = [galaxy.downsample(p, overwrite=False) for galaxy in self.galaxies]
            down_samples.append(down_sample)

        if plot:
            fig, ax = plt.subplots(len(self.galaxies), len(p_values), figsize=(len(p_values) * 4, len(self.galaxies) * 4))

            for i, galaxy in enumerate(self.galaxies):
                for j, p in enumerate(p_values):
                    ax[i, j].imshow(down_samples[j][i].data.sel(bands='avg'), cmap='gray')
                    ax[i, j].axis('off')

        process_plot(plt, save_path=save_path)

    def plot_images(self, index: None | list, bands: list[list[str]] = None, save_path=None):
        """
        Plot the images of the galaxies in the dataset.
        :param index: Galaxy indices to plot.
        :param bands: Bands to plot.
        :param save_path:
        :return:
        """

        if index is None:
            index = range(len(self.galaxies))

        if bands is None:
            bands = [['g', 'r', 'z'], ['r'], ['g'], ['z'], ['avg']]

        num_images = len(index)
        cols = len(bands)
        fig, ax = plt.subplots(num_images, cols, figsize=(cols * 4, 5 * num_images))

        for i, idx in enumerate(index):
            selected_galaxy = self.galaxies[idx]

            for j, bnd in enumerate(bands):
                if len(bnd) != 1:
                    ax[i, j].imshow(selected_galaxy.data.sel(bands=bnd).astype(int))
                else:
                    ax[i, j].imshow(selected_galaxy.data.sel(bands=bnd), cmap='gray')

            ax[i, 0].axis('off')
            ax[i, 1].axis('off')
            ax[i, 2].axis('off')
            ax[i, 3].axis('off')
            ax[i, 4].axis('off')

        ax[0, 0].set_title('Combined Bands (RGB)')
        ax[0, 1].set_title('Red Band')
        ax[0, 2].set_title('Green Band')
        ax[0, 3].set_title('Blue Band')
        ax[0, 4].set_title('Average Bands')

        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        process_plot(plt, save_path=save_path)






if __name__ == '__main__':
    desi = DESI(data_path)
    desi.average_all_galaxies()

    desi.plot_images([0, 10, 20])
    current_gala = desi.galaxies[0]

    desi.sample_all_galaxies(2)
