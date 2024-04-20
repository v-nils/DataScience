import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import pywt

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

    def compute_wavelets(self, wavelet: str = 'haar', **kwargs):
        """
        Compute the wavelet transform of the galaxy data.
        :param wavelet: The wavelet transform method.
        :param kwargs: Additional arguments.
        :return:
        """
        cA, (cH, cV, cD) = pywt.dwt2(self.data, wavelet=wavelet)
        re_image = pywt.idwt2((cA, (cH, cV, cD)), wavelet=wavelet)

        wp = pywt.WaveletPacket2D(data=self.data.sel(bands=1).to_numpy(), wavelet='db1', mode='symmetric')
        print(wp.get_level(1))
        print(type(re_image))
        self.wavelet_data = re_image

    def plot_wavelet_data(self, **kwargs):
        """
        Plot the wavelet transformed data.
        """
        if hasattr(self, 'wavelet_data'):
            plt.imshow(self.wavelet_data[:, :, 0], cmap='gray')
            plt.show()
        else:
            print("Wavelet data is not computed. Please run compute_wavelets method first.")

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
