import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numba import jit


class SDSS:

    def __init__(self, data: pd.DataFrame) -> None:

        # SDSS as Pandas Dataframe
        self.data = data
        self.ra = data.iloc[:, 0]
        self.de = data.iloc[:, 1]
        self.z_redshift = data.iloc[:, 2]
        self.u = data.iloc[:, 3]
        self.g = data.iloc[:, 4]
        self.r = data.iloc[:, 5]
        self.i = data.iloc[:, 6]
        self.z_magnitude = data.iloc[:, 7]
        self.indices = data[(self.z_redshift < 0.12) & (self.z_redshift > 0.08)]
        self.color = self.u - self.r
        self.blue = self.color[self.color <= 2.3]
        self.red = self.color[self.color > 2.3]

        # Statistic parameters
        self.mean_red = np.mean(self.r[self.red.index])
        self.mean_blue = np.mean(self.r[self.blue.index])
        self.std_red = np.std(self.r[self.red.index])
        self.std_blue = np.std(self.r[self.blue.index])

        # Coordinates
        self.phi = np.pi * self.ra / 180
        self.theta = np.pi / 2 - np.pi * self.de / 180




    def filter_params(self):
        filtered_data = self.indices
        return filtered_data


    def plot_ecdf(self, **kwargs) -> None:
        s_o = np.sort(self.z_redshift)
        fig = plt.figure()
        plt.step(s_o, np.arange(len(s_o)) / len(s_o), label='empirical CDF')
        plt.xlabel('Redshift')
        plt.ylabel('eCDF')
        plt.legend(loc='best')
        plt.show()

    def plot_rband_redshift(self, xlim=(0, 0.6), **kwargs):
        plt.scatter(self.z_redshift, self.r, s=0.5, alpha=0.1)
        plt.xlabel('Redshift')
        plt.xlim(*xlim)
        plt.ylabel('r-band magnitude')
        plt.show()

    def plot_colors(self, **kwargs):
        plt.scatter(self.r[self.blue.index], self.blue, s=0.5, alpha=0.1, color='blue', label='Blue galaxies')
        plt.scatter(self.r[self.red.index], self.red, s=0.5, alpha=0.1, color='red', label='Red galaxies')
        plt.legend(loc='best')
        plt.xlabel('r-band mag')
        plt.ylabel('u-r')
        plt.show()

    def plot_maps(self, **kwargs):
        plt.subplot(2, 2, 1)
        plt.title('Angular Map for blue galaxies', fontsize=20)
        plt.xlabel('Rektaszenion', fontsize=15)
        plt.ylabel('Declination', fontsize=15)
        plt.scatter(self.ra[self.blue.index], self.de[self.blue.index], s=0.5, alpha=0.1)

        plt.subplot(2, 2, 2)
        plt.title('Angular Map for red galaxies', fontsize=20)
        plt.xlabel('Rektaszension', fontsize=15)
        plt.ylabel('Declination', fontsize=15)
        plt.scatter(self.ra[self.red.index], self.de[self.red.index], s=0.5, alpha=0.1)

        plt.subplot(2, 2, 3)
        plt.title('Redshift-space map for blue galaxies', fontsize=20)
        plt.xlabel('RA', fontsize=15)
        plt.ylabel('Redshift', fontsize=15)
        plt.scatter(self.ra[self.blue.index], self.z_redshift[self.blue.index], s=0.5, alpha=0.1)

        plt.subplot(2, 2, 4)
        plt.title('Redshift-space map for red galaxies', fontsize=20)
        plt.xlabel('RA', fontsize=15)
        plt.ylabel('Redshift', fontsize=15)
        plt.scatter(self.ra[self.red.index], self.z_redshift[self.red.index], s=0.5, alpha=0.1)

        plt.show()

    def realisation(M):
        random_ra = np.random.uniform(130, 230, size=M)
        random_de = np.random.uniform(5, 65, size=M)
        return np.column_stack((random_ra, random_de))







if __name__ == "__main__":
    sdss = SDSS(pd.read_csv('../data/raw_data/sdss_cutout.csv'))

    print(len(sdss.data))


