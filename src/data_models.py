import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.correlation_functions import compute_angles, landy_szalay
import seaborn as sns
from numba import jit


def _plot_correlation_fun(omega, correlation_red, correlation_blue):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Find the minimum and maximum values across both arrays
    min_val = min(correlation_red.min(), correlation_blue.min())
    max_val = max(correlation_red.max(), correlation_blue.max())

    ax[0].plot(omega[:-1], correlation_red, label='Red galaxies')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\omega$')
    ax[0].set_ylabel(r'$\xi(\omega)$')
    ax[0].legend()
    ax[0].set_title('Red galaxies')
    ax[0].set_ylim([min_val, max_val])  # Set y-axis limits

    ax[1].plot(omega[:-1], correlation_blue, label='Blue galaxies')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$\omega$')
    ax[1].set_ylabel(r'$\xi(\omega)$')
    ax[1].legend()
    ax[1].set_title('Blue galaxies')
    ax[1].set_ylim([min_val, max_val])  # Set y-axis limits

    plt.show()


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

        self.blue_idx = self.color <= 2.3
        self.red_idx = self.color > 2.3
        self.num_red = len(self.red_idx[self.red_idx == True])
        self.num_blue = len(self.blue_idx[self.blue_idx == True])

        # Statistic parameters
        self.mean_red = np.mean(self.r[self.red_idx])
        self.mean_blue = np.mean(self.r[self.blue_idx])
        self.std_red = np.std(self.r[self.red_idx])
        self.std_blue = np.std(self.r[self.blue_idx])

        # Coordinates
        self.phi = np.pi * self.ra / 180
        self.theta = np.pi / 2 - np.pi * self.de / 180
        self.random_phi = None
        self.random_theta = None

        # Results of the two-point correlation function
        self.correlation_results_red: dict | None = None
        self.correlation_results_blue: dict | None = None

    def sample_down(self, n: int) -> None:
        sampled_data_red = self.data[self.red_idx].sample(int(n / 2), random_state=42)
        sampled_data_blue = self.data[self.blue_idx].sample(int(n / 2), random_state=42)

        sampled_data = pd.concat([sampled_data_red, sampled_data_blue])
        self.__init__(sampled_data)

    def filter_params(self):
        self.__init__(self.indices)

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

    def _generate_random_positions(self, m: int | None = None) -> None:

        if m is None:
            m = self.num_red if self.num_red < self.num_blue else self.num_blue

        self.random_phi = np.random.uniform(130, 230, size=m) / 180 * np.pi
        self.random_theta = np.pi / 2 - np.random.uniform(5, 65, size=m) / 180 * np.pi

    def _two_point_correlation(self, plot: bool = False, **kwargs) -> np.array:

        m = kwargs.get('m', self.num_red if self.num_red < self.num_blue else self.num_blue)

        self._generate_random_positions(m=m)


        indices_blue = np.random.choice(np.arange(len(self.theta[self.blue_idx])), m)
        subsample_theta_blue = self.theta[self.blue_idx].to_numpy()[indices_blue]
        subsample_phi_blue = self.phi[self.blue_idx].to_numpy()[indices_blue]

        indices_red = np.random.choice(np.arange(len(self.theta[self.red_idx])), m)
        subsample_theta_red = self.theta[self.red_idx].to_numpy()[indices_red]
        subsample_phi_red = self.phi[self.red_idx].to_numpy()[indices_red]

        random_positions_red = compute_angles(self.random_theta, self.random_phi)
        random_positions_blue = compute_angles(self.random_theta, self.random_phi)

        data_positions_red = compute_angles(subsample_theta_red, subsample_phi_red)
        data_positions_blue = compute_angles(subsample_theta_blue, subsample_phi_blue)

        data_random_positions_red = compute_angles(subsample_theta_red, subsample_phi_red,
                                                  self.random_theta,
                                                  self.random_phi)

        data_random_positions_blue = compute_angles(subsample_theta_blue, subsample_phi_blue,
                                                   self.random_theta,
                                                   self.random_phi)

        # Define the bins for the histogram
        omega = np.geomspace(0.003, 0.3, 11)

        # Compute the histogram for the red galaxies using the actual positions
        dd_counts_red, _ = np.histogram(data_positions_red, bins=omega)

        # Compute the histogram for the blue galaxies using the actual positions
        dd_counts_blue, _ = np.histogram(data_positions_blue, bins=omega)

        # Compute the histogram for the red galaxies using the random positions
        rr_counts_red, _ = np.histogram(random_positions_red, bins=omega)

        # Compute the histogram for the blue galaxies using the random positions
        rr_counts_blue, _ = np.histogram(random_positions_blue, bins=omega)

        # Compute the histogram for the red galaxies using a mix of actual and random positions
        dr_counts_red, _ = np.histogram(data_random_positions_red, bins=omega)

        # Compute the histogram for the blue galaxies using a mix of actual and random positions
        dr_counts_blue, _ = np.histogram(data_random_positions_blue, bins=omega)

        # Compute the Landy-Szalay estimator for the red galaxies
        correlation_red = landy_szalay(dd_counts_red, dr_counts_red, rr_counts_red, m)

        # Compute the Landy-Szalay estimator for the blue galaxies
        correlation_blue = landy_szalay(dd_counts_blue, dr_counts_blue, rr_counts_blue, m)

        if plot:
            _plot_correlation_fun(omega, correlation_red, correlation_blue)

        return correlation_blue, correlation_red

    def two_point_correlation(self, iterations: int = 100, m_samples: int = 100, **kwargs) -> None:

        print(f"Computing two-point correlation function for {iterations} iterations with {m_samples} samples")

        results_red = []
        results_blue = []

        for i in range(iterations):
            print(f"iteration {i+1}/{iterations}")
            c_blue, c_red = self._two_point_correlation(m=m_samples, plot=False)
            results_blue.append(c_blue)
            results_red.append(c_red)

        results_blue = np.array(results_blue)
        results_red = np.array(results_red)

        mean_blue = np.mean(results_blue, axis=0)
        mean_red = np.mean(results_red, axis=0)

        std_blue = np.std(results_blue, axis=0)
        std_red = np.std(results_red, axis=0)

        var_blue = np.var(results_blue, axis=0)
        var_red = np.var(results_red, axis=0)

        self.correlation_results_blue = {'mean': mean_blue, 'std': std_blue, 'var': var_blue}
        self.correlation_results_red = {'mean': mean_red, 'std': std_red, 'var': var_red}

        if kwargs.get('plot', True):
            self.plot_correlation()


    def plot_correlation(self):
        if self.correlation_results_red is not None and self.correlation_results_blue is not None:
            omega = np.geomspace(0.003, 0.3, 11)
            _plot_correlation_fun(omega, self.correlation_results_red['mean'], self.correlation_results_blue['mean'])
        else:
            raise ValueError("Two-point correlation function has not been computed yet")


if __name__ == "__main__":
    # Set the sample size
    #sample_size = 100_000

    # Initialize the SDSS class with data from the CSV file
    sdss = SDSS(pd.read_csv('../data/raw_data/sdss_cutout.csv'))

    # Filter the data based on the parameters defined in the filter_params method
    sdss.filter_params()

    # Downsample the data to the specified sample size
    #sdss.sample_down(sample_size)

    sdss.two_point_correlation(1000, 250)

    sdss.plot_correlation()










