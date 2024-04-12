import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.math_functions import compute_angles, landy_szalay
import seaborn as sns
from numba import jit
from scipy import stats
import os
import scienceplots
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.colors as mcolors
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull
from src.util_functions import _compute_voronoi_volumes, process_plot, _create_subplots


# Global settings
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def _statistics(data: np.array, alpha: float = 0.95) -> dict[str, np.array]:
    """
    Utility function to calculate statistic parameters of the
    :param data: (np.array) Data to calculate the statistic parameters
    :param alpha: (float) Confidence level
    :return: (dict) Dictionary with the mean, variance and standard deviation
    """
    results_avg, results_var, results_std = [], [], []

    for current_bin in data:
        avg, var, std = stats.bayes_mvs(current_bin, alpha=alpha)
        results_avg.append(avg)
        results_var.append(var)
        results_std.append(std)

    return {'mean': results_avg, 'var': results_var, 'std': results_std}


def _plot_correlation_fun(omega: np.array,
                          params_red: dict[str, np.array],
                          params_blue: dict[str, np.array],
                          plot_params: list | None = None,
                          plot_conf_interval: bool = True,
                          **kwargs) -> None:
    """
    Private function to plot results of the two-point correlation function
    :param omega: (np.array) Angular distance
    :param params_red: (dict) Parameters for the red galaxies
    :param params_blue: (dict) Parameters for the blue galaxies
    :param plot_params: (list) Parameters to plot
    :param plot_conf_interval: (bool) Plot confidence interval
    :param kwargs: Additional arguments
    :return:
    """

    plt.style.use(['science', 'ieee'])

    if plot_params is None:
        plot_params: list = ['mean']
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3)

    assert ('mean' in plot_params and plot_conf_interval is True) or plot_conf_interval is False

    if plot_conf_interval:
        ax[0].fill_between(omega[:-1], [i.minmax[0] for i in params_red['mean']],
                           [i.minmax[1] for i in params_red['mean']], edgecolor='black', facecolor='white',
                           alpha=0.6, ls='--', label=r'\( z_{0.95} \)', hatch=r'\\\\', zorder=1)

        ax[1].fill_between(omega[:-1], [i.minmax[0] for i in params_blue['mean']],
                           [i.minmax[1] for i in params_blue['mean']], edgecolor='black', facecolor='white',
                           alpha=0.6, ls='--', label=r'\( z_{0.95} \)', hatch=r'\\\\', zorder=1)

    for param in plot_params:
        ax[0].plot(omega[:-1], [i.statistic for i in params_red[param]],
                   label=r'\( \mu_{red} \)', ls='-.', lw=2, zorder=2, marker='^', markersize=10)
        ax[1].plot(omega[:-1], [i.statistic for i in params_blue[param]],
                   label=r'\( \mu_{blue} \)', ls='-.', lw=2, zorder=2, marker='^', markersize=10)

    x_min: float = min(omega)
    x_max: float = max(omega)
    y_min: float = min([i.minmax[0] for i in params_red['mean'] + params_blue['mean']])
    y_max: float = max([i.minmax[1] for i in params_red['mean'] + params_blue['mean']])

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\omega$ [rad]', fontsize=24)
    ax[0].set_ylabel(r'$\xi(\omega)$', fontsize=24)
    ax[0].legend(fontsize=15)
    ax[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    ax[0].tick_params(axis='both', which='major', labelsize=15, size=10, width=2,
                      bottom=True, top=False, left=True, right=False)
    ax[0].tick_params(axis='both', which='minor', labelsize=15, size=6.35, width=1.25,
                      bottom=True, top=False, left=False, right=False, labelbottom=False, labelleft=True)

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$\omega$ [rad]', fontsize=24)
    ax[1].set_ylabel(r'$\xi(\omega)$', fontsize=24)
    ax[1].legend(fontsize=15)
    ax[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    ax[1].tick_params(axis='both', which='major', labelsize=15, size=10, width=2,
                      bottom=True, top=False, left=True, right=False)
    ax[1].tick_params(axis='both', which='minor', labelsize=15, size=6.35, width=1.25,
                      bottom=True, top=False, left=False, right=False, labelbottom=False, labelleft=True)

    ax[0].set_xlim([x_min, x_max * 0.85])
    ax[0].set_ylim([y_min * 0.9, y_max * 1.1])

    # Set the x and y limits for the second plot
    ax[1].set_xlim([x_min, x_max * 0.85])
    ax[1].set_ylim([y_min * 0.9, y_max * 1.1])

    save_path: str | None = kwargs.get('save_path', None)

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            raise FileNotFoundError(f"Path {save_path} does not exist")

        print('Save figure to:', save_path)
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


class SDSS:
    """
    Class for the Sloan Digital Sky Survey (SDSS) data.
    The class provides methods to filter, plot, manipulate and analyze the data.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the SDSS class with the data from the CSV file.

        :param data: (Pandas DataFrame) Data from the CSV file (ra, de, z_redshift, u, g, r, i, z_magnitude, ...)
        """

        # SDSS as Pandas Dataframe
        self.data: pd.DataFrame = data

        # SDSS Parameter extracted
        self.ra = data.iloc[:, 0]
        self.de = data.iloc[:, 1]

        self._ra_standardized = None
        self._de_standardized = None

        self.z_redshift = data.iloc[:, 2]
        self.u = data.iloc[:, 3]
        self.g = data.iloc[:, 4]
        self.r = data.iloc[:, 5]
        self.i = data.iloc[:, 6]
        self.z_magnitude = data.iloc[:, 7]
        self.indices = data[(self.z_redshift > 0.08) & (self.z_redshift < 0.12)]
        self.color = self.u - self.r
        self.blue = self.color[self.color <= 2.3]
        self.red = self.color[self.color > 2.3]

        # Indices and number of red and blue galaxies
        self.blue_idx = self.color <= 2.3
        self.red_idx = self.color > 2.3
        self.num_red = len(self.red_idx[self.red_idx == True])
        self.num_blue = len(self.blue_idx[self.blue_idx == True])

        # Statistic parameters for red and blue galaxies
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
        """
        Downsample the data to the specified sample size.

        :param n: (int) Sample size
        :return: None
        """
        try:
            sampled_data_red = self.data[self.red_idx].sample(int(n / 2))
            sampled_data_blue = self.data[self.blue_idx].sample(int(n / 2))
        except ValueError:
            raise ValueError("Sample size is too large")

        sampled_data = pd.concat([sampled_data_red, sampled_data_blue])
        self.__init__(sampled_data)

    def _standardize_coordinates(self) -> None:
        """
        Standardize the coordinates.

        :return: None
        """
        self._ra_standardized = (self.ra - np.mean(self.ra)) / np.std(self.ra)
        self._de_standardized = (self.de - np.mean(self.de)) / np.std(self.de)

    def select_redshift_slice(self, z_min: float, z_max: float) -> None:
        """
        Select a slice of the redshift defined by z_min and z_max (excluding).
        :param z_min: (float) Minimum redshift
        :param z_max: (float) Maximum redshift
        :return: None
        """
        self.__init__(self.data[(self.z_redshift > z_min) & (self.z_redshift < z_max)])

    def filter_params(self) -> None:
        """
        Filter the data based on the parameters defined in the filter_params method.
        :return:
        """
        self.__init__(self.indices)

    def plot_ecdf(self, **kwargs) -> None:
        """
        Plot the empirical cumulative distribution function (eCDF) of the redshift.
        :param kwargs: Additional arguments
        :return: None
        """

        plt.style.use(['science', 'ieee', 'no-latex'])

        s_o = np.sort(self.z_redshift)
        fig, ax = plt.subplots(1, 1)
        ax.step(s_o, np.arange(len(s_o)) / len(s_o), label='empirical CDF')
        ax.set_xlabel('Redshift')
        ax.set_ylabel('eCDF')
        ax.legend(loc='best')
        ax.eventplot(s_o, lineoffsets=-0.1, linelengths=0.05, lw=0.1, colors='k')
        ax.legend(bbox_to_anchor=(0.57, 1.13), loc='upper left')

        process_plot(plt, kwargs.get('save_path', None))

    def plot_rband_redshift(self, xlim: tuple = (0, 0.6), **kwargs) -> None:
        """
        Plot the r-band magnitude as a function of the redshift.
        :param xlim: (tuple) limits for the galaxies
        :param kwargs: Additional arguments
        :return: None
        """

        plt.style.use(['science', 'ieee', 'scatter', 'no-latex'])
        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.z_redshift, self.r, marker='.', s=1.5, alpha=0.1, c='black', lw=0.0)
        ax.set_xlabel('Redshift')
        ax.set_xlim(*xlim)
        ax.set_ylim(12, 18)
        ax.set_ylabel('r-band magnitude')

        process_plot(plt, kwargs.get('save_path', None))

    def plot_colors(self, **kwargs) -> None:
        """
        Plot the u-r color as a function of the r-band magnitude.
        :param kwargs: Additional arguments
        :return: None
        """

        plt.style.use(['science', 'ieee', 'scatter', 'no-latex'])
        fig, ax = plt.subplots(1, 1)

        ax.scatter(self.r[self.blue.index], self.blue, s=0.2, alpha=0.1, lw=0.05, color='darkblue',
                   label='Blue galaxies')
        ax.scatter(self.r[self.red.index], self.red, s=0.2, alpha=0.1, lw=0.05, color='darkred', label='Red galaxies')
        ax.set_xlabel('r-band mag')
        ax.set_ylabel('u-r')

        process_plot(plt, kwargs.get('save_path', None))

    def plot_maps(self, content: list[str] | str | None = None, **kwargs) -> None:
        """
        Plot the angular and redshift-space maps for the red and blue galaxies.
        :param content: (list) Content to plot
        :param kwargs: Additional arguments
        :return: None
        """

        plt.style.use(['science', 'ieee', 'scatter', 'no-latex'])

        if isinstance(content, str):
            content = [content]
            print(content)

        if content is None or set(content) == {'angular', 'redshift'} or set(content) == {'both'}:
            fig, ax = plt.subplots(2, 2, figsize=(15, 15))
            ax[0, 0].set_title('Angular Map for blue galaxies', fontsize=20)
            ax[0, 0].set_xlabel('RA [deg]', fontsize=15)
            ax[0, 0].set_ylabel('DEC [deg]', fontsize=15)
            ax[0, 0].scatter(self.ra[self.blue.index], self.de[self.blue.index], s=0.6, alpha=0.15, color='darkblue')

            ax[0, 1].set_title('Angular Map for red galaxies', fontsize=20)
            ax[0, 1].set_xlabel('RA [deg]', fontsize=15)
            ax[0, 1].set_ylabel('DEC [deg]', fontsize=15)
            ax[0, 1].scatter(self.ra[self.red.index], self.de[self.red.index], s=0.6, alpha=0.15, color='darkred')

            ax[1, 0].set_title('Redshift-space map for blue galaxies', fontsize=20)
            ax[1, 0].set_xlabel('RA [deg]', fontsize=15)
            ax[1, 0].set_ylabel('Redshift [-]', fontsize=15)
            ax[1, 0].scatter(self.ra[self.blue.index], self.z_redshift[self.blue.index], s=0.6, alpha=0.15,
                             color='darkblue')

            ax[1, 1].set_title('Redshift-space map for red galaxies', fontsize=20)
            ax[1, 1].set_xlabel('RA [deg]', fontsize=15)
            ax[1, 1].set_ylabel('Redshift [-]', fontsize=15)
            ax[1, 1].scatter(self.ra[self.red.index], self.z_redshift[self.red.index], s=0.6, alpha=0.15,
                             color='darkred')

        elif not set(content).issubset({'angular', 'redshift'}):
            raise ValueError("Content must be either 'angular' or 'redshift' or 'both'")

        elif set(content) == {'angular'}:
            fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
            ax[0].set_title('Angular Map for blue galaxies', fontsize=20)
            ax[0].set_xlabel('RA [deg]', fontsize=15)
            ax[0].set_ylabel('DEC [deg]', fontsize=15)
            ax[0].scatter(self.ra[self.blue.index], self.de[self.blue.index], s=0.6, alpha=0.15, color='darkblue')

            ax[1].set_title('Angular Map for red galaxies', fontsize=20)
            ax[1].set_xlabel('RA [deg]', fontsize=15)
            ax[1].set_ylabel('DEC [deg]', fontsize=15)
            ax[1].scatter(self.ra[self.red.index], self.de[self.red.index], s=0.6, alpha=0.15, color='darkred')

        elif set(content) == {'redshift'}:
            fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
            ax[0].set_title('Redshift-space map for blue galaxies', fontsize=20)
            ax[0].set_xlabel('RA [deg]', fontsize=15)
            ax[0].set_ylabel('Redshift [-]', fontsize=15)
            ax[0].scatter(self.ra[self.blue.index], self.z_redshift[self.blue.index], s=0.6, alpha=0.15,
                          color='darkblue')

            ax[1].set_title('Redshift-space map for red galaxies', fontsize=20)
            ax[1].set_xlabel('RA [deg]', fontsize=15)
            ax[1].set_ylabel('Redshift [-]', fontsize=15)
            ax[1].scatter(self.ra[self.red.index], self.z_redshift[self.red.index], s=0.6, alpha=0.15,
                          color='darkred')
        else:
            raise ValueError("Unexpected error occurred")

        process_plot(plt, kwargs.get('save_path', None))

    def nearest_neighbor_estimation(self, n_neighbors: np.ndarray,
                                    use_standardized_vals: bool = True,
                                    bins: int = 50,
                                    **kwargs) -> None:
        """
        Nearest neighbor estimation.

        :param bins:
        :param n_neighbors: (np.ndarray) Array of neighbors
        :param use_standardized_vals: (bool) Standardize values
        :param kwargs: Additional arguments
        :return: None
        """

        plt.style.use(['science', 'ieee', 'no-latex'])

        # Check whether to use standardized values without dimension or not
        if use_standardized_vals:

            if self._ra_standardized is None or self._de_standardized is None:
                self._standardize_coordinates()

            ra = self._ra_standardized
            de = self._de_standardized
        else:
            ra = self.ra
            de = self.de

        plots = _create_subplots(len(n_neighbors))

        for i, n in enumerate(n_neighbors):
            if len(n_neighbors) > 3:
                current_ax = plots.axes[i // 3, i % 3]
            elif len(n_neighbors) > 1:
                current_ax = plots.axes[i]
            else:
                current_ax = plots.axes

            nn = NearestNeighbors(n_neighbors=n, algorithm='ball_tree')
            nn.fit(np.vstack([ra, de]).T)

            distances, indices = nn.kneighbors()
            density = 1 / (np.pi * distances[:, -1] ** 2)

            current_ax.hist(density, bins=bins, alpha=0.6, color='darkblue')
            current_ax.set_title(f'Nearest neighbor estimation for n = {n}', fontsize=18)
            current_ax.set_xlabel('Density', fontsize=12)
            current_ax.set_ylabel('Frequency', fontsize=12)

        process_plot(plt, kwargs.get('save_path', None))

    def delaunay_triangulation(self, use_standardized_vals: bool = True, **kwargs) -> None:
        """
        Delaunay triangulation.

        :param use_standardized_vals:
        :param kwargs:
        :return:
        """

        plt.style.use(['science', 'ieee', 'no-latex'])

        # Check whether to use standardized values without dimension or not
        if use_standardized_vals:

            if self._ra_standardized is None or self._de_standardized is None:
                self._standardize_coordinates()

            ra = self._ra_standardized
            de = self._de_standardized
            unit = '-'
        else:
            ra = self.ra
            de = self.de
            unit = 'deg'

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.triplot(ra, de, 'go-', lw=0.4, markersize=0.3, alpha=0.6)
        ax.set_title('Delaunay Triangulation', fontsize=20)
        ax.set_xlabel(f'RA [{unit}]', fontsize=15)
        ax.set_ylabel(f'DEC [{unit}]', fontsize=15)

        process_plot(plt, kwargs.get('save_path', None))

    def voroni_volumes(self, use_standardized_vals: bool = True, **kwargs) -> None:
        """
        Voronoi diagram.

        :param use_standardized_vals:
        :param kwargs:
        :return:
        """

        plt.style.use(['science', 'ieee', 'no-latex'])

        # Check whether to use standardized values without dimension or not
        if use_standardized_vals:

            if self._ra_standardized is None or self._de_standardized is None:
                self._standardize_coordinates()

            ra = self._ra_standardized
            de = self._de_standardized
            unit = '-'
        else:
            ra = self.ra
            de = self.de
            unit = 'deg'

        vor = Voronoi(np.vstack([ra, de]).T)
        volumes = _compute_voronoi_volumes(vor)  # Not used

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='darkblue',
                        line_width=0.4, line_alpha=0.6, point_size=0.5)
        ax.set_title('Voronoi Diagram', fontsize=20)
        ax.set_xlabel(f'RA {unit}', fontsize=15)
        ax.set_ylabel(f'DEC [deg] {unit}', fontsize=15)

        process_plot(plt, kwargs.get('save_path', None))

    def kernel_density_estimation(self,
                                  bandwidths: np.ndarray,
                                  n_bins: int = 50,
                                  use_standardizes_vals: bool = True,
                                  **kwargs) -> None:
        """
        Kernel density estimation.

        :param bandwidths: (np.ndarray) Bandwidths
        :param n_bins: (int) Number of bins
        :param use_standardizes_vals: (bool) Standardize values
        :param kwargs: Additional arguments
        :return: None
        """

        plt.style.use(['science', 'ieee', 'no-latex'])

        # Check whether to use standardized values without dimension or not
        if use_standardizes_vals:
            self._standardize_coordinates()

            ra = self._ra_standardized
            de = self._de_standardized
            unit = '-'
        else:
            ra = self.ra
            de = self.de
            unit = 'deg'

        x_min, x_max = np.min(ra), np.max(ra)
        y_min, y_max = np.min(de), np.max(de)

        x, y = np.meshgrid(np.linspace(x_min, x_max, n_bins), np.linspace(y_min, y_max, n_bins))

        rows: int = int(np.ceil(len(bandwidths) / 3))
        cols: int = 3 if len(bandwidths) > 3 else len(bandwidths)

        fig_height: float = rows * 5
        fig_width: float = 18

        fig, ax = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

        for i, bw in enumerate(bandwidths):
            if len(bandwidths) > 3:
                current_ax = ax[i // 3, i % 3]
            elif len(bandwidths) > 1:
                current_ax = ax[i]
            else:
                current_ax = ax

            kde = KernelDensity(kernel='gaussian', bandwidth=bw)
            kde.fit(np.vstack([ra, de]).T)

            z = np.exp(kde.score_samples(np.vstack([x.ravel(), y.ravel()]).T))
            z = z.reshape(x.shape)

            contour = current_ax.contourf(x, y, z, cmap='viridis')
            current_ax.set_title(f'Bandwidth = {bw}', fontsize=18)
            current_ax.set_xlabel(f'RA [{unit}]', fontsize=12)
            current_ax.set_ylabel(f'DEC [{unit}]', fontsize=12)

            cbar = fig.colorbar(contour, ax=current_ax)
            cbar.set_label('Density', fontsize=12)

        process_plot(plt, kwargs.get('save_path', None))

    def plot_2d_histograms(self, bins: float | list | np.ndarray | None = None, **kwargs) -> None:
        """
        Plot the two-dimensional density of the red and blue galaxies.
        :param bins: (float) Bins for the histogram
        :param kwargs: Additional arguments
        :return: None
        """

        global hist
        plt.style.use(['science', 'ieee', 'scatter', 'no-latex'])

        if isinstance(bins, int):
            bins = [bins]
        elif bins is None:
            bins = [50, 100, 500]

        plots = _create_subplots(len(bins))

        for i, bw in enumerate(bins):
            if len(bins) > 3:
                current_ax = plots.axes[i // 3, i % 3]
            elif len(bins) > 1:
                current_ax = plots.axes[i]
            else:
                current_ax = plots.axes

            current_ax.set_title(f'2D Histogram for binwidth = {bw}', fontsize=18)
            hist = current_ax.hist2d(self.ra, self.de, bins=bw, cmap='plasma', alpha=0.8, vmin=0, vmax=20)
            current_ax.set_xlabel('RA [deg]', fontsize=12)
            current_ax.set_ylabel('DEC [deg]', fontsize=12)

        cbar_size = 14 / plots.height if plots.height > 5 else 5
        cbar = plt.colorbar(hist[3], ax=plots.axes, shrink=cbar_size)
        cbar.set_label('Density', fontsize=12)

        process_plot(plt, kwargs.get('save_path', None))

    def _generate_random_positions(self, m: int | None = None) -> None:
        """
        Generate random positions for the two-point correlation function.

        :param m: (int) Number of random positions
        :return: None
        """

        if m is None:
            m = self.num_red if self.num_red < self.num_blue else self.num_blue

        self.random_phi = np.random.uniform(130, 230, size=m) / 180 * np.pi
        self.random_theta = np.pi / 2 - np.random.uniform(5, 65, size=m) / 180 * np.pi

    def _two_point_correlation(self, plot: bool = False, **kwargs) -> np.array:
        """
        Compute the two-point correlation function for the red and blue galaxies.

        :param plot: (bool) Plot the results
        :param kwargs: Additional arguments
        :return: (np.array) Two-point correlation function for the red and blue galaxies
        """

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

        # Compute the histogram using the actual positions
        dd_counts_red, _ = np.histogram(data_positions_red, bins=omega)
        dd_counts_blue, _ = np.histogram(data_positions_blue, bins=omega)

        # Compute the histogram using random positions
        rr_counts_red, _ = np.histogram(random_positions_red, bins=omega)
        rr_counts_blue, _ = np.histogram(random_positions_blue, bins=omega)

        # Compute the histogram using actual and random positions
        dr_counts_red, _ = np.histogram(data_random_positions_red, bins=omega)
        dr_counts_blue, _ = np.histogram(data_random_positions_blue, bins=omega)

        # Compute the Landy-Szalay estimator
        correlation_red = landy_szalay(dd_counts_red, dr_counts_red, rr_counts_red, m)
        correlation_blue = landy_szalay(dd_counts_blue, dr_counts_blue, rr_counts_blue, m)

        if plot:
            _plot_correlation_fun(omega, correlation_red, correlation_blue)

        return correlation_blue, correlation_red

    def two_point_correlation(self, iterations: int = 100, m_samples: int = 100, **kwargs) -> None:
        """
        Compute the two-point correlation function for the red and blue galaxies.

        :param iterations: (int) Number of iterations
        :param m_samples: (int) Size of samples
        :param kwargs: Additional arguments
        :return: None
        """

        print(f"Computing two-point correlation function for {iterations} iterations with {m_samples} samples")

        results_red: np.array = []
        results_blue: np.array = []

        for i in range(iterations):
            print(f"iteration {i + 1}/{iterations}")
            c_blue, c_red = self._two_point_correlation(m=m_samples, plot=False)
            results_blue.append(c_blue)
            results_red.append(c_red)

        results_blue = np.array(results_blue)
        results_red = np.array(results_red)

        self.correlation_results_blue = _statistics(results_blue.T)
        self.correlation_results_red = _statistics(results_red.T)

        if kwargs.get('plot', False):

            if kwargs.get('save_path', False):
                self.plot_correlation(save_path=kwargs.get('save_path'))
                return

            self.plot_correlation()

    def plot_correlation(self, **kwargs) -> None:
        """
        Plot the results of the two-point correlation function.

        :param kwargs: Additional arguments
        :return: None
        """

        if self.correlation_results_red is not None and self.correlation_results_blue is not None:
            omega = np.geomspace(0.003, 0.3, 11)
            _plot_correlation_fun(omega, self.correlation_results_red, self.correlation_results_blue, **kwargs)
        else:
            raise ValueError("Two-point correlation function has not been computed yet")


if __name__ == "__main__":

    pass
