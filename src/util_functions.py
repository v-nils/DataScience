import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from scipy.spatial import Voronoi
from dataclasses import dataclass


def process_plot(plot: plt, save_path: str | None = None) -> None:
    """
    Save the plot to the specified path, or display it, based on the value of save_path.

    :param plot: matplotlib.pyplot object
    :param save_path: Path to save the plot. If None, the plot is displayed.
    """
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            raise FileNotFoundError(f"Path {save_path} does not exist")

        print('Save figure to:', save_path)
        plot.savefig(save_path, bbox_inches='tight')
    else:
        plot.show()


def _compute_voronoi_volumes(v: Voronoi):
    """
    Compute the volume of the voronoi cells.

    :param v: Voronoi object
    :return:
    """

    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices or len(indices) == 0:
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol


@dataclass
class Subplot:

    height: float
    width: float

    figure: plt.Figure
    axes: plt.Axes


def _create_subplots(n_subplots: int):

    rows: int = int(np.ceil(n_subplots / 3))
    cols: int = 3 if n_subplots > 3 else n_subplots

    fig_height: float = rows * 5
    fig_width: float = 18

    fig, ax = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    return Subplot(height=fig_height, width=fig_width, figure=fig, axes=ax)
