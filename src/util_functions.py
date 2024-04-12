from scipy.spatial import ConvexHull
import numpy as np
from scipy.spatial import Voronoi


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
