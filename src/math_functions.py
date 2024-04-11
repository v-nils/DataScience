from numba import jit
import numpy as np


@jit(nopython=True)
def eq_omega(theta_1: float, theta_2: float, phi_1: float, phi_2: float) -> float:

    return np.arccos(np.cos(theta_1) * np.cos(theta_2) + np.sin(theta_1) * np.sin(theta_2) * np.cos(phi_1 - phi_2))

@jit(nopython=True)
def compute_angles(theta_1: np.array,
                   phi_1: np.array,
                   theta_2: np.array = None,
                   phi_2: np.array = None) -> np.array:

    if theta_2 is not None and phi_2 is not None:
        #assert theta_1.shape == theta_2.shape == phi_1.shape == phi_2.shape
        pass
    elif theta_2 is None and phi_2 is None:
        #assert theta_1.shape == phi_1.shape
        theta_2 = theta_1
        phi_2 = phi_1
    else:
        raise ValueError("theta_2 and phi_2 must be both None or both not None")

    n = len(theta_1)
    omega = np.zeros((n, n))
    for i in range(n):
        for j in range(i+ 1, n):
            omega[i, j] = eq_omega(theta_1[i], theta_2[j], phi_1[i], phi_2[j])
    return omega.flatten()


def landy_szalay(dd: np.array, dr: np.array, rr: np.array, n: int) -> np.array:

    m = n

    rr[rr == 0] = 1

    correlation_fun = 1 + m * (m - 1) * dd / (n * (n - 1) * rr) - m * (m - 1) * dr / (n * m * rr)

    return correlation_fun




