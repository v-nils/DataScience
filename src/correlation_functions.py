from numba import jit
import numpy as np


@jit(nopython=True)
def angles(theta, phi, outfile: str | None = None) -> np.array:
    result = np.zeros((len(phi) - 1, len(theta) - 1))
    k = 0
    for i in range(len(phi) - 1):
        for j in range(len(theta) - 1):
            m = np.arccos(np.cos(theta[j]) * np.cos(theta[j + 1]) + np.cos(phi[i] - phi[i + 1]) * np.sin(theta[j]) * np.sin(theta[j + 1]))
            result[i, j] = m
            k += 1
            if k % 100_000_000 == 0:
                print(k)
    result = result.flatten()

    return result
