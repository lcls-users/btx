import numpy as np
from numba import jit

@jit(nopython=True)
def wasserstein_distance(u, v):
    cdf_u = np.cumsum(u)
    cdf_v = np.cumsum(v)
    return np.sum(np.abs(cdf_u - cdf_v))

@jit(nopython=True)
def calculate_emd_values(histograms, average_histogram):
    """Compute the Earth Mover's Distance for each histogram."""
    emd_values = np.zeros((histograms.shape[1], histograms.shape[2]))
    for i in range(histograms.shape[1]):
        for j in range(histograms.shape[2]):
            emd_values[i, j] = wasserstein_distance(histograms[:, i, j], average_histogram)
    return emd_values

def calculate_p_values(emd_values, null_distribution):
    """calculate p-values based on the observed EMD values and the null distribution."""
    return 1 - np.mean(emd_values[:, :, np.newaxis] >= null_distribution, axis=2)
