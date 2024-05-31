import numpy as np
from scipy.ndimage import label, binary_dilation
import functools
import hashlib
import random
from numba import jit

def memoize_subsampled(func):
    """Memoize a function by creating a hashable key using deterministically subsampled data."""
    cache = {}

    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        # Generate a hashable key from a deterministic subsample
        shape_str = str(data.shape)  # Convert shape to string to use it as a seed
        seed_value = int(hashlib.sha256(shape_str.encode()).hexdigest(), 16) % 10**8
        random.seed(seed_value)

        subsample_size = min(100, data.shape[0])  # Limit the subsample size to a maximum of 100
        subsample_indices = random.sample(range(data.shape[0]), subsample_size)
        subsample = data[subsample_indices]

        hashable_key = hashlib.sha256(subsample.tobytes()).hexdigest()

        # Check cache
        if hashable_key in cache:
            return cache[hashable_key]

        # Calculate the result and store it in the cache
        result = func(data, *args, **kwargs)
        cache[hashable_key] = result

        return result

    return wrapper

def identify_roi_connected_cluster(p_values, threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    """Find the cluster connected to the ROI."""
    porous_pixels = p_values > threshold
    labeled_array, _ = label(porous_pixels)
    seed_x = (roi_x_start + roi_x_end) // 2
    seed_y = (roi_y_start + roi_y_end) // 2
    roi_cluster_label = labeled_array[seed_x, seed_y]
    return labeled_array, labeled_array == roi_cluster_label

def create_continuous_buffer(signal_mask: np.ndarray, initial_thickness: int = 10,
                             num_pixels: int = None, separator_thickness: int = 1) -> np.ndarray:
    """
    Create a continuous buffer around a signal mask with a gap, targeting a specific number of pixels.

    Args:
        signal_mask (np.ndarray): The original signal mask.
        initial_thickness (int): The initial thickness for dilation.
        num_pixels (int, optional): The target number of pixels in the buffer.
        separator_thickness (int): The thickness of the gap between the signal mask and the buffer.

    Returns:
        np.ndarray: The created buffer.
    """
    if num_pixels > np.prod(signal_mask.shape) - np.sum(signal_mask):
        raise ValueError
    assert signal_mask.sum() > 0
    
    dilated_signal = binary_dilation(signal_mask, iterations=separator_thickness)

    current_num_pixels = 0
    thickness = 0
    while num_pixels is not None and current_num_pixels < num_pixels:
        thickness += 1
        buffer = binary_dilation(dilated_signal, iterations=thickness) & (~dilated_signal)
        current_num_pixels = np.sum(buffer)

    return buffer


def create_background_mask(signal_mask, background_mask_multiple, thickness, separator_thickness = 5):
    """
    Creates a background mask based on the given signal mask, a multiple of its size, and thickness.

    Parameters:
    - signal_mask: numpy.ndarray, a boolean array representing the signal mask.
    - background_mask_multiple: float, multiple of the number of pixels in the signal mask for the background mask.
    - thickness: int, thickness of the background buffer.

    Returns:
    - numpy.ndarray, the calculated background mask.
    """
    num_pixels_signal_mask = np.sum(signal_mask)
    num_pixels_background_mask = int(num_pixels_signal_mask * background_mask_multiple)

    background_mask = create_continuous_buffer(signal_mask, 
                                               initial_thickness=thickness, 
                                               num_pixels=num_pixels_background_mask,
                                               separator_thickness=separator_thickness)
    
    return background_mask

def calculate_histograms(data, bin_boundaries=np.arange(5, 30, 0.2), hist_start_bin=1):
    """
    Calculate histograms for each pixel in the data.

    Args:
        data (ndarray): 3D array of pixel values (frames, rows, cols)
        bin_boundaries (ndarray): Array of histogram bin boundaries
        hist_start_bin (int): Index of the first bin to include in the output

    Returns:
        ndarray: 3D array of histograms (bins, rows, cols)
    """
    bins = len(bin_boundaries) - 1
    rows, cols = data.shape[1], data.shape[2]
    hist_shape = (bins, rows, cols)

    # Reshape the data for easier computation
    reshaped_data = data.reshape(-1, rows * cols)

    # Perform digitization
    bin_indices = np.digitize(reshaped_data, bin_boundaries)

    # Initialize histograms
    histograms = np.zeros(hist_shape, dtype=np.float64)

    # Populate histograms using bincount and sum along the zeroth axis
    for i in range(rows * cols):
        valid_indices = bin_indices[:, i] < bins  # Exclude indices that fall outside the bin range or equal to the last boundary
        histograms[:, i // cols, i % cols] = np.bincount(bin_indices[:, i][valid_indices], minlength=bins)
        # TODO efficiency
        # counts beyond max go into the first bin, otherwise they don't
        # contribute to the EMD
        histograms[hist_start_bin, i // cols, i % cols] += np.sum(reshaped_data[:, i] > bin_boundaries[-1])

    # Add small constant
    histograms += 1e-9
    normalized_histograms = histograms 

    return normalized_histograms[hist_start_bin:, :, :]

calculate_histograms = jit(nopython=True)(calculate_histograms)
calculate_histograms = memoize_subsampled(calculate_histograms)

def get_average_roi_histogram(histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    """Calculate the average histogram for the ROI."""
    roi_histograms = histograms[:, roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    average_roi_histogram = np.mean(roi_histograms, axis=(1, 2))
    return average_roi_histogram / np.sum(average_roi_histogram)

@jit(nopython=True)
def wasserstein_distance(u, v):
    cdf_u = np.cumsum(u)
    cdf_v = np.cumsum(v)
    return np.sum(np.abs(cdf_u - cdf_v))

