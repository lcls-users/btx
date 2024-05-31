import numpy as np
from scipy.ndimage import label
from pvalues import  calculate_p_values as _calculate_p_values
from histogram_analysis import identify_roi_connected_cluster, create_background_mask

# TODO this is currently not being used
from pvalues import calculate_emd_values
def calculate_emd(histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end, average_histogram = None):
    """
    Calculates Earth Mover's Distance (EMD) values for an array of histograms.
    
    Args:
        histograms (ndarray): 3D array of histograms with shape (bins, rows, cols).
        roi_x_start (int): Start index of the ROI along the x-axis.
        roi_x_end (int): End index of the ROI along the x-axis.
        roi_y_start (int): Start index of the ROI along the y-axis.
        roi_y_end (int): End index of the ROI along the y-axis.
        average_histogram (ndarray, optional): Average histogram to use. If not provided, it will be calculated from the ROI.
        
    Returns:
        ndarray: 2D array of EMD values with shape (rows, cols).
    """
    if histograms.ndim != 3:
        raise ValueError(f"Expected 3D histograms array, got {histograms.ndim}D")
    if not (0 <= roi_x_start < roi_x_end <= histograms.shape[1] and 
            0 <= roi_y_start < roi_y_end <= histograms.shape[2]):
        raise ValueError(f"Invalid ROI coordinates for histogram of shape {histograms.shape}")
        
    if average_histogram is None:
        average_histogram = np.mean(histograms[:, roi_x_start:roi_x_end, roi_y_start:roi_y_end], axis=(1,2))
    
    return calculate_emd_values(histograms, average_histogram)
    

def calculate_p_values(emd_values, null_distribution):
    """
    Calculates p-values from EMD values and a null distribution.
    
    Args:
        emd_values (ndarray): 2D array of EMD values with shape (rows, cols).
        null_distribution (ndarray): 1D array of null distribution EMD values.
        
    Returns:
        ndarray: 2D array of p-values with shape (rows, cols).
    """
    if emd_values.ndim != 2:
        raise ValueError(f"Expected 2D EMD values array, got {emd_values.ndim}D")
    if null_distribution.ndim != 1:
        raise ValueError(f"Expected 1D null distribution array, got {null_distribution.ndim}D")
    
    return _calculate_p_values(emd_values, null_distribution)


import numpy as np
from scipy.ndimage import label

def identify_roi_connected_cluster(p_values, threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    """Find the cluster connected to the ROI."""
    porous_pixels = p_values > threshold
    labeled_array, _ = label(porous_pixels)
    seed_x = (roi_x_start + roi_x_end) // 2
    seed_y = (roi_y_start + roi_y_end) // 2
    roi_cluster_label = labeled_array[seed_x, seed_y]
    return labeled_array, labeled_array == roi_cluster_label

def rectify_filter_mask(mask, data):
    """Ensure the mask is oriented correctly based on mean values of the data."""
    imgs_sum = data.sum(axis=0)
    if mask.sum() == 0:
        return ~mask
    mean_1 = imgs_sum[mask].mean()
    mean_0 = imgs_sum[~mask].mean()
    if mean_1 < mean_0:
        return ~mask
    else:
        return mask

def filter_negative_clusters_by_size(cluster_array, data, M=10):
    """Filter out small negative clusters."""
    cluster_array = rectify_filter_mask(cluster_array, data)
    inverted_array = np.logical_not(cluster_array)
    labeled_array, num_features = label(inverted_array)
    cluster_sizes = np.bincount(labeled_array.ravel())
    small_clusters = np.where(cluster_sizes < M)[0]
    small_cluster_mask = np.isin(labeled_array, small_clusters)
    return np.logical_or(cluster_array, small_cluster_mask)

def infill_binary_array(data, array):
    """Fill in any gaps or holes in the binary array."""
    labeled_array, num_features = label(rectify_filter_mask(array, data))
    largest_component = 0
    largest_size = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled_array == i)
        if size > largest_size:
            largest_size = size
            largest_component = i
    infilled_array = (labeled_array == largest_component)
    return infilled_array

def generate_signal_mask(p_values, threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end, data, min_cluster_size=50):
    """
    Generate a signal mask from an array of p-values and a threshold.
    
    Args:
        p_values (np.ndarray): Array of p-values.
        threshold (float): Threshold value for p-values.
        roi_coordinates (tuple): Coordinates of the region of interest (ROI) in the format (roi_x_start, roi_x_end, roi_y_start, roi_y_end).
        data (np.ndarray): Original data array.
        min_cluster_size (int, optional): Minimum size of clusters to be considered as signal. Default is 50.
    
    Returns:
        np.ndarray: Boolean mask indicating the signal region.
    """
    # Identify the cluster connected to the ROI
    labeled_array, roi_connected_cluster = identify_roi_connected_cluster(
        p_values, threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end
    )
    
    # Filter out small negative clusters
    signal_mask = filter_negative_clusters_by_size(roi_connected_cluster, data, M=min_cluster_size)
    
    # Fill in any gaps or holes in the signal mask
    signal_mask = infill_binary_array(data, signal_mask)
    
    return signal_mask
    
def generate_background_mask(signal_mask, background_mask_multiple, thickness):
    """
    Generates a background mask from a signal mask.

    Args:
        signal_mask (ndarray): 2D binary array indicating signal pixels.
        background_mask_multiple (float): Multiple of the number of pixels in the signal mask to include in the background.
        thickness (int): Thickness of the background region around the signal.

    Returns:
        ndarray: 2D binary background mask with shape (rows, cols).
    """
    if signal_mask.ndim != 2 or signal_mask.dtype != bool:
        raise ValueError(f"Expected 2D boolean signal mask, got {signal_mask.ndim}D {signal_mask.dtype}")
    if background_mask_multiple < 0:
        raise ValueError(f"Background mask multiple must be non-negative, got {background_mask_multiple}")
    if thickness < 0:
        raise ValueError(f"Background mask thickness must be non-negative, got {thickness}")

    return create_background_mask(signal_mask, background_mask_multiple, thickness)
