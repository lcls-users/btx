# Deprecated
# TODO tentative refactor, needs to be tested
#def calculate_histograms(data, bin_boundaries, hist_start_bin):
#    bins = len(bin_boundaries) - 1
#    rows, cols = data.shape[1], data.shape[2]
#    hist_shape = (bins, rows, cols)
#
#    reshaped_data = data.reshape(-1, rows * cols)
#    bin_indices = np.digitize(reshaped_data, bin_boundaries, right=False)
#
#    histograms = np.zeros(hist_shape, dtype=np.float64)
#
#    for i in range(rows * cols):
#        hist_slice = histograms[:, i // cols, i % cols]
#        hist_slice += np.bincount(bin_indices[:, i], minlength=bins)
#
#    histograms += 1e-9
#    normalized_histograms = histograms / np.sum(histograms, axis=0)
#
#    return normalized_histograms[hist_start_bin:, :, :]


