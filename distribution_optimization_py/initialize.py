import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans


def initialize_parameters_with_kmeans(X: np.ndarray, nr_of_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = KMeans(n_clusters=nr_of_modes).fit(X.reshape(-1, 1)).labels_
    means = np.array([X[labels == i].mean() for i in range(nr_of_modes)])
    stds = np.array([X[labels == i].std() for i in range(nr_of_modes)])
    weights = np.array([np.mean(labels == i) for i in range(nr_of_modes)])
    return weights, stds, means


def initialize_parameters_with_quantiles(X: np.ndarray, nr_of_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = np.percentile(X, np.linspace(0, 100, nr_of_modes + 1))
    means = np.array([(quantiles[i] + quantiles[i + 1]) / 2 for i in range(nr_of_modes)])
    stds = np.array([np.std(X[(X >= quantiles[i]) & (X < quantiles[i + 1])]) for i in range(nr_of_modes)])
    weights = np.array([len(X[(X >= quantiles[i]) & (X < quantiles[i + 1])]) / len(X) for i in range(nr_of_modes)])
    return weights, stds, means


def initialize_parameters_with_kde(X: np.ndarray, nr_of_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kde = gaussian_kde(X)
    data_range = np.linspace(np.min(X), np.max(X), len(X))
    density = kde(data_range)
    peaks, _ = find_peaks(density, height=0.01)
    if len(peaks) < nr_of_modes:
        raise ValueError("Couldn't find enough peaks to initialize means.")
    peak_positions = data_range[peaks][:nr_of_modes]
    peak_heights = density[peaks][:nr_of_modes]
    weights = peak_heights / np.sum(peak_heights)
    std_devs = np.zeros(nr_of_modes)
    for i, peak_pos in enumerate(peak_positions):
        half_max = peak_heights[i] / 2
        left_half_max_idx = np.argmax(density[: peaks[i]] < half_max)
        right_half_max_idx = peaks[i] + np.argmax(density[peaks[i] :] < half_max)
        std_devs[i] = (data_range[right_half_max_idx] - data_range[left_half_max_idx]) / 2

    return weights, std_devs, peak_positions


METHOD_TO_INITIALIZE_PARAMETERS = {
    "kmeans": initialize_parameters_with_kmeans,
    "quantiles": initialize_parameters_with_quantiles,
    "kde": initialize_parameters_with_kde,
}
