import numpy as np
from sklearn.cluster import KMeans


def initialize_parameters_with_kmeans(
    X: np.ndarray, nr_of_modes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = KMeans(n_clusters=nr_of_modes).fit(X.reshape(-1, 1)).labels_
    means = np.array([X[labels == i].mean() for i in range(nr_of_modes)])
    stds = np.array([X[labels == i].std() for i in range(nr_of_modes)])
    weights = np.array([np.mean(labels == i) for i in range(nr_of_modes)])
    return weights, stds, means


def initialize_parameters_with_quantiles(
    X: np.ndarray, nr_of_modes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = np.percentile(X, np.linspace(0, 100, nr_of_modes + 1))
    means = np.array(
        [(quantiles[i] + quantiles[i + 1]) / 2 for i in range(nr_of_modes)]
    )
    stds = np.array(
        [
            np.std(X[(X >= quantiles[i]) & (X < quantiles[i + 1])])
            for i in range(nr_of_modes)
        ]
    )
    weights = np.array(
        [
            len(X[(X >= quantiles[i]) & (X < quantiles[i + 1])]) / len(X)
            for i in range(nr_of_modes)
        ]
    )
    return weights, stds, means


METHOD_TO_INITIALIZE_PARAMETERS = {
    "kmeans": initialize_parameters_with_kmeans,
    "quantiles": initialize_parameters_with_quantiles,
}
