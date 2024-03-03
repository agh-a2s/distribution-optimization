import numpy as np
from scipy.stats import norm

DEFAULT_NR_OF_BINS = 10


def mixture_probability(
    x: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    weights: np.ndarray,
    normalize: bool | None = True,
) -> np.ndarray:
    values = norm.pdf(x, means, sds) * weights
    if normalize:
        return values / np.sum(values)
    return values


def cdf_mixtures(kernel: np.ndarray, means: np.ndarray, sds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cdf_values = []
    for mean_, sd_ in zip(means, sds):
        cdf_values.append(norm.cdf(kernel, mean_, sd_))
    return np.array(cdf_values).T @ weights


def bin_prob_for_mixtures(means: np.ndarray, sds: np.ndarray, weights: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    cdfs = cdf_mixtures(breaks, means, sds, weights)
    return cdfs[1:] - cdfs[:-1]


def std_dev_estimate(
    data: np.ndarray,
    q: list[int] | None = [25, 75],
    method: str | None = "median_unbiased",
) -> float:
    """
    Keating, J.P. (1999). A Primer on Density Estimation for the Great Home Run Race of '98.
    """
    p = np.percentile(data, q=q, method=method)  # type: ignore[call-overload]
    interquantile_range = p[1] - p[0]
    normalizing_constant = 1.349
    iqr_std_dev_estimate = interquantile_range / normalizing_constant
    return min(np.std(data), iqr_std_dev_estimate)


def optimal_no_bins(
    data: np.ndarray,
) -> int:
    """
    Keating, J.P. (1999). A Primer on Density Estimation for the Great Home Run Race of '98.
    """
    sigma = std_dev_estimate(data)
    opt_bin_width = 3.49 * sigma / (len(data)) ** (1 / 3)
    if opt_bin_width > 0:
        data_range = np.max(data) - np.min(data)
        return max(
            int(np.ceil(data_range / opt_bin_width)),
            DEFAULT_NR_OF_BINS,
        )
    return DEFAULT_NR_OF_BINS
