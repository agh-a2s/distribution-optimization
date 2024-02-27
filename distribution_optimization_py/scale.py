import numpy as np


def scale_linearly(
    x: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
) -> np.ndarray:
    normalized_x = (x - source_lower) / (source_upper - source_lower)
    return target_lower + normalized_x * (target_upper - target_lower)


def scale_uniformly_simplex(x: np.ndarray) -> np.ndarray:
    n = len(x)
    return np.array([np.power(x_, 1 / (n - k)) for k, x_ in enumerate(x)])


def unscale_uniformly_simplex(x: np.ndarray) -> np.ndarray:
    n = len(x)
    return np.array([np.power(x_, n - k) for k, x_ in enumerate(x)])


def reals_to_full_simplex(
    x: np.ndarray,
) -> np.ndarray:
    return (1 - x) * np.cumprod(np.concatenate([np.array([1]), x[:-1]]))


def full_simplex_to_reals(x: np.ndarray) -> np.ndarray:
    x_cum_sum = np.cumsum(x)
    return (x_cum_sum - 1) / (np.concatenate([np.array([0]), x_cum_sum[:-1]]) - 1)


def reals_to_simplex(
    x: np.ndarray,
) -> np.ndarray:
    return np.concatenate([(1 - x), np.array([1])]) * np.cumprod(np.concatenate([np.array([1]), x]))


def simplex_to_reals(x: np.ndarray) -> np.ndarray:
    flipped_cumsum = np.flip(np.cumsum(np.flip(x[1:])))
    return flipped_cumsum / (x[:-1] + flipped_cumsum)


def reals_to_reals_with_offset(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x)


def reals_with_offset_to_reals(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:1], np.diff(x)])


def reals_to_uniform_simplex(x: np.ndarray) -> np.ndarray:
    n = x.size
    ksi = np.log(x) / np.arange(n, 0, -1)
    logvalues = np.concatenate((np.log(-np.expm1(ksi)), (0,))) + np.concatenate(((0,), np.cumsum(ksi)))
    values = np.exp(logvalues - logvalues.max())
    return values / values.sum()


def uniform_simplex_to_reals(x: np.ndarray) -> np.ndarray:
    n = x.size - 1
    logvalues = -np.arange(n, 0, -1) * np.log1p(x[:-1] / np.cumsum(x[-1:0:-1])[::-1])
    return np.exp(logvalues)
