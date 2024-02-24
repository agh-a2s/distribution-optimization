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


def reals_to_simplex(
    x: np.ndarray,
) -> np.ndarray:
    return (1 - x) * np.cumprod(np.concatenate([np.array([1]), x[:-1]]))


def simplex_to_reals(x: np.ndarray) -> np.ndarray:
    x_cum_sum = np.cumsum(x)
    return (x_cum_sum - 1) / (np.concatenate([np.array([0]), x_cum_sum[:-1]]) - 1)


def reals_to_reals_with_offset(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x)


def reals_with_offset_to_reals(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:1], np.diff(x)])