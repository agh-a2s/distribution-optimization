import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm

DEFAULT_NR_OF_BINS = 10
DEFAULT_N_SAMPLES = 100000


def solution_to_params(solution: np.ndarray):
    n_components = solution.shape[0] // 3
    weights = solution[:n_components]
    sds = solution[n_components : 2 * n_components]
    means = solution[2 * n_components :]
    means_order = np.argsort(means)
    if not np.isclose(np.sum(weights), 1):
        print("Weights do not sum to 1:", np.sum(weights))
    weights = weights / np.sum(weights)
    return weights[means_order], sds[means_order], means[means_order]


def generate_gaussian_mixture_data(
    means: np.ndarray,
    sds: np.ndarray,
    weights: np.ndarray,
    n_samples: int | None = DEFAULT_N_SAMPLES,
):
    components = np.random.choice(len(weights), size=n_samples, p=weights)
    samples = np.zeros(n_samples)
    for component_idx in range(len(weights)):
        mask = components == component_idx
        n_component_samples = mask.sum()
        if n_component_samples > 0:
            samples[mask] = np.random.normal(
                loc=means[component_idx],
                scale=sds[component_idx],
                size=n_component_samples,
            )
    return samples


def get_log_likelihood(
    data: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    weights: np.ndarray,
):
    log_weights = np.log(weights)
    log_probabilities = log_weights + norm.logpdf(data[:, np.newaxis], means, sds)
    log_likelihood_values = logsumexp(log_probabilities, axis=1)
    return log_likelihood_values


def mixture_probabilities(
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
    cdf_values = norm.cdf(kernel[:, np.newaxis], loc=means, scale=sds)
    return cdf_values @ weights


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


def mann_wald_number_of_bins(data: np.ndarray) -> int:
    """
    Mann, H. B., & Wald, A. (1942). On the choice of the number of class intervals in the application of the chi square test.
    """
    n = len(data)
    # Find c value by solving the integral equation:
    # 1/sqrt(2π) ∫[c,∞] exp(-x²/2)dx = alpha
    # This is equivalent to finding c where P(X > c) = alpha for X ~ N(0,1)
    alpha = 0.05
    c = stats.norm.ppf(1 - alpha)

    k = 4 * np.power(2 * (n - 1) ** 2 / c**2, 1 / 5)
    number_of_bins = int(np.round(k))
    return max(DEFAULT_NR_OF_BINS, number_of_bins)


def max_chi2_number_of_bins(data: np.ndarray) -> int:
    min_count_per_bin = 5
    return len(data) // min_count_per_bin


def kl_div(solution1: np.ndarray, solution2: np.ndarray, n_samples: int = DEFAULT_N_SAMPLES):
    weights1, sds1, means1 = solution_to_params(solution1)
    weights2, sds2, means2 = solution_to_params(solution2)
    samples = generate_gaussian_mixture_data(
        means=means1,
        sds=sds1,
        weights=weights1,
        n_samples=n_samples,
    )
    log_p = get_log_likelihood(data=samples, means=means1, sds=sds1, weights=weights1)
    log_q = get_log_likelihood(data=samples, means=means2, sds=sds2, weights=weights2)
    return np.mean(log_p) - np.mean(log_q)


def js_div(solution1: np.ndarray, solution2: np.ndarray, n_samples: int = DEFAULT_N_SAMPLES):
    weights1, sds1, means1 = solution_to_params(solution1)
    weights2, sds2, means2 = solution_to_params(solution2)
    samples1 = generate_gaussian_mixture_data(
        means=means1,
        sds=sds1,
        weights=weights1,
        n_samples=n_samples,
    )
    log_p1 = get_log_likelihood(data=samples1, means=means1, sds=sds1, weights=weights1)
    log_q1 = get_log_likelihood(data=samples1, means=means2, sds=sds2, weights=weights2)
    log_mix1 = np.logaddexp(log_p1, log_q1) - np.log(2)

    samples2 = generate_gaussian_mixture_data(
        means=means2,
        sds=sds2,
        weights=weights2,
        n_samples=n_samples,
    )
    log_p2 = get_log_likelihood(data=samples2, means=means1, sds=sds1, weights=weights1)
    log_q2 = get_log_likelihood(data=samples2, means=means2, sds=sds2, weights=weights2)
    log_mix2 = np.logaddexp(log_p2, log_q2) - np.log(2)

    return (np.mean(log_p1 - log_mix1) + np.mean(log_q2 - log_mix2)) / 2
