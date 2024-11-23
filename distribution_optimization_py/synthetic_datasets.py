import numpy as np
from distribution_optimization_py.problem import GaussianMixtureProblem
import pandas as pd
import plotly.express as px

N_SAMPLES_PER_COMPONENT = 1000
MIN_WEIGHT = 0.03
MIN_STD_FRAC = 0.001
MAX_STD_FRAC = 0.1


def sample_weights(n_components: int):
    weights = np.full(n_components, MIN_WEIGHT)
    left_weight = 1 - np.sum(weights)
    weights_surplus = np.random.dirichlet(np.ones(n_components))
    weights += left_weight * weights_surplus
    return weights / np.sum(weights)


def generate_gaussian_mixture_data(
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
    n_samples: int | None = 10000,
):
    assert (
        len(means) == len(stds) == len(weights)
    ), "All parameter lists must have the same length."

    weights = np.array(weights)
    weights /= weights.sum()

    n_components = len(means)

    component_samples = (weights * n_samples).astype(int)

    data = []
    for i in range(n_components):
        component_data = np.random.normal(means[i], stds[i], component_samples[i])
        data.extend(component_data)

    return np.array(data)


def generate_mixture_parameters(
    n_mixtures: int,
    n_components: int,
    mean_range: tuple[float, float],
):
    mixture_parameters = []
    generated_datasets = []

    for _ in range(n_mixtures):
        while True:
            weights = sample_weights(n_components)
            means = np.random.uniform(mean_range[0], mean_range[1], n_components)
            estimated_data_range = (mean_range[1] - mean_range[0]) + 2 * (
                mean_range[1] - mean_range[0]
            ) * MAX_STD_FRAC
            stds = np.random.uniform(
                MIN_STD_FRAC * estimated_data_range,
                MAX_STD_FRAC * estimated_data_range,
                n_components,
            )
            means_order = np.argsort(means)
            means = means[means_order]
            stds = stds[means_order]
            weights = weights[means_order]
            if weights.min() < 0.03:
                print("weights too small")
                continue
            generated_data = generate_gaussian_mixture_data(
                means, stds, weights, n_samples=n_components * N_SAMPLES_PER_COMPONENT
            )
            data_range = np.abs(np.max(generated_data) - np.min(generated_data))
            if stds.max() > 0.1 * data_range:
                print("stds too large")
                continue
            if stds.min() < 0.001 * data_range:
                print("stds too small")
                continue
            problem = GaussianMixtureProblem(generated_data, n_components)
            overlap = problem.overlap_error_by_density(means, stds, weights)
            if overlap > 0.5:
                print("overlap too large")
                continue
            yield np.concatenate([weights, stds, means]), generated_data
            break

    return mixture_parameters, generated_datasets
