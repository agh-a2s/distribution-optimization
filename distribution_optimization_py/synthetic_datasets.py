import random

import numpy as np
import pandas as pd
import plotly.express as px
from distribution_optimization_py.problem import GaussianMixtureProblem

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
    assert len(means) == len(stds) == len(weights), "All parameter lists must have the same length."

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
    min_overlap_value: float = 0.0,
    max_overlap_value: float = 0.5,
    log: bool = False,
):
    """
    This function is responsible for generating mixture parameters for GMMs.
    The idea is to generate a set of parameters that are feasible for the problem.
    By feasible, we mean that the parameters are in the domain of the DistributionOptimization problem.
    This function is implementing a rejection sampling algorithm, which tries to improve
    efficiency of sampling by generating parameters that are more likely to be feasible.
    The idea is to generate first and last standard deviations and use them to estimate the range of the data.
    This way the likelihood of generating parameters that are not feasible is reduced.
    At the same time it should not introduce a bias in the sampling.
    """
    mixture_parameters = []
    generated_datasets = []

    for idx in range(n_mixtures):
        random.seed(idx)
        np.random.seed(idx)
        while True:
            weights = sample_weights(n_components)
            means = np.random.uniform(mean_range[0], mean_range[1], n_components)
            estimated_data_range = (mean_range[1] - mean_range[0]) + 2 * (mean_range[1] - mean_range[0]) * MAX_STD_FRAC
            first_std = np.random.uniform(
                MIN_STD_FRAC * estimated_data_range,
                MAX_STD_FRAC * estimated_data_range,
            )
            last_std = np.random.uniform(
                MIN_STD_FRAC * estimated_data_range,
                MAX_STD_FRAC * estimated_data_range,
            )
            estimated_data_range = (mean_range[1] - mean_range[0]) + 2 * (first_std + last_std)
            stds = np.random.uniform(
                MIN_STD_FRAC * estimated_data_range,
                MAX_STD_FRAC * estimated_data_range,
                n_components - 2,
            )
            stds = np.concatenate([[first_std], stds, [last_std]])
            means_order = np.argsort(means)
            means = means[means_order]
            stds = stds[means_order]
            weights = weights[means_order]
            # TODO: replace magic numbers with a constant/method in Problem class
            if weights.min() < 0.03:
                if log:
                    print("weights too small")
                continue
            generated_data = generate_gaussian_mixture_data(
                means, stds, weights, n_samples=n_components * N_SAMPLES_PER_COMPONENT
            )
            data_range = np.abs(np.max(generated_data) - np.min(generated_data))
            if stds.max() > 0.1 * data_range:
                if log:
                    print("stds too large")
                continue
            if stds.min() < 0.001 * data_range:
                if log:
                    print("stds too small")
                continue
            problem = GaussianMixtureProblem(generated_data, n_components)
            overlap = problem.overlap_error_by_density(means, stds, weights)
            if overlap > max_overlap_value:
                if log:
                    print("overlap too large")
                continue
            if overlap < min_overlap_value:
                if log:
                    print("overlap too small")
                continue
            yield np.concatenate([weights, stds, means]), generated_data
            break

    return mixture_parameters, generated_datasets


def generate_difficult_mixture_parameters(
    n_mixtures: int,
):
    for idx in range(n_mixtures):
        while True:
            random.seed(idx)
            np.random.seed(idx)
            means = [0.0, 1.0]
            # At least one std dev should be very small
            stds = np.random.uniform(0.001, 0.005, 2)
            weights = sample_weights(2)
            generated_data = generate_gaussian_mixture_data(means, stds, weights, n_samples=2 * N_SAMPLES_PER_COMPONENT)
            parameters = np.concatenate([weights, stds, means])
            problem = GaussianMixtureProblem(generated_data, 2)
            lower, upper = problem.get_bounds()
            if np.any(parameters < lower) or np.any(parameters > upper):
                continue
            yield parameters, generated_data
            break
