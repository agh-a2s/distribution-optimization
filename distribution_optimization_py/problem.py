import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

from .scale import (
    full_simplex_to_reals,
    reals_to_full_simplex,
    reals_to_reals_with_offset,
    reals_to_simplex,
    reals_with_offset_to_reals,
    scale_linearly,
    scale_uniformly_simplex,
    simplex_to_reals,
    unscale_uniformly_simplex,
)
from .utils import bin_prob_for_mixtures, optimal_no_bins

INFINITY = 1000000  # TODO: use np.inf instead of 1000000
DEFAULT_NR_OF_KERNELS = 40
DEFAULT_OVERLAP_TOLERANCE = 0.5
DEFAULT_LOWER = -5
DEFAULT_UPPER = 5


def initialize_parameters(X: np.ndarray, nr_of_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = KMeans(n_clusters=nr_of_modes).fit(X.reshape(-1, 1)).labels_
    means = np.array([X[labels == i].mean() for i in range(nr_of_modes)])
    stds = np.array([X[labels == i].std() for i in range(nr_of_modes)])
    weights = np.array([np.mean(labels == i) for i in range(nr_of_modes)])
    return weights, stds, means


class GaussianMixtureProblem:
    def __init__(
        self,
        data: np.ndarray,
        nr_of_modes: int,
        nr_of_kernels: int | None = DEFAULT_NR_OF_KERNELS,
        overlap_tolerance: float | None = DEFAULT_OVERLAP_TOLERANCE,
    ):
        self.data = data
        self.nr_of_modes = nr_of_modes
        self.N = len(data)
        self.nr_of_bins = optimal_no_bins(data)
        self.breaks = np.linspace(np.min(data), np.max(data), self.nr_of_bins + 1)
        self.observed_bins, _ = np.histogram(data, self.breaks)
        self.nr_of_kernels = nr_of_kernels  # Kernels are used to estimate the overlap error.
        self.overlap_tolerance = overlap_tolerance
        self.data_lower, self.data_upper = self.get_bounds()

    def __call__(self, x: np.ndarray) -> float:
        weights = x[: self.nr_of_modes]
        sds = x[self.nr_of_modes : 2 * self.nr_of_modes]
        means = x[2 * self.nr_of_modes :]
        overlap_error = self.overlap_error_by_density(means, sds, weights)
        if overlap_error > self.overlap_tolerance:
            return INFINITY
        similarity_error = self.similarity_error(means, sds, weights)
        return similarity_error

    def overlap_error_by_density(self, means: np.ndarray, sds: np.ndarray, weights: np.ndarray) -> float:
        kernels = np.linspace(np.min(self.data), np.max(self.data), self.nr_of_kernels)
        densities = np.array([norm.pdf(kernels, loc=m, scale=sd) * w for m, sd, w in zip(means, sds, weights)])

        overlap_in_component = np.zeros_like(densities)
        for i in range(len(means)):
            max_other_modes = np.max(np.delete(densities, i, axis=0), axis=0)
            overlap_in_component[i] = np.minimum(densities[i], max_other_modes)

        area_in_component = np.sum(densities, axis=1)
        overlap_in_components = np.sum(overlap_in_component, axis=1)
        ov_ratio_in_component = overlap_in_components / area_in_component

        error_overlap_component = np.max(ov_ratio_in_component)

        return error_overlap_component

    def similarity_error(self, means: np.ndarray, sds: np.ndarray, weights: np.ndarray) -> float:
        estimated_bins = bin_prob_for_mixtures(means, sds, weights, self.breaks) * self.N
        norm = estimated_bins.copy()
        norm[norm < 1] = 1
        diffssq = np.power((self.observed_bins - estimated_bins), 2)
        diffssq[diffssq < 4] = 0
        return np.sum(diffssq / norm) / self.N

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        data_range = np.abs(np.max(self.data) - np.min(self.data))
        weights_lower = [0.03] * self.nr_of_modes
        weights_upper = [1.00] * self.nr_of_modes
        sds_lower = [0.001 * data_range] * self.nr_of_modes
        sds_upper = [0.1 * data_range] * self.nr_of_modes
        means_lower = [np.min(self.data)] * self.nr_of_modes
        means_upper = [np.max(self.data)] * self.nr_of_modes
        lower = np.concatenate([weights_lower, sds_lower, means_lower])
        upper = np.concatenate([weights_upper, sds_upper, means_upper])
        return lower, upper

    def initialize(
        self,
    ) -> np.ndarray:
        weights = np.random.uniform(
            self.data_lower[: self.nr_of_modes],
            self.data_upper[: self.nr_of_modes],
            size=self.nr_of_modes,
        )
        sds = np.random.uniform(
            0.0,
            (np.max(self.data) - np.min(self.data)) / (self.nr_of_modes * 3),
            size=self.nr_of_modes,
        )
        breaks = np.linspace(np.min(self.data), np.max(self.data), self.nr_of_modes + 2)[1:-1]
        means = np.random.normal(
            breaks,
            (np.max(self.data) - np.min(self.data)) / (self.nr_of_modes * 5),
            size=self.nr_of_modes,
        )
        weights = weights / np.sum(weights)
        x = np.concatenate([weights, sds, means])
        x = np.minimum(np.maximum(x, self.data_lower), self.data_upper)
        return x

    def initialize_warm_start(self) -> np.ndarray:
        weights, sds, means = initialize_parameters(self.data, self.nr_of_modes)
        means_order = np.argsort(means)
        weights = weights[means_order]
        sds = sds[means_order]
        means = means[means_order]
        x = np.concatenate([weights, sds, means])
        return np.minimum(np.maximum(x, self.data_lower), self.data_upper)


class LinearlyScaledGaussianMixtureProblem(GaussianMixtureProblem):
    def __init__(
        self,
        data: np.ndarray,
        nr_of_modes: int,
        lower: float | None = DEFAULT_LOWER,
        upper: float | None = DEFAULT_UPPER,
        nr_of_kernels: int | None = DEFAULT_NR_OF_KERNELS,
        overlap_tolerance: float | None = DEFAULT_OVERLAP_TOLERANCE,
    ):
        super().__init__(data, nr_of_modes, nr_of_kernels, overlap_tolerance)
        self.lower = np.array([lower] * self.nr_of_modes * 3)
        self.upper = np.array([upper] * self.nr_of_modes * 3)

    def __call__(self, x: np.ndarray) -> float:
        if self.lower is not None and self.upper is not None:
            x = scale_linearly(x, self.lower, self.upper, self.data_lower, self.data_upper)
        fixed_x = self.fix(x)
        return super().__call__(fixed_x)

    def fix(self, x: np.ndarray) -> np.ndarray:
        x[: self.nr_of_modes] = x[: self.nr_of_modes] / np.sum(x[: self.nr_of_modes])
        means_order = np.argsort(x[2 * self.nr_of_modes :])
        x[: self.nr_of_modes] = x[: self.nr_of_modes][means_order]
        x[self.nr_of_modes : 2 * self.nr_of_modes] = x[self.nr_of_modes : 2 * self.nr_of_modes][means_order]
        x[2 * self.nr_of_modes :] = x[2 * self.nr_of_modes :][means_order]
        return x

    def fix_and_scale(self, x: np.ndarray) -> np.ndarray:
        if self.lower is not None and self.upper is not None:
            x = scale_linearly(x, self.lower, self.upper, self.data_lower, self.data_upper)
        x = self.fix(x)
        if self.lower is not None and self.upper is not None:
            x = scale_linearly(x, self.data_lower, self.data_upper, self.lower, self.upper)
        x = np.clip(x, self.lower, self.upper)
        return x

    def scale_to_data_bounds(self, x: np.ndarray) -> np.ndarray:
        scaled_x = scale_linearly(x, self.lower, self.upper, self.data_lower, self.data_upper)
        return self.fix(scaled_x)

    def initialize(self) -> np.ndarray:
        x = super().initialize()
        fixed_x = self.fix(x)
        scaled_x = scale_linearly(fixed_x, self.data_lower, self.data_upper, self.lower, self.upper)
        scaled_x = np.minimum(np.maximum(scaled_x, self.lower), self.upper)
        return scaled_x


class ScaledGaussianMixtureProblem(GaussianMixtureProblem):
    def __init__(
        self,
        data: np.ndarray,
        nr_of_modes: int,
        nr_of_kernels: int | None = DEFAULT_NR_OF_KERNELS,
        overlap_tolerance: float | None = DEFAULT_OVERLAP_TOLERANCE,
    ):
        super().__init__(data, nr_of_modes, nr_of_kernels, overlap_tolerance)
        self.lower = np.array([0.0] * (self.nr_of_modes * 3 - 1))
        self.upper = np.array([1.0] * (self.nr_of_modes * 3 - 1))

    def __call__(self, x: np.ndarray) -> float:
        internal_x = self.reals_to_internal(x)
        return super().__call__(internal_x)

    def initialize(self) -> np.ndarray:
        internal_x = super().initialize()
        return self.internal_to_reals(internal_x)

    def initialize_warm_start(self) -> np.ndarray:
        internal_x = super().initialize_warm_start()
        return self.internal_to_reals(internal_x)

    def reals_to_internal(self, x: np.ndarray) -> np.ndarray:
        # Scale weights to simplex:
        weights = x[: self.nr_of_modes - 1]
        internal_weights = reals_to_simplex(weights)
        # Scale sds linearly:
        sds = x[self.nr_of_modes - 1 : 2 * self.nr_of_modes - 1]
        internal_sds = scale_linearly(
            sds,
            self.lower[self.nr_of_modes - 1 : 2 * self.nr_of_modes - 1],
            self.upper[self.nr_of_modes - 1 : 2 * self.nr_of_modes - 1],
            self.data_lower[self.nr_of_modes : 2 * self.nr_of_modes],
            self.data_upper[self.nr_of_modes : 2 * self.nr_of_modes],
        )
        # Scale means using offset and full simplex:
        means = x[2 * self.nr_of_modes - 1 :]
        internal_means = scale_linearly(
            reals_to_reals_with_offset(reals_to_full_simplex(scale_uniformly_simplex(means))),
            self.lower[2 * self.nr_of_modes - 1 :],
            self.upper[2 * self.nr_of_modes - 1 :],
            self.data_lower[2 * self.nr_of_modes :],
            self.data_upper[2 * self.nr_of_modes :],
        )
        return np.concatenate([internal_weights, internal_sds, internal_means])

    def internal_to_reals(self, x: np.ndarray) -> np.ndarray:
        # Scale weights from simplex:
        internal_weights = x[: self.nr_of_modes]
        weights = simplex_to_reals(internal_weights)
        # Scale sds linearly:
        internal_sds = x[self.nr_of_modes : 2 * self.nr_of_modes]
        sds = scale_linearly(
            internal_sds,
            self.data_lower[self.nr_of_modes : 2 * self.nr_of_modes],
            self.data_upper[self.nr_of_modes : 2 * self.nr_of_modes],
            self.lower[self.nr_of_modes - 1 : 2 * self.nr_of_modes - 1],
            self.upper[self.nr_of_modes - 1 : 2 * self.nr_of_modes - 1],
        )
        # Scale means using offset and full simplex:
        internal_means = x[2 * self.nr_of_modes :]
        means = scale_linearly(
            internal_means,
            self.data_lower[2 * self.nr_of_modes :],
            self.data_upper[2 * self.nr_of_modes :],
            self.lower[2 * self.nr_of_modes - 1 :],
            self.upper[2 * self.nr_of_modes - 1 :],
        )
        means = unscale_uniformly_simplex(full_simplex_to_reals(reals_with_offset_to_reals(means)))
        return np.concatenate([weights, sds, means])


BBOB_LOWER = -5
BBOB_UPPER = 5


class ScaledGaussianMixtureProblemForELA(ScaledGaussianMixtureProblem):
    def __init__(
        self,
        data: np.ndarray,
        nr_of_modes: int,
        nr_of_kernels: int | None = DEFAULT_NR_OF_KERNELS,
        overlap_tolerance: float | None = DEFAULT_OVERLAP_TOLERANCE,
    ):
        super().__init__(data, nr_of_modes, nr_of_kernels, overlap_tolerance)
        self.base_lower = np.array([0.0] * (self.nr_of_modes * 3 - 1))
        self.base_upper = np.array([1.0] * (self.nr_of_modes * 3 - 1))
        self.lower = np.array([BBOB_LOWER] * (self.nr_of_modes * 3 - 1))
        self.upper = np.array([BBOB_UPPER] * (self.nr_of_modes * 3 - 1))

    def reals_to_internal(self, x: np.ndarray) -> np.ndarray:
        # Scale weights to simplex:
        weights = x[: self.nr_of_modes - 1]
        weights = scale_linearly(
            weights,
            self.lower[: self.nr_of_modes - 1],
            self.upper[: self.nr_of_modes - 1],
            self.base_lower[: self.nr_of_modes - 1],
            self.base_upper[: self.nr_of_modes - 1],
        )
        internal_weights = reals_to_simplex(weights)
        # Scale sds linearly:
        sds = x[self.nr_of_modes - 1 : 2 * self.nr_of_modes - 1]
        internal_sds = scale_linearly(
            sds,
            self.lower[self.nr_of_modes - 1 : 2 * self.nr_of_modes - 1],
            self.upper[self.nr_of_modes - 1 : 2 * self.nr_of_modes - 1],
            self.data_lower[self.nr_of_modes : 2 * self.nr_of_modes],
            self.data_upper[self.nr_of_modes : 2 * self.nr_of_modes],
        )
        # Scale means using offset and full simplex:
        means = x[2 * self.nr_of_modes - 1 :]
        means = scale_linearly(
            means,
            self.lower[2 * self.nr_of_modes - 1 :],
            self.upper[2 * self.nr_of_modes - 1 :],
            self.base_lower[2 * self.nr_of_modes - 1 :],
            self.base_upper[2 * self.nr_of_modes - 1 :],
        )
        internal_means = scale_linearly(
            reals_to_reals_with_offset(reals_to_full_simplex(scale_uniformly_simplex(means))),
            self.base_lower[2 * self.nr_of_modes - 1 :],
            self.base_upper[2 * self.nr_of_modes - 1 :],
            self.data_lower[2 * self.nr_of_modes :],
            self.data_upper[2 * self.nr_of_modes :],
        )
        return np.concatenate([internal_weights, internal_sds, internal_means])

    def internal_to_reals(self, x: np.ndarray) -> np.ndarray:
        # Scale weights from simplex:
        internal_weights = x[: self.nr_of_modes]
        weights = simplex_to_reals(internal_weights)
        weights = scale_linearly(
            weights,
            self.base_lower[: self.nr_of_modes],
            self.base_upper[: self.nr_of_modes],
            self.lower[: self.nr_of_modes],
            self.upper[: self.nr_of_modes],
        )
        # Scale sds linearly:
        internal_sds = x[self.nr_of_modes : 2 * self.nr_of_modes]
        sds = scale_linearly(
            internal_sds,
            self.data_lower[self.nr_of_modes : 2 * self.nr_of_modes],
            self.data_upper[self.nr_of_modes : 2 * self.nr_of_modes],
            self.lower[self.nr_of_modes - 1 : 2 * self.nr_of_modes - 1],
            self.upper[self.nr_of_modes - 1 : 2 * self.nr_of_modes - 1],
        )
        # Scale means using offset and full simplex:
        internal_means = x[2 * self.nr_of_modes :]
        means = scale_linearly(
            internal_means,
            self.data_lower[2 * self.nr_of_modes :],
            self.data_upper[2 * self.nr_of_modes :],
            self.base_lower[2 * self.nr_of_modes - 1 :],
            self.base_upper[2 * self.nr_of_modes - 1 :],
        )
        means = unscale_uniformly_simplex(full_simplex_to_reals(reals_with_offset_to_reals(means)))
        means = scale_linearly(
            means,
            self.base_lower[2 * self.nr_of_modes - 1 :],
            self.base_upper[2 * self.nr_of_modes - 1 :],
            self.lower[2 * self.nr_of_modes - 1 :],
            self.upper[2 * self.nr_of_modes - 1 :],
        )
        return np.concatenate([weights, sds, means])
