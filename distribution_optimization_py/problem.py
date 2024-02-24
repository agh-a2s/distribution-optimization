import numpy as np
from scipy.stats import norm

from .scale import scale_linearly as scale
from .utils import bin_prob_for_mixtures, optimal_no_bins

INFINITY = 1000000
DEFAULT_NR_OF_KERNELS = 40
DEFAULT_OVERLAP_TOLERANCE = 0.5


class GaussianMixtureProblem:
    def __init__(
        self,
        data: np.ndarray,
        nr_of_modes: int,
        lower: float | None = None,
        upper: float | None = None,
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
        if lower is not None and upper is not None:
            self.lower = np.array([lower] * self.nr_of_modes * 3)
            self.upper = np.array([upper] * self.nr_of_modes * 3)
        else:
            self.lower = None
            self.upper = None

    def __call__(self, x: np.ndarray) -> float:
        if self.lower is not None and self.upper is not None:
            x = scale(x, self.lower, self.upper, self.data_lower, self.data_upper)
        x = self.fix(x)
        weights = x[: self.nr_of_modes]
        sds = x[self.nr_of_modes : 2 * self.nr_of_modes]
        means = x[2 * self.nr_of_modes :]
        overlap_error = self.overlap_error_by_density(means, sds, weights)
        if overlap_error > self.overlap_tolerance:
            # return np.Inf
            return INFINITY
        similarity_error = self.similarity_error(means, sds, weights)
        return similarity_error

    def fix(self, x: np.ndarray) -> np.ndarray:
        x[: self.nr_of_modes] = x[: self.nr_of_modes] / np.sum(x[: self.nr_of_modes])
        means_order = np.argsort(x[2 * self.nr_of_modes :])
        x[: self.nr_of_modes] = x[: self.nr_of_modes][means_order]
        x[self.nr_of_modes : 2 * self.nr_of_modes] = x[self.nr_of_modes : 2 * self.nr_of_modes][means_order]
        x[2 * self.nr_of_modes :] = x[2 * self.nr_of_modes :][means_order]
        return x

    def fix_and_scale(self, x: np.ndarray) -> np.ndarray:
        if self.lower is not None and self.upper is not None:
            x = scale(x, self.lower, self.upper, self.data_lower, self.data_upper)
        x = self.fix(x)
        if self.lower is not None and self.upper is not None:
            x = scale(x, self.data_lower, self.data_upper, self.lower, self.upper)
        x = np.clip(x, self.lower, self.upper)
        return x

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

    def scale_to_data_bounds(self, x: np.ndarray) -> np.ndarray:
        scaled_x = scale(x, self.lower, self.upper, self.data_lower, self.data_upper)
        return self.fix(scaled_x)

    def initialize_equidistant(self, should_scale: bool | None = True) -> np.ndarray:
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
        fixed_x = self.fix(x)
        if not should_scale:
            return fixed_x
        scaled_x = scale(fixed_x, self.data_lower, self.data_upper, self.lower, self.upper)
        scaled_x = np.minimum(np.maximum(scaled_x, self.lower), self.upper)
        return scaled_x

    def initialize(self) -> np.ndarray:
        return self.initialize_equidistant()
        # p = random.random()
        # if p < 0.7:
        #     return self.initialize_equidistant()
        # else:
        #     return np.random.uniform(self.lower[0], self.upper[0], self.nr_of_modes * 3)

    def initialize_population(self, size: int) -> np.ndarray:
        return np.array([self.initialize() for _ in range(size)])
