from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np
from scipy.stats import norm

from .problem import DEFAULT_NR_OF_KERNELS, DEFAULT_OVERLAP_TOLERANCE, INFINITY
from .utils import bin_prob_for_mixtures, optimal_no_bins


class ConstraintHandlingTechnique(StrEnum):
    DEATH_PENALTY = "death_penalty"
    WEIGHTED_PENALTY = "weighted_penalty"
    FEASIBILITY_RANKING = "feasibility_ranking"
    EPS_FEASIBILITY_RANKING = "eps_feasibility_ranking"
    BIOBJECTIVE = "biobjective"


class ConstrainedProblem(ABC):
    def __init__(
        self,
        bounds: np.ndarray,
        method: ConstraintHandlingTechnique = ConstraintHandlingTechnique.DEATH_PENALTY,
    ) -> None:
        self.dim = bounds.shape[0]
        self.eps = 1e-4
        self.bounds = bounds
        self.method = method

    @abstractmethod
    def fitness(self, xs: np.ndarray) -> np.ndarray:
        """
        Compute the fitness value for the given input x.

        :param x: A 2D numpy array - solutions.
        :return: A 1D numpy array with fitness values.
        """
        pass

    @abstractmethod
    def constraints(self, xs: np.ndarray) -> np.ndarray:
        """
        Compute the constraints for the given input x.

        :param x: A 2D numpy array - a row is a solution.
        :return: A 2D numpy array where each row contains the constraint values [g1, g2, ..., gN].
        """
        pass

    def violation(self, xs: np.ndarray) -> np.ndarray:
        g = self.constraints(xs)
        num_constraints = g.shape[1]
        return np.sum(np.maximum(0, g), axis=1) / num_constraints

    def _adjust_equality_constraint(self, h: np.ndarray) -> np.ndarray:
        # Convert equality constraint to inequality constraint:
        # h(x) = 0 -> |h(x)| - eps <= 0
        h_adjusted = np.abs(h) - self.eps
        return h_adjusted

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        xs = np.array([x])
        fitness = self.fitness(xs)[0]
        violation = self.violation(xs)[0]
        if self.method == ConstraintHandlingTechnique.DEATH_PENALTY:
            return INFINITY if violation > 0 else fitness
        elif self.method == ConstraintHandlingTechnique.WEIGHTED_PENALTY:
            return fitness + violation
        else:
            return np.array([fitness, violation])


class DistributionOptimizationProblem(ConstrainedProblem):
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
        bounds = self.get_bounds()
        self.data_lower, self.data_upper = bounds[:, 0], bounds[:, 1]
        super().__init__(bounds)

    def fitness(self, xs: np.ndarray) -> np.ndarray:
        return np.array([self.similarity_error(x) for x in xs])

    def constraints(self, xs: np.ndarray) -> np.ndarray:
        # TODO: Test if it's correct
        g_overlap = self.overlap_constraint(xs)
        h_weights = self.weights_constraint(xs)
        g_weights = self._adjust_equality_constraint(h_weights)
        g_means = []
        for i in range(self.nr_of_modes - 1):
            g_means.append(xs[:, 2 * self.nr_of_modes + i] - xs[:, 2 * self.nr_of_modes + i + 1])
        return np.column_stack([g_overlap, g_weights] + g_means)

    def overlap_constraint(self, xs: np.ndarray) -> float:
        overlap_errors = []
        for x in xs:
            weights = x[: self.nr_of_modes]
            sds = x[self.nr_of_modes : 2 * self.nr_of_modes]
            means = x[2 * self.nr_of_modes :]
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
            overlap_errors.append(error_overlap_component)
        return self.overlap_tolerance - np.array(overlap_errors)

    def weights_constraint(self, xs: np.ndarray) -> np.ndarray:
        weights = xs[:, : self.nr_of_modes]
        return np.sum(weights, axis=1) - 1

    def similarity_error(self, x: np.ndarray) -> np.ndarray:
        weights = x[: self.nr_of_modes]
        sds = x[self.nr_of_modes : 2 * self.nr_of_modes]
        means = x[2 * self.nr_of_modes :]
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
        return np.column_stack([lower, upper])

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
