from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..problem import GaussianMixtureProblem

CONVERGENCE_PLOT_STEP_SIZE = 1000


@dataclass
class Solution:
    fitness: float
    genome: np.ndarray
    scaled_genome: np.ndarray | None
    log_likelihood: float | None
    fitness_values: np.ndarray | None


class Solver(Protocol):
    def __call__(
        self,
        problem: GaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        ...
