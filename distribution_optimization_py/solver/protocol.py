from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from ..problem import ScaledGaussianMixtureProblem

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
        problem: ScaledGaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        ...
