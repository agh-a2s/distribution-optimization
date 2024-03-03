from typing import Type

import numpy as np

from .problem import ScaledGaussianMixtureProblem
from .solver import CMAESSolver, GASolver, HMSSolver, Solver
from .utils import mixture_probability

SOLVER_NAME_TO_CLASS: dict[str, Type[Solver]] = {
    "GA": GASolver,
    "CMA-ES": CMAESSolver,
    "HMS": HMSSolver,
}


class GaussianMixture:
    def __init__(
        self,
        n_components: int = 1,
        max_n_evals: int = 10000,
        random_state: int | None = None,
        algorithm: str = "HMS",
    ):
        self._n_components: int = n_components
        self._max_n_evals: int = max_n_evals
        self._random_state: int | None = random_state
        self._algorithm: str = self._validate_algorithm(algorithm)
        self._solver = SOLVER_NAME_TO_CLASS[self._algorithm]()
        # Solution:
        self._weights: np.ndarray | None = None
        self._sds: np.ndarray | None = None
        self._means: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "GaussianMixture":
        X = self._validate_data(X)
        problem = ScaledGaussianMixtureProblem(X, self._n_components)
        solution = self._solver(problem, self._max_n_evals, self._random_state)
        scaled_solution = problem.reals_to_internal(solution)
        self._weights = scaled_solution[: self._n_components]
        self._sds = scaled_solution[self._n_components : 2 * self._n_components]
        self._means = scaled_solution[2 * self._n_components :]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = [
            mixture_probability(x, self._means, self._sds, self._weights) for x in X
        ]
        return np.array(probabilities)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if X.ndim != 1:
            raise ValueError("X must be a 1D array")
        return X

    def _validate_algorithm(self, algorithm: str) -> str:
        if algorithm not in SOLVER_NAME_TO_CLASS:
            raise ValueError(f"'algorithm' must be one of {list(SOLVER_NAME_TO_CLASS)}")
        return algorithm
