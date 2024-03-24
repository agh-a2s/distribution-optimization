from typing import Type

import matplotlib.pyplot as plt
import numpy as np

from .problem import ScaledGaussianMixtureProblem
from .solver import CMAESSolver, DESolver, GASolver, HMSSolver, Solver
from .utils import mixture_probability

SOLVER_NAME_TO_CLASS: dict[str, Type[Solver]] = {
    "GA": GASolver,
    "CMA-ES": CMAESSolver,
    "HMS": HMSSolver,
    "DE": DESolver,
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
        self._X = X
        self._weights = scaled_solution[: self._n_components]
        self._sds = scaled_solution[self._n_components : 2 * self._n_components]
        self._means = scaled_solution[2 * self._n_components :]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = [mixture_probability(x, self._means, self._sds, self._weights) for x in X]
        return np.array(probabilities)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        likelihood = np.array([mixture_probability(x, self._means, self._sds, self._weights, False) for x in X])
        return np.sum(likelihood, axis=1)

    def plot(self, num: int | None = 1000, bins: int | None = 30) -> None:
        if self._weights is None:
            raise ValueError("Model has not been fitted yet")

        x = np.linspace(self._X.min(), self._X.max(), num)
        pdf = self.score_samples(x)

        plt.hist(
            self._X,
            bins=bins,
            density=True,
            alpha=0.6,
            color="g",
            label="Empirical Data",
        )
        plt.plot(x, pdf, "-r", label="GMM PDF")
        plt.xlabel("Data Values")
        plt.ylabel("Probability Density")
        plt.legend(loc="upper left")
        plt.title("Histogram and GMM PDF")
        plt.show()

    def set_params(self, X: np.ndarray, solution: np.ndarray) -> "GaussianMixture":
        self._X = X
        self._weights = solution[: self._n_components]
        self._sds = solution[self._n_components : 2 * self._n_components]
        self._means = solution[2 * self._n_components :]
        return self

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


def compare_solutions(
    X: np.ndarray,
    nr_of_modes: int,
    solution1: np.ndarray,
    solution2: np.ndarray,
    label1: str,
    label2: str,
    num: int | None = 1000,
    bins: int | None = 30,
    title: str | None = None,
) -> None:
    gmm1 = GaussianMixture(n_components=nr_of_modes, random_state=1).set_params(X, solution1)
    gmm2 = GaussianMixture(n_components=nr_of_modes, random_state=1).set_params(X, solution2)
    x = np.linspace(X.min(), X.max(), num)
    pdf1 = gmm1.score_samples(x)
    pdf2 = gmm2.score_samples(x)
    pdf1 = pdf1 / pdf1.sum() * 100
    pdf2 = pdf2 / pdf2.sum() * 100
    plt.hist(
        X,
        bins=bins,
        density=True,
        alpha=0.6,
        color="g",
        label="Empirical Data",
    )
    plt.plot(x, pdf1, "--", label=label1, color="r")
    plt.plot(x, pdf2, "--", label=label2, color="b")
    plt.legend(loc="upper left")
    if title:
        plt.title(title)
    plt.show()
