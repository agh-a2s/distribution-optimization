from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .datasets import Dataset
from .problem import GaussianMixtureProblem
from .solver import DESolver, GASolver, HMSSolver, Solver
from .utils import mixture_probabilities

SOLVER_NAME_TO_CLASS: dict[str, Type[Solver]] = {
    "GA": GASolver,
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
        problem = GaussianMixtureProblem(X, self._n_components)
        solution = self._solver(problem, self._max_n_evals, self._random_state)
        self._X = X
        self._weights = solution.genome[: self._n_components]
        self._sds = solution.genome[self._n_components : 2 * self._n_components]
        self._means = solution.genome[2 * self._n_components :]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = [mixture_probabilities(x, self._means, self._sds, self._weights) for x in X]
        return np.array(probabilities)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        likelihood = np.array([mixture_probabilities(x, self._means, self._sds, self._weights, False) for x in X])
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
) -> None:
    gmm1 = GaussianMixture(n_components=nr_of_modes, random_state=1).set_params(X, solution1)
    gmm2 = GaussianMixture(n_components=nr_of_modes, random_state=1).set_params(X, solution2)
    x = np.linspace(X.min(), X.max(), num)
    pdf1 = gmm1.score_samples(x)
    pdf2 = gmm2.score_samples(x)
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
    plt.title("Histogram and GMM PDF")
    plt.show()


def compare_solutions_plotly(
    X: np.ndarray,
    nr_of_modes: int,
    solutions: list[np.ndarray],
    labels: list[str],
    num: int | None = 1000,
    bins: int | None = 30,
    title: str | None = "Comparison of GMM Fits to 1D Data",
) -> None:
    assert len(solutions) == len(labels)
    gmms = [GaussianMixture(n_components=nr_of_modes, random_state=1).set_params(X, solution) for solution in solutions]

    x = np.linspace(X.min(), X.max(), num)
    pdfs = [gmm.score_samples(x) for gmm in gmms]
    hist_data = np.histogram(X, bins=bins, density=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=hist_data[1][:-1],
            y=hist_data[0],
            opacity=0.6,
            name="Empirical Data",
        )
    )

    for pdf, label in zip(pdfs, labels):
        fig.add_trace(
            go.Scatter(
                x=x.flatten(),
                y=pdf,
                mode="lines",
                name=label,
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Density",
        legend=dict(y=0.95),
        barmode="overlay",
        width=1200,
        height=800,
    )

    fig.show()


def compare_solutions_for_datasets_plotly(
    datasets: list[Dataset],
    dataset_name_to_label_to_solution: dict[str, dict[str, np.ndarray]],
    color_map: dict[str, str],
    num: int | None = 1000,
    bins: int | None = 30,
    cols: int = 1,
    height: int = 2400,
    width: int = 1000,
) -> None:
    fig = make_subplots(
        rows=len(datasets) // cols,
        cols=cols,
        subplot_titles=[dataset.name for dataset in datasets],
    )
    for dataset_idx, dataset in enumerate(datasets):
        row = dataset_idx // cols + 1
        col = dataset_idx % cols + 1
        label_to_solution = dataset_name_to_label_to_solution[dataset.name]
        labels = list(label_to_solution.keys())
        solutions = list(label_to_solution.values())
        gmms = [
            GaussianMixture(n_components=dataset.nr_of_modes, random_state=1).set_params(dataset.data, solution)
            for solution in solutions
        ]
        x = np.linspace(dataset.data.min(), dataset.data.max(), num)
        pdfs = [gmm.score_samples(x) for gmm in gmms]
        hist_data = np.histogram(dataset.data, bins=bins, density=True)
        fig.add_trace(
            go.Bar(
                x=hist_data[1][:-1],
                y=hist_data[0],
                opacity=0.3,
                name="Empirical Data",
                legendgroup="Empirical Data",
                showlegend=(dataset_idx == 0),
                marker=dict(color=color_map["Empirical Data"]),
            ),
            row=row,
            col=col,
        )

        for pdf, label in zip(pdfs, labels):
            fig.add_trace(
                go.Scatter(
                    x=x.flatten(),
                    y=pdf,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=(dataset_idx == 0),
                    line=dict(dash="dash", color=color_map[label]),
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title="Histogram and GMM PDF",
        xaxis_title="X",
        yaxis_title="Density",
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        height=height,
        width=width,
    )

    fig.show()
