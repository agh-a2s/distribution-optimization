from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from ..problem import ScaledGaussianMixtureProblem


@dataclass
class Solution:
    fitness: float
    genome: np.ndarray
    scaled_genome: np.ndarray


class Solver(Protocol):
    def __call__(
        self,
        problem: ScaledGaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        ...


def read_dataset_from_csv(path: str, column: str | None = "value") -> np.ndarray:
    return pd.read_csv(path)[column].values


@dataclass
class Dataset:
    data: np.ndarray
    name: str
    nr_of_modes: int


DATASETS = [
    Dataset(
        data=read_dataset_from_csv("../data/truck_driving_data.csv"),
        name="truck_driving_data",
        nr_of_modes=3,
    ),
    Dataset(
        data=read_dataset_from_csv("../data/mixture3.csv"),
        name="mixture3",
        nr_of_modes=3,
    ),
    Dataset(
        data=read_dataset_from_csv("../data/textbook_1k.csv"),
        name="textbook_data",
        nr_of_modes=3,
    ),
    Dataset(
        data=read_dataset_from_csv("../data/iris_ica.csv"),
        name="iris_ica",
        nr_of_modes=3,
    ),
    Dataset(
        data=read_dataset_from_csv("../data/chromatogram_time.csv"),
        name="chromatogram_time",
        nr_of_modes=5,
    ),
    Dataset(
        data=read_dataset_from_csv("../data/atmosphere_data.csv"),
        name="atmosphere_data",
        nr_of_modes=5,
    ),
]
