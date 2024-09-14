import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .solver.protocol import Solution


def read_dataset_from_csv(path: str, column: str | None = "value") -> np.ndarray:
    return pd.read_csv(path)[column].values


@dataclass(frozen=True)
class Dataset:
    data: np.ndarray
    name: str
    nr_of_modes: int
    solution: Solution | None
    fitness_values: list[float] | None = None


TRUCK_DRIVING_SOLUTION = Solution(
    fitness=0.07063351507584274,
    genome=np.array(
        [
            3.560141580335023370e-01,
            4.722198560131814493e-01,
            1.717659859533161859e-01,
            6.958281561863719844e01,
            7.704599999999967963e01,
            6.270656333478625299e01,
            6.362230966522014342e01,
            3.766797266946350078e02,
            6.024754672926660533e02,
        ]
    ),
    scaled_genome=None,
    log_likelihood=None,
    fitness_values=[],
)

MIXTURE3_SOLUTION = Solution(
    fitness=0.0041164007644249045,
    genome=np.array(
        [
            6.657430814165578303e-02,
            5.022681493592618535e-02,
            8.831988769224181635e-01,
            9.450785615145761431e-01,
            2.708735408762603658e00,
            2.007322877862607768e00,
            -1.011326175933570681e01,
            -1.139966127772719773e00,
            9.970537094038517623e00,
        ]
    ),
    scaled_genome=None,
    log_likelihood=None,
    fitness_values=[],
)

TEXTBOOK_DATA_SOLUTION = Solution(
    fitness=0.009500957132228559,
    genome=np.array(
        [
            1.809489544887743495e-01,
            6.611352073423293341e-01,
            1.579158381688963164e-01,
            9.404065983981610977e-01,
            9.404065984978412507e-01,
            4.693200937560261243e-01,
            -1.943174388128667385e00,
            4.974181326312798745e-02,
            3.038495816127385396e00,
        ]
    ),
    scaled_genome=None,
    log_likelihood=None,
    fitness_values=[],
)

IRIS_ICA_SOLUTION = Solution(
    fitness=0.0,
    genome=np.array(
        [
            0.23419558,
            0.41920456,
            0.34659985,
            0.312949,
            0.33587196,
            0.10171662,
            -1.1100256,
            -0.3637117,
            1.2932061,
        ]
    ),
    scaled_genome=None,
    log_likelihood=None,
    fitness_values=[],
)

CHROMATOGRAM_TIME_SOLUTION = Solution(
    fitness=0.0,
    genome=np.array(
        [
            0.05448153,
            0.32027948,
            0.14398287,
            0.32264763,
            0.15860853,
            0.01270635,
            0.0207545,
            0.02109946,
            0.02603111,
            0.02546384,
            1.5195698,
            1.6282822,
            1.7513955,
            1.9881668,
            2.2600677,
        ]
    ),
    scaled_genome=None,
    log_likelihood=None,
    fitness_values=[],
)

ATMOSPHERE_DATA_SOLUTION = Solution(
    fitness=0.0,
    genome=np.array(
        [
            2.00150460e-01,
            1.00061335e-01,
            9.97321308e-02,
            2.99930423e-01,
            3.00125659e-01,
            4.84303385e-02,
            6.70865625e-02,
            1.99107919e-02,
            6.11717999e-02,
            9.92694199e-02,
            -2.39771986e00,
            -9.99630511e-01,
            3.23275104e-04,
            1.00026751e00,
            2.40160060e00,
        ]
    ),
    scaled_genome=None,
    log_likelihood=None,
    fitness_values=[],
)

DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = [
    Dataset(
        data=read_dataset_from_csv(os.path.join(DIR, "./data/truck_driving_data.csv")),
        name="truck_driving_data",
        nr_of_modes=3,
        solution=TRUCK_DRIVING_SOLUTION,
    ),
    Dataset(
        data=read_dataset_from_csv(os.path.join(DIR, "./data/mixture3.csv")),
        name="mixture3",
        nr_of_modes=3,
        solution=MIXTURE3_SOLUTION,
    ),
    Dataset(
        data=read_dataset_from_csv(os.path.join(DIR, "./data/textbook_1k.csv")),
        name="textbook_data",
        nr_of_modes=3,
        solution=TEXTBOOK_DATA_SOLUTION,
    ),
    Dataset(
        data=read_dataset_from_csv(os.path.join(DIR, "./data/iris_ica.csv")),
        name="iris_ica",
        nr_of_modes=3,
        solution=IRIS_ICA_SOLUTION,
    ),
    Dataset(
        data=read_dataset_from_csv(os.path.join(DIR, "./data/chromatogram_time.csv")),
        name="chromatogram_time",
        nr_of_modes=5,
        solution=CHROMATOGRAM_TIME_SOLUTION,
    ),
    Dataset(
        data=read_dataset_from_csv(os.path.join(DIR, "./data/atmosphere_data.csv")),
        name="atmosphere_data",
        nr_of_modes=5,
        solution=ATMOSPHERE_DATA_SOLUTION,
    ),
]

# SYNTHETIC_DATASETS = [
#     Dataset(
#         data=read_dataset_from_csv("../data/synthetic_1.csv"),
#         name="synthetic_1",
#         nr_of_modes=4,
#         solution=None,
#     ),
#     Dataset(
#         data=read_dataset_from_csv("../data/synthetic_2.csv"),
#         name="synthetic_2",
#         nr_of_modes=4,
#         solution=None,
#     ),
#     Dataset(
#         data=read_dataset_from_csv("../data/synthetic_3.csv"),
#         name="synthetic_3",
#         nr_of_modes=4,
#         solution=None,
#     ),
#     Dataset(
#         data=read_dataset_from_csv("../data/synthetic_4.csv"),
#         name="synthetic_4",
#         nr_of_modes=6,
#         solution=None,
#     ),
#     Dataset(
#         data=read_dataset_from_csv("../data/synthetic_5.csv"),
#         name="synthetic_5",
#         nr_of_modes=6,
#         solution=None,
#     ),
# ]
