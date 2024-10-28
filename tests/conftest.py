import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def truck_driving_data() -> np.ndarray:
    truck_driving_data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "distribution_optimization_py",
        "data",
        "truck_driving_data.csv",
    )
    return pd.read_csv(truck_driving_data_path)["value"].values


@pytest.fixture
def truck_driving_nr_of_modes() -> int:
    return 3
