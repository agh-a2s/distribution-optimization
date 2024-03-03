import numpy as np
from distribution_optimization_py.gaussian_mixture import GaussianMixture

RANDOM_STATE = 42
MAX_N_EVALS = 10000


def test_gaussian_mixture_with_hms(truck_driving_data: np.ndarray, truck_driving_nr_of_modes: int):
    gmm = GaussianMixture(
        n_components=truck_driving_nr_of_modes,
        algorithm="HMS",
        random_state=RANDOM_STATE,
        max_n_evals=MAX_N_EVALS,
    )
    gmm.fit(truck_driving_data)
    assert gmm._weights is not None
    assert gmm._sds is not None
    assert gmm._means is not None


def test_gaussian_mixture_with_ga(truck_driving_data: np.ndarray, truck_driving_nr_of_modes: int):
    gmm = GaussianMixture(
        n_components=truck_driving_nr_of_modes,
        algorithm="GA",
        random_state=RANDOM_STATE,
        max_n_evals=MAX_N_EVALS,
    )
    gmm.fit(truck_driving_data)
    assert gmm._weights is not None
    assert gmm._sds is not None
    assert gmm._means is not None


def test_gaussian_mixture_with_cma_es(truck_driving_data: np.ndarray, truck_driving_nr_of_modes: int):
    gmm = GaussianMixture(
        n_components=truck_driving_nr_of_modes,
        algorithm="CMA-ES",
        random_state=RANDOM_STATE,
        max_n_evals=MAX_N_EVALS,
    )
    gmm.fit(truck_driving_data)
    assert gmm._weights is not None
    assert gmm._sds is not None
    assert gmm._means is not None
