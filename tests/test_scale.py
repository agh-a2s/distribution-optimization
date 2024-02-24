import numpy as np
from distribution_optimization_py.problem import ScaledGaussianMixtureProblem
from distribution_optimization_py.scale import (
    full_simplex_to_reals,
    reals_to_full_simplex,
    reals_to_reals_with_offset,
    reals_to_simplex,
    reals_with_offset_to_reals,
    scale_linearly,
    simplex_to_reals,
)

N = 10


def test_offset_scale_example():
    x = np.array([0.12, 3.64, 107.24])
    x_with_offset = np.array([0.12, 3.76, 111.0])
    assert np.isclose(reals_to_reals_with_offset(x), x_with_offset).all()
    assert np.isclose(reals_with_offset_to_reals(x_with_offset), x).all()
    assert np.isclose(reals_with_offset_to_reals(reals_to_reals_with_offset(x)), x).all()


def test_full_simplex_scale_example():
    x = np.array([0.9, 0.2, 1.0])
    x_full_simplex = np.array([0.1, 0.72, 0.0])
    assert np.isclose(reals_to_full_simplex(x), x_full_simplex).all()
    assert np.isclose(full_simplex_to_reals(x_full_simplex), x).all()
    assert np.isclose(full_simplex_to_reals(reals_to_full_simplex(x)), x).all()


def test_simplex_scale_example():
    x = np.array([0.9, 0.2, 1.0])
    x_simplex = np.array([0.1, 0.72, 0.0, 0.18])
    assert np.isclose(reals_to_simplex(x), x_simplex).all()
    assert np.isclose(simplex_to_reals(x_simplex), x).all()
    assert np.isclose(simplex_to_reals(reals_to_simplex(x)), x).all()


def test_composition_of_offset_scale():
    x = np.random.rand(N)
    assert np.isclose(reals_with_offset_to_reals(reals_to_reals_with_offset(x)), x).all()


def test_composition_of_full_simplex_scale():
    x = np.random.rand(N)
    assert np.isclose(full_simplex_to_reals(reals_to_full_simplex(x)), x).all()


def test_composition_of_simplex_scale():
    x = np.random.rand(N)
    assert np.isclose(simplex_to_reals(reals_to_simplex(x)), x).all()


def test_composition_of_linear_scale():
    source_lower = -5
    source_upper = 5
    target_lower = 0
    target_upper = 1
    x = np.random.rand(N)
    x_scaled = scale_linearly(x, source_lower, source_upper, target_lower, target_upper)
    assert np.isclose(
        scale_linearly(
            x_scaled,
            target_lower,
            target_upper,
            source_lower,
            source_upper,
        ),
        x,
    ).all()


def test_scaled_problem(truck_driving_data: np.ndarray):
    nr_of_modes = 3
    problem = ScaledGaussianMixtureProblem(truck_driving_data, nr_of_modes)
    x = np.array([0.4, 0.3, 0.1, 0.3, 0.2, 0.4, 0.7, 0.2])
    internal_x = np.array([0.6, 0.28, 0.12, 8.398014, 23.653122, 16.025568, 462.276, 554.7312, 727.31424])
    assert np.isclose(problem.reals_to_internal(x), internal_x).all()
    assert np.isclose(problem.internal_to_reals(internal_x), x).all()
