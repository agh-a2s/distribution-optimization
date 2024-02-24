import pytest
import numpy as np
from distribution_optimization_py.scale import (
    reals_to_simplex,
    simplex_to_reals,
    reals_to_reals_with_offset,
    reals_with_offset_to_reals,
    scale_linearly,
)

N = 10


def test_offset_scale_example():
    x = np.array([0.12, 3.64, 107.24])
    x_with_offset = np.array([0.12, 3.76, 111.0])
    assert np.isclose(reals_to_reals_with_offset(x), x_with_offset).all()
    assert np.isclose(reals_with_offset_to_reals(x_with_offset), x).all()
    assert np.isclose(
        reals_with_offset_to_reals(reals_to_reals_with_offset(x)), x
    ).all()


def test_simplex_scale_example():
    x = np.array([0.9, 0.2, 1.0])
    x_simplex = np.array([0.1, 0.72, 0.0])
    assert np.isclose(reals_to_simplex(x), x_simplex).all()
    assert np.isclose(simplex_to_reals(x_simplex), x).all()
    assert np.isclose(reals_to_simplex(simplex_to_reals(x)), x).all()


def test_composition_of_offset_scale():
    x = np.random.rand(N)
    assert np.isclose(
        reals_with_offset_to_reals(reals_to_reals_with_offset(x)), x
    ).all()


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
