import numpy as np
import pytest
from distribution_optimization_py.problem import GaussianMixtureProblem, LinearlyScaledGaussianMixtureProblem
from distribution_optimization_py.scale import scale_linearly

SOLUTIONS_WITH_FITNESS_VALUES_CALCULATED_IN_R = [
    (
        np.array(
            [
                3.502513144058477246e-01,
                4.444053269729363320e-01,
                2.053433586212158879e-01,
                6.831937284481902850e01,
                7.682604015993844371e01,
                7.704103848669365107e01,
                6.269724765376414410e01,
                3.691086274604800792e02,
                5.883345529323370329e02,
            ]
        ),
        0.07255805,
    ),
    (
        np.array(
            [
                3.498850127320053405e-01,
                4.595568279506902498e-01,
                1.905581593173044375e-01,
                6.742252858653138503e01,
                7.702753463847007254e01,
                7.186662310527424324e01,
                6.222881666907016296e01,
                3.714437558479370409e02,
                5.934439691096640672e02,
            ]
        ),
        0.07139837,
    ),
    (
        np.array(
            [
                3.075428052026262793e-01,
                4.627173375449771808e-01,
                2.297398572523966509e-01,
                5.427775754878306458e01,
                7.676803935282599411e01,
                7.704566720584965367e01,
                7.709290920958379445e01,
                3.652133997775118246e02,
                5.826579146728188334e02,
            ]
        ),
        0.08524585,
    ),
    (
        np.array(
            [
                4.074401545468814279e-01,
                4.132368200474662578e-01,
                1.793230254056523421e-01,
                6.989461266232507342e01,
                7.657531779132681038e01,
                7.335801161707527740e01,
                6.307291031967541528e01,
                3.690987918858785974e02,
                5.814822659279673189e02,
            ]
        ),
        0.08687456,
    ),
    (
        np.array(
            [
                3.285277826889702046e-01,
                4.701788897884126017e-01,
                2.012933275226171104e-01,
                6.292241875295022169e01,
                7.100255958580295612e01,
                6.382771878738402194e01,
                9.500577878659012754e01,
                3.815728015395030184e02,
                5.992884409887415131e02,
            ]
        ),
        0.1209622,
    ),
    (
        np.array(
            [
                3.706583299593614322e-01,
                4.555518543783356189e-01,
                1.737898156623029489e-01,
                7.701107761018657527e01,
                7.612238330179185652e01,
                5.626822589450155476e01,
                6.277913877839937840e01,
                3.774866790567190264e02,
                5.993886356202309571e02,
            ]
        ),
        0.07721006,
    ),
]


@pytest.mark.parametrize(
    "x,expected_fitness_value",
    SOLUTIONS_WITH_FITNESS_VALUES_CALCULATED_IN_R,
)
def test_gaussian_mixture_problem(
    x: np.ndarray,
    expected_fitness_value: float,
    truck_driving_data: np.ndarray,
    truck_driving_nr_of_modes: int,
):
    problem = GaussianMixtureProblem(truck_driving_data, truck_driving_nr_of_modes)
    assert np.isclose(problem(x), expected_fitness_value, atol=1e-4)


@pytest.mark.parametrize(
    "x,expected_fitness_value",
    SOLUTIONS_WITH_FITNESS_VALUES_CALCULATED_IN_R,
)
def test_linearly_scaled_gaussian_mixture_problem(
    x: np.ndarray,
    expected_fitness_value: float,
    truck_driving_data: np.ndarray,
    truck_driving_nr_of_modes: int,
):
    LOWER = -5
    UPPER = 5
    problem = LinearlyScaledGaussianMixtureProblem(
        truck_driving_data, truck_driving_nr_of_modes, lower=LOWER, upper=UPPER
    )
    scaled_x = scale_linearly(x, problem.data_lower, problem.data_upper, problem.lower, problem.upper)
    assert np.isclose(problem(scaled_x), expected_fitness_value, atol=1e-4)


def test_gaussian_mixture_problem_bounds(truck_driving_data: np.ndarray, truck_driving_nr_of_modes: int):
    problem = GaussianMixtureProblem(truck_driving_data, truck_driving_nr_of_modes)
    lower, upper = problem.get_bounds()
    assert lower.shape == (truck_driving_nr_of_modes * 3,)
    assert upper.shape == (truck_driving_nr_of_modes * 3,)
    assert np.all(lower < upper)
    assert (lower[:truck_driving_nr_of_modes] == 0.03).all()
    assert (upper[:truck_driving_nr_of_modes] == 1.00).all()
    assert (
        lower[truck_driving_nr_of_modes : 2 * truck_driving_nr_of_modes]
        == 0.001 * (truck_driving_data.max() - truck_driving_data.min())
    ).all()
    assert (
        upper[truck_driving_nr_of_modes : 2 * truck_driving_nr_of_modes]
        == 0.1 * (truck_driving_data.max() - truck_driving_data.min())
    ).all()
    assert (lower[2 * truck_driving_nr_of_modes :] == truck_driving_data.min()).all()
    assert (upper[2 * truck_driving_nr_of_modes :] == truck_driving_data.max()).all()
