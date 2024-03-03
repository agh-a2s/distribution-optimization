import random
from enum import Enum

import numpy as np
from leap_ec.problem import FunctionProblem
from leap_ec.representation import Representation
from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.problem import EvalCutoffProblem
from pyhms.sprout.sprout_filters import DemeLimit, LevelLimit, NBC_FarEnough
from pyhms.sprout.sprout_generators import NBC_Generator
from pyhms.sprout.sprout_mechanisms import SproutMechanism
from pyhms.stop_conditions.gsc import singular_problem_eval_limit_reached
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from pyhms.tree import DemeTree

from .ga_style_operators import GAStyleEA
from .problem import ScaledGaussianMixtureProblem
from .utils import mixture_probability


class OptimizationAlgorithm(str, Enum):
    HMS = "HMS"
    GA = "GA"


HMS_CONFIG = {
    "nbc_cut": 3.599711774779532,
    "nbc_trunc": 0.8995919493637037,
    "nbc_far": 3.490948090512303,
    "level_limit": 2,
    "pop1": 22,
    "p_mutation1": 0.8574418205037987,
    "p_crossover1": 0.6800376375407469,
    "k_elites1": 1,
    "generations1": 2,
    "generations2": 25,
    "metaepoch2": 39,
    "sigma2": 1.3620228921127224,
}


class GaussianMixture:
    def __init__(
        self,
        n_components: int = 1,
        max_n_evals: int = 10000,
        random_state: int | None = None,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.HMS,
    ):
        self._n_components = n_components
        self._max_n_evals = max_n_evals
        self._random_state = random_state
        self._algorithm = algorithm
        self._weights: np.ndarray | None = None
        self._sds: np.ndarray | None = None
        self._means: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        X = self._validate_data(X)
        if self._random_state:
            random.seed(self._random_state)
            np.random.seed(self._random_state)
        options = {"random_seed": self._random_state} if self._random_state else {}
        problem = ScaledGaussianMixtureProblem(X, self._n_components)
        function_problem = FunctionProblem(problem, maximize=False)
        cutoff_problem = EvalCutoffProblem(function_problem, self._max_n_evals)
        bounds = np.column_stack((problem.lower, problem.upper))
        levels_config = [
            EALevelConfig(
                ea_class=GAStyleEA,
                generations=HMS_CONFIG["generations1"],
                problem=cutoff_problem,
                bounds=bounds,
                pop_size=HMS_CONFIG["pop1"],
                lsc=dont_stop(),
                k_elites=HMS_CONFIG["k_elites1"],
                representation=Representation(initialize=problem.initialize),
                p_mutation=HMS_CONFIG["p_mutation1"],
                p_crossover=HMS_CONFIG["p_crossover1"],
            ),
            CMALevelConfig(
                problem=cutoff_problem,
                bounds=bounds,
                lsc=metaepoch_limit(HMS_CONFIG["metaepoch2"]),
                sigma0=HMS_CONFIG["sigma2"],
                generations=HMS_CONFIG["generations2"],
            ),
        ]
        sprout = SproutMechanism(
            NBC_Generator(HMS_CONFIG["nbc_cut"], HMS_CONFIG["nbc_trunc"]),
            [NBC_FarEnough(HMS_CONFIG["nbc_far"], 2), DemeLimit(1)],
            [LevelLimit(HMS_CONFIG["level_limit"])],
        )
        gsc = singular_problem_eval_limit_reached(self._max_n_evals)
        tree_config = TreeConfig(levels_config, gsc, sprout, options)
        deme_tree = DemeTree(tree_config)
        deme_tree.run()
        best_solution = deme_tree.best_individual
        scaled_solution = problem.reals_to_internal(best_solution.genome)
        self._weights = scaled_solution[: self._n_components]
        self._sds = scaled_solution[self._n_components : 2 * self._n_components]
        self._means = scaled_solution[2 * self._n_components :]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = [mixture_probability(x, self._means, self._sds, self._weights) for x in X]
        return np.array(probabilities)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if X.ndim != 1:
            raise ValueError("X must be a 1D array")
        return X
