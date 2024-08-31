import random
from typing import Callable

import numpy as np
from pyhms.core.problem import EvalCutoffProblem, FunctionProblem

from ..evaluator.problem_wrapper import ProblemMonitor
from ..problem import GaussianMixtureProblem, ScaledGaussianMixtureProblem
from ..pyade.de import apply as de
from ..pyade.de import get_default_params as get_default_params_de
from ..pyade.ilshade import apply as ilshade
from ..pyade.ilshade import get_default_params as get_default_params_ilshade
from ..pyade.jade import apply as jade
from ..pyade.jade import get_default_params as get_default_params_jade
from ..pyade.lshade import apply as lshade
from ..pyade.lshade import get_default_params as get_default_params_lshade
from ..pyade.shade import apply as shade
from ..pyade.shade import get_default_params as get_default_params_shade
from .protocol import CONVERGENCE_PLOT_STEP_SIZE, Solution, Solver


class DEBaseSolver(Solver):
    def __init__(self, get_default_params: Callable, apply: Callable):
        self.get_default_params = get_default_params
        self.apply = apply

    def __call__(
        self,
        problem: GaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)

        def init_population(population_size: int, individual_size: int, bounds: np.ndarray | list) -> np.ndarray:
            return np.array([problem.initialize() for _ in range(population_size)])

        bounds = np.column_stack((problem.lower, problem.upper))
        function_problem = FunctionProblem(problem, bounds=bounds, maximize=False)
        eval_cutoff_problem = ProblemMonitor(function_problem, max_n_evals, CONVERGENCE_PLOT_STEP_SIZE)
        params = self.get_default_params(dim=problem.lower.shape[0])
        params["bounds"] = np.column_stack([problem.lower, problem.upper])
        params["func"] = lambda x: eval_cutoff_problem.evaluate(x)
        params["max_evals"] = max_n_evals
        params["init_population"] = init_population
        solution, fitness = self.apply(**params)
        scaled_genome = (
            problem.reals_to_internal(solution) if isinstance(problem, ScaledGaussianMixtureProblem) else None
        )
        return Solution(
            fitness=fitness,
            genome=solution,
            scaled_genome=scaled_genome,
            log_likelihood=problem.log_likelihood(solution),
            fitness_values=np.array(eval_cutoff_problem._problem_values),
        )


class DESolver(DEBaseSolver):
    def __init__(self):
        super().__init__(get_default_params=get_default_params_de, apply=de)


class SHADESolver(DEBaseSolver):
    def __init__(self):
        super().__init__(get_default_params=get_default_params_shade, apply=shade)


class iLSHADESolver(DEBaseSolver):
    def __init__(self):
        super().__init__(get_default_params=get_default_params_ilshade, apply=ilshade)


class LSHADESolver(DEBaseSolver):
    def __init__(self):
        super().__init__(get_default_params=get_default_params_lshade, apply=lshade)


class JADESolver(DEBaseSolver):
    def __init__(self):
        super().__init__(get_default_params=get_default_params_jade, apply=jade)
