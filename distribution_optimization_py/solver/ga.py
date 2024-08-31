import random

import numpy as np
from leap_ec.representation import Representation
from pyhms.core.problem import EvalCutoffProblem, FunctionProblem

from ..evaluator.problem_wrapper import ProblemMonitor
from ..ga_style_operators import GAStyleEA
from ..problem import GaussianMixtureProblem, ScaledGaussianMixtureProblem
from .protocol import CONVERGENCE_PLOT_STEP_SIZE, Solution, Solver

GA_CONFIG = {"k_elites": 5, "p_mutation": 0.8, "p_crossover": 0.9, "pop_size": 50}


class GASolver(Solver):
    def __call__(
        self,
        problem: GaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ):
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)
        bounds = np.column_stack((problem.lower, problem.upper))
        function_problem = FunctionProblem(problem, bounds=bounds, maximize=False)
        eval_cutoff_problem = ProblemMonitor(function_problem, max_n_evals, CONVERGENCE_PLOT_STEP_SIZE)
        ea = GAStyleEA(
            generations=1,
            problem=eval_cutoff_problem,
            bounds=bounds,
            pop_size=GA_CONFIG["pop_size"],
            k_elites=GA_CONFIG["k_elites"],
            p_mutation=GA_CONFIG["p_mutation"],
            p_crossover=GA_CONFIG["p_crossover"],
            representation=Representation(initialize=lambda: problem.initialize()),
        )
        all_populations = []
        population = None
        iterations_count = int(max_n_evals / GA_CONFIG["pop_size"])
        for _ in range(iterations_count):
            population = ea.run(population)
            all_populations.extend(population)
        best_solution = max(all_populations)
        scaled_genome = (
            problem.reals_to_internal(best_solution.genome)
            if isinstance(problem, ScaledGaussianMixtureProblem)
            else None
        )
        return Solution(
            fitness=best_solution.fitness,
            genome=best_solution.genome,
            scaled_genome=scaled_genome,
            log_likelihood=problem.log_likelihood(best_solution.genome),
            fitness_values=np.array(eval_cutoff_problem._problem_values),
        )
