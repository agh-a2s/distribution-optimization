# import random

# import numpy as np
# from cma import CMAEvolutionStrategy

# from ..problem import GaussianMixtureProblem, ScaledGaussianMixtureProblem
# from .protocol import Solution, Solver

# class CMAESSolver(Solver):
#     def __call__(
#         self,
#         problem: GaussianMixtureProblem,
#         max_n_evals: int,
#         random_state: int | None = None,
#     ) -> Solution:
#         if random_state:
#             np.random.seed(random_state)
#             random.seed(random_state)
#         x0 = problem.initialize()
#         options = {
#             "bounds": [problem.lower, problem.upper],
#             "verbose": -9,
#             "maxfevals": max_n_evals,
#             "seed": random_state,
#         }
#         options_from_config = {key: value for key, value in self.config.items() if key in CONFIG_FIELDS_OPTIONS}
#         options |= options_from_config
#         res = fmin(
#             problem,
#             x0,
#             sigma0=self.config["sigma0"],
#             options=options,
#             bipop=True,
#             restart_from_best=self.config["restart_from_best"],
#             restarts=self.config["restarts"],
#             incpopsize=self.config["incpopsize"],
#         )
#         return Solution(res[0], res[1], problem, res)

#     @property
#     def configspace(self) -> ConfigurationSpace:
#         return ConfigurationSpace(
#             {
#                 "popsize": (10, 100),
#                 "sigma0": (0.5, 7.5),
#                 "incpopsize": (1, 5),
#                 "tolfun": (1e-15, 1e-8),
#             }
#         )


# class CMAESSolver(Solver):
#     def __call__(
#         self,
#         problem: GaussianMixtureProblem,
#         max_n_evals: int,
#         random_state: int | None = None,
#     ) -> np.ndarray:
#         if random_state:
#             np.random.seed(random_state)
#             random.seed(random_state)
#         x0 = problem.initialize()
#         sigma = 1.0
#         cma = CMAEvolutionStrategy(
#             x0,
#             sigma,
#             inopts={
#                 "bounds": [problem.lower, problem.upper],
#                 "verbose": -9,
#                 "seed": random_state,
#                 "maxfevals": max_n_evals,
#             },
#         )
#         cma.optimize(problem)
#         solution, fitness = cma.xbest, cma.result
#         scaled_genome = (
#             problem.reals_to_internal(solution) if isinstance(problem, ScaledGaussianMixtureProblem) else None
#         )
#         Solution(
#             fitness=fitness,
#             genome=solution,
#             scaled_genome=None,
#             log_likelihood=problem.log_likelihood(cma.xbest),
#             fitness_values=np.array(eval_cutoff_problem._problem_values),
#         )
