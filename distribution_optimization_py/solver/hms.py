import random

import numpy as np
from pyhms.config import CMALevelConfig, EALevelConfig, SHADELevelConfig, TreeConfig
from pyhms.core.problem import FunctionProblem
from pyhms.sprout import DemeLimit, FarEnoughFromSeeds, LevelLimit, NBC_FarEnough, NBC_Generator, SproutMechanism
from pyhms.stop_conditions.gsc import SingularProblemEvalLimitReached
from pyhms.stop_conditions.usc import DontStop, MetaepochLimit
from pyhms.tree import DemeTree

from ..evaluator.problem_wrapper import ProblemMonitor
from ..ga_style_operators import GAStyleEA
from ..problem import GaussianMixtureProblem, ScaledGaussianMixtureProblem
from .protocol import CONVERGENCE_PLOT_STEP_SIZE, Solution, Solver

HMS_CONFIG = {
    "nbc_cut": 3.7963357241892517,
    "nbc_trunc": 0.7854441324546146,
    "nbc_far": 3.698180094782821,
    "level_limit": 4,
    "pop1": 49,
    "p_mutation1": 0.21707882439727,
    "p_crossover1": 0.20534306545391615,
    "k_elites1": 1,
    "generations1": 5,
    "generations2": 26,
    "metaepoch2": 32,
    "sigma2": 0.11230013378964185,
    "use_warm_start": False,
}


class HMSSolver(Solver):
    def __call__(
        self,
        problem: GaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        if random_state:
            random.seed(random_state)
            np.random.seed(random_state)
        options = {"random_seed": random_state} if random_state else {}
        bounds = np.column_stack((problem.lower, problem.upper))
        function_problem = FunctionProblem(problem, bounds=bounds, maximize=False)
        cutoff_problem = ProblemMonitor(function_problem, max_n_evals, CONVERGENCE_PLOT_STEP_SIZE)

        # def improved_initialize():
        #     r = np.random.rand()
        #     counter = 0
        #     if r < 0.6:
        #         sample = problem.initialize()
        #         cutoff_problem._n_evals += 1
        #         while problem(sample) == INFINITY and counter < 5:
        #             cutoff_problem._n_evals += 1
        #             sample = np.random.uniform(problem.lower, problem.upper)
        #             counter += 1
        #         return sample
        #     elif r < 0.8:
        #         sample = problem.initialize()
        #         cutoff_problem._n_evals += 1
        #         while problem(sample) == INFINITY and counter < 5:
        #             cutoff_problem._n_evals += 1
        #             sample = problem.initialize()
        #             counter += 1
        #         return sample
        #     else:
        #         sample = problem.initialize_warm_start()
        #         cutoff_problem._n_evals += 1
        #         while problem(sample) == INFINITY and counter < 5:
        #             cutoff_problem._n_evals += 1
        #             problem.initialize_warm_start()
        #             counter += 1
        #         return sample

        levels_config = [
            EALevelConfig(
                ea_class=GAStyleEA,
                generations=HMS_CONFIG["generations1"],
                problem=cutoff_problem,
                pop_size=HMS_CONFIG["pop1"],
                lsc=DontStop(),
                k_elites=HMS_CONFIG["k_elites1"],
                p_mutation=HMS_CONFIG["p_mutation1"],
                p_crossover=HMS_CONFIG["p_crossover1"],
                use_warm_start=HMS_CONFIG["use_warm_start"],
                initialize=problem.initialize,
            ),
            CMALevelConfig(
                problem=cutoff_problem,
                lsc=MetaepochLimit(HMS_CONFIG["metaepoch2"]),
                sigma0=None,
                generations=HMS_CONFIG["generations2"],
            ),
        ]

        sprout = SproutMechanism(
            NBC_Generator(HMS_CONFIG["nbc_cut"], HMS_CONFIG["nbc_trunc"]),
            [
                NBC_FarEnough(HMS_CONFIG["nbc_far"], 2, False),
                DemeLimit(1),
                FarEnoughFromSeeds(0.25),
            ],
            [LevelLimit(HMS_CONFIG["level_limit"])],
        )
        gsc = SingularProblemEvalLimitReached(max_n_evals)
        tree_config = TreeConfig(levels_config, gsc, sprout, options)
        deme_tree = DemeTree(tree_config)
        deme_tree.run()
        best_genome = deme_tree.best_individual.genome
        scaled_genome = (
            problem.reals_to_internal(best_genome) if isinstance(problem, ScaledGaussianMixtureProblem) else None
        )
        return Solution(
            fitness=deme_tree.best_individual.fitness,
            genome=best_genome,
            scaled_genome=scaled_genome,
            log_likelihood=problem.log_likelihood(best_genome),
            fitness_values=np.array(cutoff_problem._problem_values),
        )


class HMSSHADESolver(Solver):
    def __call__(
        self,
        problem: GaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        if random_state:
            random.seed(random_state)
            np.random.seed(random_state)
        options = {"random_seed": random_state} if random_state else {}
        bounds = np.column_stack((problem.lower, problem.upper))
        function_problem = FunctionProblem(problem, bounds=bounds, maximize=False)
        cutoff_problem = ProblemMonitor(function_problem, max_n_evals, CONVERGENCE_PLOT_STEP_SIZE)

        levels_config = [
            SHADELevelConfig(
                generations=HMS_CONFIG["generations1"],
                problem=cutoff_problem,
                pop_size=100,
                lsc=DontStop(),
                initialize=problem.initialize,
                memory_size=200,
            ),
            CMALevelConfig(
                problem=cutoff_problem,
                lsc=MetaepochLimit(HMS_CONFIG["metaepoch2"]),
                sigma0=None,
                generations=HMS_CONFIG["generations2"],
            ),
        ]

        sprout = SproutMechanism(
            NBC_Generator(HMS_CONFIG["nbc_cut"], HMS_CONFIG["nbc_trunc"]),
            [
                NBC_FarEnough(HMS_CONFIG["nbc_far"], 2, False),
                DemeLimit(1),
                FarEnoughFromSeeds(0.25),
            ],
            [LevelLimit(HMS_CONFIG["level_limit"])],
        )
        gsc = SingularProblemEvalLimitReached(max_n_evals)
        tree_config = TreeConfig(levels_config, gsc, sprout, options)
        deme_tree = DemeTree(tree_config)
        deme_tree.run()
        best_genome = deme_tree.best_individual.genome
        scaled_genome = (
            problem.reals_to_internal(best_genome) if isinstance(problem, ScaledGaussianMixtureProblem) else None
        )
        return Solution(
            fitness=deme_tree.best_individual.fitness,
            genome=best_genome,
            scaled_genome=scaled_genome,
            log_likelihood=problem.log_likelihood(best_genome),
            fitness_values=np.array(cutoff_problem._problem_values),
        )
