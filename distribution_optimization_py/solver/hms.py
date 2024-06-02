import random

import numpy as np
from leap_ec.representation import Representation
from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.core.problem import EvalCutoffProblem, FunctionProblem
from pyhms.sprout.sprout_filters import DemeLimit, LevelLimit, NBC_FarEnough
from pyhms.sprout.sprout_generators import NBC_Generator
from pyhms.sprout.sprout_mechanisms import SproutMechanism
from pyhms.stop_conditions.gsc import SingularProblemEvalLimitReached
from pyhms.stop_conditions.usc import DontStop, MetaepochLimit
from pyhms.tree import DemeTree

from ..ga_style_operators import GAStyleEA
from ..problem import ScaledGaussianMixtureProblem
from .protocol import Solver

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
        problem: ScaledGaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        if random_state:
            random.seed(random_state)
            np.random.seed(random_state)
        options = {"random_seed": random_state} if random_state else {}
        function_problem = FunctionProblem(problem, maximize=False)
        cutoff_problem = EvalCutoffProblem(function_problem, max_n_evals)
        bounds = np.column_stack((problem.lower, problem.upper))
        levels_config = [
            EALevelConfig(
                ea_class=GAStyleEA,
                generations=HMS_CONFIG["generations1"],
                problem=cutoff_problem,
                bounds=bounds,
                pop_size=HMS_CONFIG["pop1"],
                lsc=DontStop(),
                k_elites=HMS_CONFIG["k_elites1"],
                representation=Representation(initialize=problem.initialize),
                p_mutation=HMS_CONFIG["p_mutation1"],
                p_crossover=HMS_CONFIG["p_crossover1"],
                use_warm_start=HMS_CONFIG["use_warm_start"],
            ),
            CMALevelConfig(
                problem=cutoff_problem,
                bounds=bounds,
                lsc=MetaepochLimit(HMS_CONFIG["metaepoch2"]),
                sigma0=HMS_CONFIG["sigma2"],
                generations=HMS_CONFIG["generations2"],
            ),
        ]

        sprout = SproutMechanism(
            NBC_Generator(HMS_CONFIG["nbc_cut"], HMS_CONFIG["nbc_trunc"]),
            [NBC_FarEnough(HMS_CONFIG["nbc_far"], 2), DemeLimit(1)],
            [LevelLimit(HMS_CONFIG["level_limit"])],
        )
        gsc = SingularProblemEvalLimitReached(max_n_evals)
        tree_config = TreeConfig(levels_config, gsc, sprout, options)
        deme_tree = DemeTree(tree_config)
        deme_tree.run()
        return deme_tree.best_individual.genome
