from ..datasets import DATASETS
from ..problem import GaussianMixtureProblem
from ..solver.cs import CSDESolver, CSSolver
from ..solver.de import iLSHADESolver
from ..solver.ga import GASolver
from .evaluator import ProblemEvaluator

if __name__ == "__main__":
    solvers = [iLSHADESolver(), GASolver(), CSSolver(), CSDESolver()]
    problems = [
        # DistributionOptimizationProblem(
        #     dataset.data,
        #     dataset.nr_of_modes,
        #     method=ConstraintHandlingTechnique.DEATH_PENALTY,
        #     id=dataset.name,
        #     use_repair=True,
        # )
        GaussianMixtureProblem(
            dataset.data,
            dataset.nr_of_modes,
            id=dataset.name,
        )
        for dataset in DATASETS
    ]
    evaluator = ProblemEvaluator(solvers, problems)
    results = evaluator()
    results[0].to_csv("cs_results.csv", index=False)
