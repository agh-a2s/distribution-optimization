from ..datasets import DATASETS
from ..solver.de import iLSHADESolver
from ..solver.ga import GASolver
from .evaluator import ProblemEvaluator

if __name__ == "__main__":
    from ..constrained_landscape_analysis import ConstraintHandlingTechnique, DistributionOptimizationProblem
    from ..problem import GaussianMixtureProblem, ScaledGaussianMixtureProblem

    solvers = [iLSHADESolver(), GASolver()]
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
    results[0].to_csv("death_penalty_results.csv", index=False)
